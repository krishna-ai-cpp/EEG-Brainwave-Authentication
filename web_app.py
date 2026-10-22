"""Upload-based browser adapter. The research pipeline in hello.py is unchanged."""
import base64
import csv
import hashlib
import json
import math
import re
import os
from pathlib import Path
import secrets
import selectors
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parent
JOBS = {}
LOCK = threading.Lock()
BUSY = threading.Semaphore(1)
MAX_FILE = 8 * 1024 * 1024
MAX_TOTAL = 20 * 1024 * 1024


def execute_original(folder, output, settings, temp, env):
    """Answer the original script's conditional prompts without changing its logic."""
    initial = '\n'.join([str(folder), str(output), str(settings['low']),
                         str(settings['high']), str(settings['window']), '150']) + '\n'
    with (Path(temp) / 'stderr.txt').open('w+') as errors:
        process = subprocess.Popen([sys.executable, '-u', '-B', str(ROOT / 'hello.py')],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
                                   cwd=temp, env=env)
        chunks = bytearray()
        answered = set()
        try:
            process.stdin.write(initial.encode())
            process.stdin.flush()
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                deadline = time.monotonic() + 180
                while selector.get_map():
                    if time.monotonic() > deadline:
                        raise RuntimeError('The analysis exceeded three minutes. Try fewer or shorter recordings.')
                    for key, _ in selector.select(timeout=1):
                        chunk = key.fileobj.read1(65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        chunks.extend(chunk)
                        text = chunks.decode(errors='replace')
                        for marker, answer in [('Type yes to continue:', settings['reviewed']),
                                               ('Build the CNN candidate too? Type yes or press Enter to finish:', settings['cnn'])]:
                            if marker in text and marker not in answered:
                                process.stdin.write(b'yes\n' if answer else b'no\n')
                                process.stdin.flush()
                                answered.add(marker)
            process.wait(timeout=5)
            errors.seek(0)
            return SimpleNamespace(returncode=process.returncode, stdout=chunks.decode(errors='replace'), stderr=errors.read())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdin.close()
            process.stdout.close()


def run_pipeline(job, settings, uploads):
    try:
        with tempfile.TemporaryDirectory(prefix='eeg-upload-') as temp:
            folder = Path(temp) / 'recordings'
            folder.mkdir()
            provenance = []
            for index, item in enumerate(uploads):
                job.update(stage=f'Preparing uploaded recording {index + 1} of {len(uploads)}', progress=10 + index * 3)
                (folder / f'recording_{index:03}.edf').write_bytes(item['bytes'])
                provenance.append({'recording': item['name'], 'sha256': hashlib.sha256(item['bytes']).hexdigest()})
                del item['bytes']
            with (folder / 'recordings.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['path', 'subject_id', 'group_id'])
                for index, item in enumerate(uploads):
                    writer.writerow([f'recording_{index:03}.edf', item['subject'], item['group']])
            output = Path(temp) / 'results'
            job.update(stage='Running the original EEG pipeline', progress=52)
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(Path(temp) / 'mpl'),
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
            run = execute_original(folder, output, settings, temp, env)
            status_file = output / 'baseline_status.txt'
            if status_file.exists():
                training = 'Not fitted: at least 3 independent recording/session groups per person are required.'
            elif 'Need at least two people' in run.stderr:
                training = 'Not fitted: at least two people are required for genuine-versus-impostor evaluation.'
            elif 'Review artifacts before fitting' in run.stderr:
                training = 'Not fitted: investigate the artifact flags before proceeding with training in the local script.'
            elif (output / 'svm_metrics.csv').exists():
                training = 'SVM evaluated by the original script. Interpret metrics alongside recording counts and your session protocol.'
            else:
                raise RuntimeError('Analysis stopped. Check that these are valid EDF recordings with matching EEG channels and sampling rates, and that the high cutoff is below half the sampling rate.')
            job.update(stage='Reading validation outputs', progress=90)
            with (output / 'recording_checks.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            with (output / 'features.csv').open() as stream:
                reader = csv.reader(stream)
                feature_columns = len(next(reader)) - 4
                feature_rows = sum(1 for _ in reader)
            recordings = []
            for i, row in enumerate(rows):
                recordings.append({'name': uploads[i]['name'], 'subject': row['subject_id'],
                    'windows': int(row['windows']), 'flagged': int(row['flagged_windows']),
                    'duration': round(float(row['duration_s']), 1),
                    'preview': base64.b64encode((output / f'recording_{i}_preview.png').read_bytes()).decode()})
            cnn_summary = run.stdout[run.stdout.find('Model:'):] if 'Model:' in run.stdout else ''
            cnn_status = ('Architecture built only; not trained or evaluated.' if cnn_summary else
                          'CNN build did not finish in this runtime. SVM results are retained; try the original script locally.' if settings['cnn'] and (output / 'svm_metrics.csv').exists() else
                          'Not built: the source stops before the optional CNN when training requirements are unmet.' if settings['cnn'] else
                          'Not requested. The source defines an untrained architecture candidate.')
            result = {'recordings': recordings, 'windows': feature_rows, 'features': feature_columns,
                      'channels': len(rows[0]['channels'].split('|')), 'sampling_hz': float(rows[0]['sampling_hz']),
                      'flagged': sum(r['flagged'] for r in recordings), 'settings': settings,
                      'training': training, 'svm_metrics': [], 'cnn_status': cnn_status, 'cnn_summary': cnn_summary,
                      'provenance': provenance, 'pipeline_sha256': hashlib.sha256((ROOT / 'hello.py').read_bytes()).hexdigest()}
            if (output / 'svm_metrics.csv').exists():
                with (output / 'svm_metrics.csv').open() as stream:
                    result['svm_metrics'] = list(csv.DictReader(stream))
        job.update(state='complete', stage='Preprocessing complete', progress=100, result=result)
    except Exception as error:
        # No filesystem paths or subprocess logs are exposed to public clients.
        message = str(error) if isinstance(error, (ValueError, RuntimeError)) else 'The run could not finish. Check your EDF files and settings, then try again.'
        job.update(state='error', stage=message, progress=0)
    finally:
        BUSY.release()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send(self, code, data, kind='application/json'):
        body = data if isinstance(data, bytes) else json.dumps(data).encode()
        self.send_response(code)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            return self.send(200, (ROOT / 'index.html').read_bytes(), 'text/html; charset=utf-8')
        if self.path == '/health':
            return self.send(200, {'status': 'ok'})
        if self.path.startswith('/api/jobs/'):
            key = self.path.removeprefix('/api/jobs/')
            with LOCK:
                job = JOBS.get(key)
                if job and time.monotonic() - job['created'] < 1200:
                    return self.send(200, {k: v for k, v in job.items() if k != 'created'})
            return self.send(404, {'error': 'Run expired or was not found. Start a new run.'})
        return self.send(404, {'error': 'Not found'})

    def do_POST(self):
        if self.path != '/api/run':
            return self.send(404, {'error': 'Not found'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 29 * 1024 * 1024:
                return self.send(413, {'error': 'Upload at most 20 MB of EDF recordings.'})
            if not self.headers.get('Content-Type', '').startswith('application/json'):
                return self.send(415, {'error': 'JSON upload required.'})
            self.connection.settimeout(30)
            data = json.loads(self.rfile.read(length))
            settings = {k: float(data[k]) for k in ('low', 'high', 'window')}
            if not all(math.isfinite(x) for x in settings.values()):
                raise ValueError('Use finite numbers.')
            if not (0.5 <= settings['low'] < settings['high'] <= 70 and 0.5 <= settings['window'] <= 10):
                raise ValueError('Use 0.5–70 Hz with low below high, and a 0.5–10 second window.')
            settings.update(cnn=data.get('cnn') is True, reviewed=data.get('reviewed') is True)
            entries = data.get('files')
            if not isinstance(entries, list) or not 1 <= len(entries) <= 12:
                raise ValueError('Select between 1 and 12 EDF files.')
            uploads = []
            total = 0
            for entry in entries:
                name = entry['name']
                if not isinstance(name, str) or not re.fullmatch(r'[\w .()-]{1,120}\.edf', name, re.IGNORECASE):
                    raise ValueError('Use EDF filenames containing letters, numbers, spaces, dots, hyphens, or parentheses.')
                subject, group = entry['subject'], entry['group']
                if not all(isinstance(x, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,48}', x) for x in (subject, group)):
                    raise ValueError('Provide subject and session/group IDs using letters, numbers, underscores or hyphens.')
                raw = base64.b64decode(entry['data'], validate=True)
                total += len(raw)
                if not 256 <= len(raw) <= MAX_FILE or total > MAX_TOTAL:
                    raise ValueError('Use EDF files up to 8 MB each, with at most 20 MB total.')
                if raw[:8] != b'0       ':
                    raise ValueError('A selected file does not have a valid EDF header.')
                uploads.append({'name': name, 'subject': subject, 'group': group, 'bytes': raw})
            del data
        except (ValueError, TypeError, KeyError, AttributeError):
            return self.send(400, {'error': 'Check your EDF files, subject/session IDs, and settings. Limits: 12 files, 8 MB each, 20 MB total; 0.5–70 Hz, low below high; 0.5–10 second windows.'})
        if not BUSY.acquire(blocking=False):
            return self.send(429, {'error': 'Another analysis is running. Please try again in a minute.'})
        key = secrets.token_urlsafe(24)
        job = {'state': 'running', 'stage': 'Preparing your uploaded recordings', 'progress': 3, 'created': time.monotonic()}
        with LOCK:
            for old in list(JOBS):
                if time.monotonic() - JOBS[old]['created'] > 1200:
                    del JOBS[old]
            while len(JOBS) >= 10:
                del JOBS[next(iter(JOBS))]
            JOBS[key] = job
        threading.Thread(target=run_pipeline, args=(job, settings, uploads), daemon=True).start()
        def expire():
            with LOCK:
                JOBS.pop(key, None)
        expiry = threading.Timer(1200, expire)
        expiry.daemon = True
        expiry.start()
        self.send(202, {'job': key})


if __name__ == '__main__':
    server = ThreadingHTTPServer(('0.0.0.0', int(os.environ.get('PORT', '8080'))), Handler)
    print('EEG browser interface ready', flush=True)
    server.serve_forever()
