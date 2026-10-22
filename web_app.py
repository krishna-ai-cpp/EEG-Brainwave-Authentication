"""Public-data browser adapter. The research pipeline in hello.py is unchanged."""
import base64
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
SAMPLES = ('S001R01', 'S001R03', 'S002R03')
SOURCE = 'https://physionet.org/files/eegmmidb/1.0.0/'
JOBS = {}
LOCK = threading.Lock()
BUSY = threading.Semaphore(1)
MAX_FILE = 4 * 1024 * 1024


def run_pipeline(job, settings):
    try:
        with tempfile.TemporaryDirectory(prefix='eeg-public-') as temp:
            folder = Path(temp) / 'recordings'
            folder.mkdir()
            provenance = []
            for index, name in enumerate(SAMPLES):
                job.update(stage=f'Loading public recording {index + 1} of 3', progress=10 + index * 12)
                url = SOURCE + name[:4] + '/' + name + '.edf'
                with urlopen(url, timeout=45) as response:
                    data = response.read(MAX_FILE + 1)
                if len(data) > MAX_FILE:
                    raise ValueError('Public sample exceeded the expected file-size limit.')
                (folder / (name + '.edf')).write_bytes(data)
                provenance.append({'recording': name, 'url': url, 'sha256': hashlib.sha256(data).hexdigest()})
            with (folder / 'recordings.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['path', 'subject_id', 'group_id'])
                # Separate runs are not evidence of independent sessions.
                # Conservatively keep all runs from one subject in one group.
                for name in SAMPLES:
                    writer.writerow([name + '.edf', name[:4], name[:4] + '_public_session'])
            output = Path(temp) / 'results'
            answers = '\n'.join([str(folder), str(output), str(settings['low']),
                                 str(settings['high']), str(settings['window']), '150', 'no', 'no']) + '\n'
            job.update(stage='Running the original EEG pipeline', progress=52)
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(Path(temp) / 'mpl'),
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
            run = subprocess.run([sys.executable, '-B', str(ROOT / 'hello.py')], input=answers,
                                 text=True, capture_output=True, cwd=temp, env=env, timeout=180)
            status_file = output / 'baseline_status.txt'
            guarded = status_file.exists() and 'SVM not fitted: need at least 3 independent' in status_file.read_text()
            if run.returncode != 0 and not guarded:
                raise RuntimeError('The analysis did not complete. Please retry with the default settings.')
            job.update(stage='Reading validation outputs', progress=90)
            with (output / 'recording_checks.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            with (output / 'features.csv').open() as stream:
                reader = csv.reader(stream)
                feature_columns = len(next(reader)) - 4
                feature_rows = sum(1 for _ in reader)
            recordings = []
            for i, row in enumerate(rows):
                recordings.append({'name': SAMPLES[i], 'subject': row['subject_id'],
                    'windows': int(row['windows']), 'flagged': int(row['flagged_windows']),
                    'duration': round(float(row['duration_s']), 1),
                    'preview': base64.b64encode((output / f'recording_{i}_preview.png').read_bytes()).decode()})
            result = {'recordings': recordings, 'windows': feature_rows, 'features': feature_columns,
                      'channels': len(rows[0]['channels'].split('|')), 'sampling_hz': float(rows[0]['sampling_hz']),
                      'flagged': sum(r['flagged'] for r in recordings), 'settings': settings,
                      'training': 'Not fitted: at least 3 independent session groups per person are required.',
                      'provenance': provenance, 'pipeline_sha256': hashlib.sha256((ROOT / 'hello.py').read_bytes()).hexdigest()}
        job.update(state='complete', stage='Preprocessing complete', progress=100, result=result)
    except Exception as error:
        # No filesystem paths or subprocess logs are exposed to public clients.
        message = str(error) if isinstance(error, (ValueError, RuntimeError)) else 'The public-data run could not finish. Please try again shortly.'
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
            if not 0 < length <= 2048:
                raise ValueError('Invalid request size.')
            data = json.loads(self.rfile.read(length))
            settings = {k: float(data[k]) for k in ('low', 'high', 'window')}
            if not all(math.isfinite(x) for x in settings.values()):
                raise ValueError('Use finite numbers.')
            if not (0.5 <= settings['low'] < settings['high'] <= 70 and 0.5 <= settings['window'] <= 10):
                raise ValueError('Use 0.5–70 Hz with low below high, and a 0.5–10 second window.')
        except (ValueError, TypeError, KeyError):
            return self.send(400, {'error': 'Check frequency and window settings. Low must be below high; use 0.5–70 Hz and 0.5–10 seconds.'})
        if not BUSY.acquire(blocking=False):
            return self.send(429, {'error': 'Another public demonstration is running. Please try again in a minute.'})
        key = secrets.token_urlsafe(24)
        job = {'state': 'running', 'stage': 'Preparing public sample recordings', 'progress': 3, 'created': time.monotonic()}
        with LOCK:
            for old in list(JOBS):
                if time.monotonic() - JOBS[old]['created'] > 1200:
                    del JOBS[old]
            while len(JOBS) >= 10:
                del JOBS[next(iter(JOBS))]
            JOBS[key] = job
        threading.Thread(target=run_pipeline, args=(job, settings), daemon=True).start()
        self.send(202, {'job': key})


if __name__ == '__main__':
    server = ThreadingHTTPServer(('0.0.0.0', int(os.environ.get('PORT', '8080'))), Handler)
    print('EEG browser interface ready', flush=True)
    server.serve_forever()
