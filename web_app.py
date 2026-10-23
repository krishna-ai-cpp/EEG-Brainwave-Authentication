"""Browser inputs and output presentation for the unchanged hello.py pipeline.

No filtering, feature extraction, model fitting or thresholds are reimplemented here.
"""
import base64
import csv
import hashlib
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import secrets
import selectors
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs

ROOT = Path(__file__).resolve().parent
JOBS = {}
LOCK = threading.RLock()
BUSY = threading.Semaphore(1)
TTL = 1200
MAX_FILE = 8 * 1024 * 1024
MAX_TOTAL = 20 * 1024 * 1024
TABLES = ('features.csv', 'recording_checks.csv', 'window_checks.csv',
          'subject_counts.csv', 'recording_split.csv', 'svm_metrics.csv',
          'svm_recording_scores.csv')
OUTPUT = re.compile(r'(?:' + '|'.join(re.escape(x) for x in TABLES) +
                    r'|baseline_status\.txt|svm_test_roc\.png|recording_\d+_(?:windows\.npy|preview\.png)|pipeline_console\.txt|run_report\.json)')


def safe_text(value, job):
    value = re.sub(r'\x1b\[[0-9;]*m', '', value)
    return value.replace(job['_temp'], '[run]').replace(str(ROOT), '[application]')


def table(path, offset=0, limit=50):
    if not path.is_file():
        return {'columns': [], 'rows': [], 'total': 0, 'offset': offset}
    rows = []
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        count = 0
        for count, row in enumerate(reader, 1):
            if offset < count <= offset + limit:
                if 'path' in row:
                    row['path'] = Path(row['path']).name
                rows.append(row)
    return {'columns': columns, 'rows': rows, 'total': count, 'offset': offset}


def collect(job):
    out = Path(job['_output'])
    checks = table(out / 'recording_checks.csv', limit=10000)['rows']
    metrics = table(out / 'svm_metrics.csv', limit=10000)['rows']
    recordings = []
    for row in checks:
        index = int(row['recording_id'])
        recordings.append(dict(row, name=job['_names'][index],
                               preview=f'recording_{index}_preview.png',
                               windows=int(row['windows']), flagged=int(row['flagged_windows'])))
    stdout = job.get('console', '')
    cnn = stdout[stdout.find('Model:'):] if 'Model:' in stdout else ''
    result = {
        'recordings': recordings, 'windows': sum(r['windows'] for r in recordings),
        'flagged': sum(r['flagged'] for r in recordings),
        'channels': len(checks[0]['channels'].split('|')) if checks else 0,
        'features': 3 * len(checks[0]['channels'].split('|')) if checks else 0,
        'sampling_hz': float(checks[0]['sampling_hz']) if checks else None,
        'settings': job['_settings'], 'provenance': job['_provenance'],
        'pipeline_sha256': hashlib.sha256((ROOT / 'hello.py').read_bytes()).hexdigest(),
        'svm_metrics': metrics, 'cnn_summary': cnn,
        'cnn_status': ('Architecture built only; not trained or evaluated.' if cnn else
                       'Not reached or not requested. hello.py builds the candidate only after SVM evaluation.'),
        'macro': {k: sum(float(r[k]) for r in metrics) / len(metrics)
                  for k in ('FAR', 'FRR', 'EER_approx', 'ROC_AUC', 'accuracy')} if metrics else {},
        'tables': {name: table(out / name) for name in TABLES if (out / name).exists()},
        'files': [{'name': p.name, 'bytes': p.stat().st_size} for p in sorted(out.glob('*'))
                  if p.is_file() and OUTPUT.fullmatch(p.name)],
    }
    return result


def execute(job, folder, output):
    settings = job['_settings']
    temp = job['_temp']
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(Path(temp) / 'mpl'),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', TF_NUM_INTRAOP_THREADS='1',
               TF_NUM_INTEROP_THREADS='1', TF_CPP_MIN_LOG_LEVEL='2')
    initial = '\n'.join(map(str, [folder, output, settings['low'], settings['high'],
                                  settings['window'], settings['artifact']])) + '\n'
    with (Path(temp) / 'stderr.txt').open('w+') as errors:
        process = subprocess.Popen([sys.executable, '-u', '-B', str(ROOT / 'hello.py')],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
                                   cwd=temp, env=env)
        chunks = bytearray()
        answered = set()
        deadline = time.monotonic() + 240
        try:
            process.stdin.write(initial.encode())
            process.stdin.flush()
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while selector.get_map():
                    if time.monotonic() > deadline:
                        raise RuntimeError('Cloud run exceeded four minutes. Use smaller recordings or the original local pipeline for a full dataset.')
                    for key, _ in selector.select(timeout=1):
                        chunk = key.fileobj.read1(65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        chunks.extend(chunk)
                        text = chunks.decode(errors='replace')
                        job['console'] = safe_text(text[-100000:], job)
                        marker = 'Type yes to continue:'
                        if marker in text and marker not in answered:
                            job['result'] = collect(job)
                            job.update(state='review', stage='Inspect quality flags and previews, then confirm or stop.', progress=65)
                            began = time.monotonic()
                            if not job['_review'].wait(timeout=600):
                                job['_answer'] = False
                            deadline += time.monotonic() - began
                            process.stdin.write(b'yes\n' if job.get('_answer') else b'no\n')
                            process.stdin.flush()
                            answered.add(marker)
                            job.update(state='running', stage='Evaluating recording-level SVMs', progress=75)
                        marker = 'Build the CNN candidate too? Type yes or press Enter to finish:'
                        if marker in text and marker not in answered:
                            process.stdin.write(b'yes\n' if settings['cnn'] else b'no\n')
                            process.stdin.flush()
                            answered.add(marker)
                            job.update(stage='Building optional CNN architecture' if settings['cnn'] else 'Reading final outputs', progress=95)
            process.wait(timeout=5)
            errors.seek(0)
            stderr = safe_text(errors.read(), job)
            job['console'] = safe_text(chunks.decode(errors='replace'), job)[-100000:] + '\n' + stderr[-12000:]
            return process.returncode, stderr
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdin.close()
            process.stdout.close()


def run_pipeline(job, uploads):
    output = Path(job['_output'])
    folder = Path(job['_temp']) / 'recordings'
    try:
        folder.mkdir()
        for index, item in enumerate(uploads):
            name = f'recording_{index:03}{item["extension"]}'
            (folder / name).write_bytes(item.pop('bytes'))
            item['path'] = name
        with (folder / 'recordings.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['path', 'subject_id', 'group_id'])
            for item in uploads:
                writer.writerow([item['path'], item['subject'], item['group']])
        job.update(stage='Running hello.py: load, filter, window and extract features', progress=20)
        code, stderr = execute(job, folder, output)
        if code == 0:
            state, stage = 'complete', 'Final output ready — original pipeline finished'
        elif (output / 'baseline_status.txt').exists():
            state, stage = 'blocked', 'Preprocessing complete; SVM needs at least three independent groups per person.'
        elif 'Need at least two people' in stderr:
            state, stage = 'blocked', 'Preprocessing complete; SVM needs at least two people.'
        elif 'Review artifacts before fitting' in stderr:
            state, stage = 'blocked', 'Preprocessing complete; artifact review was declined or timed out. SVM was not fitted.'
        else:
            state = 'error'
            last = next((x for x in reversed(stderr.splitlines()) if x.strip()), 'Pipeline process stopped without completing.')
            stage = last[:500]
        job.update(exit_code=code, _final_state=state, stage=stage, progress=98)
    except Exception as error:
        job.update(_final_state='error', stage=safe_text(str(error), job)[:500], progress=98)
    finally:
        # Delete only this run's temporary uploads. Keep generated outputs for download until expiry.
        shutil.rmtree(folder, ignore_errors=True)
        if output.exists():
            (output / 'pipeline_console.txt').write_text(job.get('console', ''), encoding='utf-8')
            job['result'] = collect(job)
            job['result']['training'] = job['stage']
            report = {k: v for k, v in job['result'].items() if k != 'tables'}
            report.update(state=job['_final_state'], exit_code=job.get('exit_code'), status=job['stage'])
            (output / 'run_report.json').write_text(json.dumps(report, indent=2))
            job['result'] = collect(job)
            job['result']['training'] = job['stage']
        job['_done'] = True
        job.update(state=job['_final_state'], progress=100)
        job['finished'] = time.monotonic()
        BUSY.release()
        timer = threading.Timer(TTL, expire, args=(job,))
        timer.daemon = True
        timer.start()


def expire(job):
    with LOCK:
        JOBS.pop(job['_key'], None)
        shutil.rmtree(job['_temp'], ignore_errors=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def headers_for(self, code, size, kind, filename=None):
        self.send_response(code)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(size))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'")
        if filename:
            self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.end_headers()

    def send(self, code, data, kind='application/json'):
        body = data if isinstance(data, bytes) else json.dumps(data).encode()
        self.headers_for(code, len(body), kind)
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_file(self, path, attachment=False):
        self.headers_for(200, path.stat().st_size, mimetypes.guess_type(path.name)[0] or 'application/octet-stream',
                         path.name if attachment else None)
        try:
            with path.open('rb') as stream:
                shutil.copyfileobj(stream, self.wfile, 1024 * 1024)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        url = urlsplit(self.path)
        if url.path in ('/', '/index.html'):
            return self.send(200, (ROOT / 'index.html').read_bytes(), 'text/html; charset=utf-8')
        if url.path == '/health':
            return self.send(200, {'status': 'ok', 'version': 'script-output-v2'})
        parts = url.path.strip('/').split('/')
        if len(parts) >= 3 and parts[:2] == ['api', 'jobs']:
            with LOCK:
                job = JOBS.get(parts[2])
                if not job:
                    return self.send(404, {'error': 'Run expired or not found. Outputs are retained for 20 minutes after completion.'})
                if len(parts) == 3:
                    return self.send(200, {k: v for k, v in job.items() if not k.startswith('_') and k != 'finished'})
                output = Path(job['_output'])
                if len(parts) == 5 and parts[3] == 'tables' and parts[4] in TABLES:
                    try:
                        offset = max(0, int(parse_qs(url.query).get('offset', ['0'])[0]))
                    except ValueError:
                        return self.send(400, {'error': 'Invalid page offset.'})
                    return self.send(200, table(output / parts[4], offset))
                if len(parts) == 5 and parts[3] == 'files' and OUTPUT.fullmatch(parts[4]):
                    path = output / parts[4]
                    if path.is_file():
                        return self.send_file(path, not path.name.endswith('.png'))
                if len(parts) == 4 and parts[3] == 'download' and job.get('_done'):
                    target = Path(job['_temp']) / 'eeg-results.zip'
                    if not target.exists():
                        with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
                            for p in output.iterdir():
                                if p.is_file() and OUTPUT.fullmatch(p.name):
                                    archive.write(p, p.name)
                    return self.send_file(target, True)
        self.send(404, {'error': 'Not found'})

    def do_POST(self):
        # A foreign website may not act on this server through a browser.
        origin = self.headers.get('Origin')
        if origin and urlsplit(origin).netloc != self.headers.get('Host'):
            return self.send(403, {'error': 'Cross-origin requests are not allowed.'})
        parts = self.path.strip('/').split('/')
        if len(parts) == 4 and parts[:2] == ['api', 'jobs'] and parts[3] == 'review':
            try:
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size < 1024:
                    raise ValueError()
                data = json.loads(self.rfile.read(size))
                with LOCK:
                    job = JOBS.get(parts[2])
                    if not job or job['state'] != 'review':
                        return self.send(409, {'error': 'This run is not waiting for artifact review.'})
                    if not isinstance(data.get('continue'), bool):
                        raise ValueError()
                    job['_answer'] = data['continue']
                    job['_review'].set()
                return self.send(200, {'status': 'Review decision received'})
            except (ValueError, TypeError):
                return self.send(400, {'error': 'Choose continue or stop.'})
        if self.path != '/api/run':
            return self.send(404, {'error': 'Not found'})
        try:
            length = int(self.headers.get('Content-Length', 0))
            if not 0 < length <= 29 * 1024 * 1024:
                return self.send(413, {'error': 'Upload at most 20 MB total.'})
            if not self.headers.get('Content-Type', '').startswith('application/json'):
                return self.send(415, {'error': 'JSON upload required.'})
            self.connection.settimeout(30)
            data = json.loads(self.rfile.read(length))
            settings = {k: float(data[k]) for k in ('low', 'high', 'window')}
            settings['artifact'] = float(data.get('artifact', 150))
            if not all(math.isfinite(x) for x in settings.values()):
                raise ValueError('Use finite settings.')
            if not (0 <= settings['low'] < settings['high'] and 0 < settings['window'] <= 30 and settings['artifact'] > 0):
                raise ValueError('Use 0 ≤ low < high, a positive window up to 30 seconds, and a positive review limit.')
            settings['cnn'] = data.get('cnn') is True
            entries = data.get('files')
            if not isinstance(entries, list) or not 1 <= len(entries) <= 12:
                raise ValueError('Select 1–12 recordings.')
            uploads, total, provenance = [], 0, []
            for entry in entries:
                name = entry['name']
                if not isinstance(name, str) or not re.fullmatch(r'[\w .()-]{1,120}\.(edf|bdf|fif)', name, re.I):
                    raise ValueError('Browser uploads support single-file EDF, BDF and FIF recordings with simple filenames.')
                subject, group = entry['subject'], entry['group']
                if not all(isinstance(x, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,48}', x) for x in (subject, group)):
                    raise ValueError('Provide subject and group IDs using letters, digits, underscores or hyphens.')
                raw = base64.b64decode(entry['data'], validate=True)
                total += len(raw)
                if not 256 <= len(raw) <= MAX_FILE or total > MAX_TOTAL:
                    raise ValueError('Limit: 8 MB per recording and 20 MB total. Run larger datasets locally.')
                extension = Path(name).suffix.lower()
                if extension == '.edf' and raw[:8] != b'0       ':
                    raise ValueError('Invalid EDF header.')
                uploads.append(dict(name=name, subject=subject, group=group, bytes=raw, extension=extension))
                provenance.append(dict(recording=name, sha256=hashlib.sha256(raw).hexdigest()))
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            message = str(error) if isinstance(error, ValueError) else 'Check your files, subject/group IDs and settings.'
            return self.send(400, {'error': message[:300]})
        if not BUSY.acquire(blocking=False):
            return self.send(429, {'error': 'Another run is processing or awaiting artifact review. Please try again later.'})
        key = secrets.token_urlsafe(24)
        temp = tempfile.mkdtemp(prefix='eeg-run-')
        job = dict(state='running', stage='Preparing uploads', progress=3, console='', _key=key,
                   _temp=temp, _output=str(Path(temp) / 'results'), _settings=settings,
                   _names=[x['name'] for x in uploads], _provenance=provenance, _review=threading.Event())
        with LOCK:
            JOBS[key] = job
        threading.Thread(target=run_pipeline, args=(job, uploads), daemon=True).start()
        self.send(202, {'job': key})


if __name__ == '__main__':
    print('EEG browser interface ready', flush=True)
    ThreadingHTTPServer(('0.0.0.0', int(os.environ.get('PORT', '8080'))), Handler).serve_forever()
