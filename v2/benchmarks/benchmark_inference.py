"""Isolate framework runtimes and time matching CNNs without changing project code."""
import ast
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--framework', choices=['tensorflow', 'pytorch'], required=True)
parser.add_argument('--windows', required=True, help='NPY with shape (windows, channels, time)')
parser.add_argument('--threads', type=int, default=1)
parser.add_argument('--output', required=True)
args = parser.parse_args()
if args.threads < 1:
    parser.error('--threads must be positive')
framework, output_name, windows_name = args.framework, args.output, args.windows
project = Path(__file__).resolve().parents[2]
windows = np.load(windows_name).astype(np.float32)
if windows.ndim != 3 or len(windows) == 0 or windows.shape[2] < 2 or not np.isfinite(windows).all():
    raise ValueError('Expected nonempty finite windows with shape (windows, channels, time>=2).')
threads = args.threads
report = {'framework':framework, 'threads':threads, 'warmup':25, 'iterations':100, 'results':[]}

if framework == 'tensorflow':
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.random.set_seed(42)
    report['version'] = tf.__version__
    report['gpus'] = [str(d) for d in tf.config.list_physical_devices('GPU')]
    devices = [('cpu','/CPU:0')]
    if report['gpus']:
        devices.append(('gpu','/GPU:0'))
else:
    import torch
    from torch import nn
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    report['version'] = torch.__version__
    report['gpus'] = [torch.cuda.get_device_name(0)] if torch.cuda.is_available() else []
    devices = [('cpu','cpu')]
    if report['gpus']:
        devices.append(('gpu','cuda'))

for device_label, device in devices:
    if framework == 'tensorflow':
        # Execute the original Keras imports and Sequential assignment unchanged.
        source = (project/'hello.py').read_text(encoding='utf-8-sig')
        tree = ast.parse(source)
        optional_block = next(n for n in tree.body if isinstance(n, ast.If)
                              and isinstance(n.test, ast.Compare)
                              and isinstance(n.test.left, ast.Name) and n.test.left.id == 'show_cnn')
        nodes = [n for n in optional_block.body if isinstance(n, ast.ImportFrom)
                 or (isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='nn' for t in n.targets))]
        namespace = {'win_size':windows.shape[2], 'n_channels':windows.shape[1]}
        with tf.device(device):
            exec(compile(ast.Module(body=nodes,type_ignores=[]),'original_cnn','exec'), namespace)
            model = namespace['nn']
    else:
        # Execute the installed model class unchanged; its imported data tools are not timed.
        tree = ast.parse((project/'v2/hello.py').read_text(encoding='utf-8'))
        model_node = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EEGCNN')
        namespace = {'torch':torch,'nn':nn}
        exec(compile(ast.Module(body=[model_node],type_ignores=[]),'pytorch_cnn','exec'),namespace)
        model = namespace['EEGCNN'](windows.shape[1]).to(device)
        model.eval()

    for batch_size in [1,32]:
        samples = windows[np.arange(batch_size) % len(windows)]
        if framework == 'tensorflow':
            # Keras expects (batch,time,channels); PyTorch uses (batch,channels,time).
            inputs = np.ascontiguousarray(samples.transpose(0,2,1))
            def response():
                with tf.device(device):
                    value = model(tf.convert_to_tensor(inputs),training=False)
                    return value.numpy()  # Finish execution and return output to the CPU.
            for _ in range(report['warmup']): response()
            durations=[]
            for _ in range(report['iterations']):
                start=time.perf_counter_ns()
                result=response()
                durations.append((time.perf_counter_ns()-start)/1e6)
        else:
            inputs=np.ascontiguousarray(samples)
            def response():
                value=model(torch.from_numpy(inputs).to(device))
                return value.cpu().numpy()  # Synchronizes CUDA and includes output transfer.
            with torch.no_grad():
                for _ in range(report['warmup']): response()
                durations=[]
                for _ in range(report['iterations']):
                    start=time.perf_counter_ns()
                    result=response()
                    durations.append((time.perf_counter_ns()-start)/1e6)
        assert result.shape==(batch_size,1) and np.isfinite(result).all()
        report['results'].append({'device':device_label,'batch_size':batch_size,
            'shape':list(samples.shape), 'median_ms':float(np.median(durations)),
            'p90_ms':float(np.percentile(durations,90)), 'samples_ms':durations})

Path(output_name).write_text(json.dumps(report,indent=2),encoding='utf-8')
print(framework,report['version'],[(r['device'],r['batch_size'],round(r['median_ms'],3)) for r in report['results']],flush=True)
