# Version 2 — Faster CNN Inference with PyTorch

This version rewrites the optional TensorFlow CNN in PyTorch, adds GPU support and a small Optuna search, and organizes the code so the workflow is easier to follow and explain.

**Both versions remain available:** [original V1 script](../hello.py), [V1 browser app](../web_app.py), and [V2 script](hello.py). The existing browser app still runs V1. V2 is a separate command-line workflow; it has not been connected to or deployed through that app.

## What changed?

| Part | V1 at the repository root | V2 in this folder |
|---|---|---|
| CNN framework | TensorFlow/Keras | PyTorch |
| Structure | Original sequential script | Named workflow functions, EEGDataset and EEGCNN classes |
| Batches | Future CNN work | Train/validation/test DataLoaders |
| Device | Published V1 requirements use tensorflow-cpu | CUDA GPU when available, with CPU fallback |
| CNN preview | Untrained architecture candidate | Untrained architecture plus a sample forward pass |
| Optional training | Not implemented | Six Optuna trials for one claimed person |
| Settings searched | None | Learning rate and training batch size |
| EEG preprocessing and SVM | Original protocol | Preserved protocol and baseline outputs |
| Browser interface | Existing root web app | Command line only |

## Run V2 in Ubuntu or WSL

Use a separate environment so V1 keeps its own dependencies:

```bash
git clone https://github.com/krishna-ai-cpp/EEG-Brainwave-Authentication.git
cd EEG-Brainwave-Authentication/v2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python hello.py
```

The requirements reproduce the tested PyTorch 2.11.0+cu128 and Optuna 5.0.0 setup on Ubuntu with an NVIDIA GPU. A compatible NVIDIA driver is needed for CUDA. The same installed build falls back to CPU when CUDA is unavailable. V1 still uses the separate requirements at the repository root.

Supply your own authorized EEG recording folder and a **new** results folder. Enter the low/high frequency cutoffs, window duration and amplitude review limit. The script reads `recordings.csv` when present, or asks for subject and session/group IDs. The manifest columns are `path,subject_id,group_id`.

After successful SVM evaluation, choose:

- `preview`: show the CNN architecture and one untrained forward pass. The original `yes` answer also works.
- `tune`: run exactly six Optuna trials. Enter the claimed subject when prompted.
- Enter: finish without running a CNN step.

## Why the script may stop before tuning

The inherited protocol requires at least **two people and three independent recording/session groups per person**. Related windows stay in the same split. Training, validation and test sessions must not overlap, including sessions shared by different people.

If there are too few groups, preprocessing results are still saved, but SVM fitting and the later CNN menu are not reached. Do not rename windows or task runs as independent sessions to bypass this guard. Artifact flags also require review before fitting; flagged windows are not silently removed or repaired.

## Read the code in this order

Start with `main()` at the bottom of [hello.py](hello.py):

```text
ask_settings()
    -> find_recordings()
    -> prepare_recordings()
    -> run_svm()
    -> optional preview or Optuna tuning
```

`EEGDataset` supplies one window and binary label. `DataLoader` groups examples into batches. `EEGCNN.__init__()` defines the layers, and `forward()` shows their order. Saved windows already use `(windows, channels, time)`, which matches PyTorch Conv1d.

The CNN keeps the original arrangement: Conv1d(16), ReLU, BatchNorm, MaxPool, Conv1d(32), ReLU, BatchNorm, average across time, Linear(16), ReLU, Dropout(0.5), Linear(1), Sigmoid. The class and forward path remain fixed during the Optuna search.

## Six beginner-friendly Optuna trials

The separate [optuna_tuning.py](optuna_tuning.py) keeps the main file readable:

1. Reuse the existing training/validation groups. No test window file is loaded.
2. Fit channel normalization on training windows only, then apply it to validation.
3. Pick a learning rate from 0.0001 to 0.01 and a training batch size from 16, 32 or 64.
4. Train a fresh copy of the fixed CNN for five epochs.
5. Compare window-averaged validation binary cross-entropy; lower is better.
6. Repeat until `study.optimize(objective, n_trials=6, n_jobs=1)` finishes.

One study tunes one claimed person against the other enrolled people. Five epochs is a short introductory setting, not proof of convergence. Training batch size does not force predictions to wait for that many windows: the CNN still accepts a single window.

The new results folder contains `optuna_cnn/trials.csv`, `best_parameters.json`, and `best_cnn.pt`. The checkpoint includes weights, channel order, the claimed subject, and training normalization statistics. New prediction windows must use that channel order and those saved statistics.

The preview remains untrained. Tuning trains a model only when eligible data is supplied and the search completes. This search does not perform held-out test evaluation, select an authentication threshold, or establish CNN authentication accuracy. Recording-level evaluation with validation-selected thresholds is still required. No SVM/CNN fusion is implemented.

See the [Optuna walkthrough](OPTUNA-GUIDE.md) and [PyTorch walkthrough](PYTORCH-GUIDE.md).

## Faster CNN response: what was measured?

These local results use prepared EEG windows with **64 channels and 320 samples**, eager float32 inference, and matched CPU thread counts for each comparison. They benchmark untrained architecture candidates, not trained real-data authentication.

| Measurement | TensorFlow CPU | PyTorch CPU | PyTorch RTX 3050 GPU |
|---|---:|---:|---:|
| Initial comparison, 4 CPU threads, one window | 14.698 ms | 0.846 ms | 0.586 ms |
| Later 4-thread repeat, one window | 9.870 ms | 0.786 ms | 0.604 ms |
| Single-thread latency setting, one window | 12.165 ms | 0.317 ms | 0.564 ms |

The initial same-CPU comparison was about **17x** faster. The later four-thread comparison was about **12.6x**, and the matched single-thread comparison was about **38x**. The ratio varies with settings and system load; 17x is not a guaranteed product property. TensorFlow detected no GPU, so TensorFlow CPU versus PyTorch GPU also changes hardware.

The CPU preview uses one thread to reduce overhead for small-window inference. Larger CPU batches can benefit from more threads. CUDA is chosen automatically when available. Optuna is imported only for tuning and does not run during prediction.

**This is not a whole-project speedup claim.** EEG preprocessing and scikit-learn SVM fitting stay on CPU. In the earlier fresh-process real-data run, the original script took 4.99 s and the PyTorch rewrite took 7.13 s, both stopping at the insufficient-group guard; startup/import overhead mattered. Six training trials add offline work.

See [benchmark methods and results](benchmarks/README.md), [recorded summaries](benchmarks/results.json), and the [reproduction script](benchmarks/benchmark_inference.py).

## Checks you can run

From this `v2/` directory:

```bash
python -m unittest discover -s tests -v
```

These tests use temporary synthetic arrays. They check batches and labels, CPU forward/backward execution, training-only normalization, exactly six trials, saved-checkpoint reload, group guards, and exclusion of test windows. They do not download EEG recordings or claim biometric accuracy.

Additional local conversion checks compared every generated preprocessing file on the supplied EDF samples and the complete SVM outputs on a synthetic grouped dataset. The six-trial CLI workflow was checked on GPU at the default five epochs; CPU fallback was checked too. No full-dataset V2 authentication evaluation is claimed. V1's previously documented BI2015 results remain V1 results.

Keep EEG data, generated biometric arrays, environments, checkpoints and credentials out of Git. The original V1 attribution and research notes remain in the [root README](../README.md). This is an exploratory research project, not a validated access-control or medical system.
