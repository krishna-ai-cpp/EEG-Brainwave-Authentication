# Your EEG project in PyTorch

Run in Ubuntu using the project's existing environment:

```bash
cd EEG-Brainwave-Authentication/v2
.venv/bin/python hello.py
```

For the recording folder, enter the path to your own authorized EEG recordings.
Choose a new results directory for each run. Existing results are protected.

## Read the code in this order

Start at `main()` near the bottom. It shows the entire workflow in a few calls:

1. `ask_settings()` collects your folder, filter settings and window duration.
2. `find_recordings()` identifies each person's recordings and session groups.
3. `prepare_recordings()` filters the EEG, makes windows, extracts mean/std/variance, and saves quality checks.
4. `run_svm()` trains and evaluates the original SVM baseline using separate recording groups.
5. Choose `preview` to show the PyTorch CNN, or `tune` to run six Optuna training trials. See OPTUNA-GUIDE.md.

The original preprocessing, SVM settings, splitting, thresholds and saved baseline results are preserved. MNE processes EEG; scikit-learn runs the SVM on the CPU. PyTorch replaces the optional TensorFlow CNN.

## The two classes

`EEGDataset` is a collection of windows and labels. `__init__` prepares it, `__len__` reports its size, and `__getitem__` returns one example. A label of 1 means the claimed person; 0 means another enrolled person.

`DataLoader` is PyTorch's batch maker. We use it directly rather than creating an extra wrapper class. The train, validation and test loaders use the same recording split as the SVM. Only the training loader shuffles windows.

`EEGCNN` stores layers in `__init__`. Its `forward()` method shows the exact order in which data passes through them:

```text
EEG window batch: (batch, channels, time)
  -> Conv1d(16) -> ReLU -> BatchNorm -> MaxPool
  -> Conv1d(32) -> ReLU -> BatchNorm
  -> mean across time (global average pooling)
  -> Linear(16) -> ReLU -> Dropout(0.5)
  -> Linear(1) -> Sigmoid
Output: (batch, 1)
```

This keeps the original layer arrangement. Padding 3 keeps the length of a kernel-7 convolution unchanged. PyTorch already accepts the saved window order; no transpose is needed. BatchNorm uses epsilon 0.001 and momentum 0.01 to match the Keras settings' update convention. The frameworks do not guarantee identical random initial weights or batch-normalization numerical results.

## GPU in three steps

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EEGCNN(number_of_channels).to(device)
windows = windows.to(device)
```

The model and its inputs must be on the same device. The preview prints the selected device and GPU name. `model.eval()` turns off training behavior such as dropout; `torch.no_grad()` avoids storing gradients during the preview.

## What the current script does

The preview remains an untrained architecture candidate with real DataLoaders and one forward pass. The optional `tune` mode now trains a fixed-architecture CNN in six Optuna trials, using training-only normalization and validation loss. It saves the best model for one claimed person. It does not use test data or establish authentication accuracy. Read OPTUNA-GUIDE.md for the short search workflow and saved files.

For example, two groups for S001 and one for S002 are insufficient. The original requirement is at least three independent groups per person, so a normal run saves preprocessing results and stops before fitting the SVM or offering the CNN preview. That guard remains in place. CPU/GPU model checks can still be performed separately with synthetic data; they do not establish real EEG authentication accuracy.

The original V1 `hello.py` remains at the repository root and in Git history. The recordings and previous result folders are preserved.
