# Six simple Optuna trials for your EEG CNN

Run the project in Ubuntu:

```bash
cd EEG-Brainwave-Authentication/v2
.venv/bin/python hello.py
```

After preprocessing and SVM evaluation, the CNN prompt accepts:

- `preview`: show the untrained architecture and one forward pass.
- `tune`: run the six-trial Optuna search.
- Enter: finish without either optional step.

The original `yes` answer still selects the preview.

For example, a dataset with two groups for S001 and one for S002 is insufficient. The existing requirement of at least three independent groups per person remains. Until you have enough data, the script saves preprocessing results and stops before SVM fitting or the Optuna prompt. Do not relabel windows from the same session as separate groups to bypass this check.

## What is a trial?

A trial is one experiment with a particular combination of settings:

```text
Choose a learning rate and training batch size
                   ↓
Create a fresh copy of the SAME CNN
                   ↓
Train it on training windows for 5 epochs
                   ↓
Measure loss on held-out validation windows
                   ↓
Keep the checkpoint if it is the best so far
```

This is repeated exactly six times:

```python
study.optimize(objective, n_trials=6, n_jobs=1)
```

`objective(trial)` is simply the function that runs one experiment and returns its validation loss. `direction="minimize"` tells Optuna that a smaller loss is better. It chooses the best of six tested settings, not a guaranteed global optimum.

## The settings that change

| Setting | Choices/range | Meaning |
|---|---|---|
| Learning rate | 0.0001 to 0.01, logarithmic sampling | How large Adam's learning steps are |
| Training batch size | 16, 32 or 64 | How many windows are used in each training update |

Five epochs means five passes through the training data per trial. This is a short introductory search, not evidence that training has converged. `epochs=5` is the default argument of `tune_cnn()` and can be changed there later.

The model architecture, kernel sizes, channel counts, dropout and inference code are unchanged. The selected training batch size does not force you to wait for that many windows during prediction: the model can still receive one window at a time.

## The small functions in optuna_tuning.py

1. `prepare_tuning_data()` checks the existing group split and loads only training/validation windows. It computes each channel's mean and standard deviation from training data, then uses those same values to normalize validation data.
2. `train_one_epoch()` performs the usual PyTorch steps: clear gradients, predict, calculate loss, run backward, update weights.
3. `get_validation_loss()` measures error without learning. It averages binary cross-entropy across validation windows.
4. `tune_cnn()` creates the study, runs the six trials and saves the best checkpoint.

`hello.py` remains the entry point and contains the Dataset/model classes and the original preprocessing/SVM workflow. The separate tuning file keeps the main script readable. Optuna is imported only when you choose `tune`.

## What gets saved?

Inside your new results folder, `optuna_cnn/` contains:

- `trials.csv`: all six trials and their validation losses.
- `best_parameters.json`: the selected settings, subject, epochs, device and validation loss.
- `best_cnn.pt`: selected model weights, channel order, claimed identity, and the training mean/std needed for future predictions.

One search tunes one claimed person against the other enrolled people. Choose the subject when prompted. It does not train a separate model for every person in that one search.

To use a saved model later, load its weights into `EEGCNN`, keep the saved channel order, normalize new windows with its saved training mean/std, and use `model.eval()` plus `torch.no_grad()`. No Optuna search runs during prediction.

## Speed and evaluation

The earlier 17x result measured warmed CNN inference on prepared windows. Optuna adds offline training/search time; it is not a way to make every training run 17x faster. Keeping the exact CNN forward path avoids adding Optuna overhead to predictions, but an exact speed ratio depends on the computer load and measurement method and must be rechecked.

The CPU preview fallback now uses `torch.set_num_threads(1)`, a simple setting that reduced single-window CPU latency on this laptop. For standalone single-window CPU predictions, apply the same setting before calling the model. This favors response latency; larger CPU batches may benefit from more threads. GPU execution is still selected automatically when available.

The new benchmark is in [benchmarks/README.md](benchmarks/README.md). With one thread for both frameworks, the measured single-window medians were 12.165 ms for TensorFlow CPU, 0.317 ms for PyTorch CPU, and 0.564 ms for PyTorch GPU. The same-CPU ratio was about 38x in this measurement. A separate four-thread repeat measured about 12.6x; the earlier 17x is not a fixed property. These compare untrained architecture candidates with the same prepared input as before, not a trained authentication model or end-to-end pipeline.

The search never reads test windows, chooses thresholds from test results, or reports authentication accuracy. The saved CNN is trained only if tuning actually completes on eligible data. Separate held-out recording-level evaluation and validation-based threshold selection are still needed before using it for authentication. Synthetic smoke tests verify execution, not real EEG accuracy.

Optuna reference: https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/001_first.html
