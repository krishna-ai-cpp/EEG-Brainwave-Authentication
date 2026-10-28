"""A small, optional six-trial CNN search. Read tune_cnn() first.

Optuna chooses settings -> train on training groups -> compare validation loss.
The CNN architecture stays fixed. Test recordings never enter this search.
"""

import json
import os

import numpy as np
import optuna
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from hello import EEGDataset, EEGCNN


def prepare_tuning_data(feature_table, results_folder, claimed_subject):
    """Reuse the existing session split, then normalize using training data only."""
    counts = feature_table.groupby("subject_id")["group_id"].nunique()
    if feature_table["subject_id"].nunique() < 2 or (counts < 3).any():
        raise ValueError("CNN tuning needs at least two people and three independent groups per person.")
    if claimed_subject not in feature_table["subject_id"].unique():
        raise ValueError("Choose a claimed subject from the feature table.")
    if set(feature_table["split"].unique()) != {"train", "validation", "test"}:
        raise ValueError("Use the train/validation/test split created by run_svm().")
    if (feature_table.groupby("group_id")["split"].nunique() > 1).any():
        raise ValueError("A session group crossed splits. Correct the split before tuning.")
    if (feature_table.groupby("recording_id")["split"].nunique() > 1).any():
        raise ValueError("A recording crossed splits. Correct the split before tuning.")

    train_rows = feature_table[feature_table["split"] == "train"]
    validation_rows = feature_table[feature_table["split"] == "validation"]
    # Only these two datasets are loaded. No test window file is opened here.
    train_data = EEGDataset(train_rows, results_folder, claimed_subject)
    validation_data = EEGDataset(validation_rows, results_folder, claimed_subject)
    for dataset in [train_data, validation_data]:
        if set(dataset.labels) != {0.0, 1.0}:
            raise ValueError("Training and validation each need genuine and impostor examples.")

    training_windows = np.asarray(train_data.windows, dtype=np.float32)
    if training_windows.shape[2] < 4:
        raise ValueError("CNN training needs at least 4 time samples per window.")
    if not np.isfinite(training_windows).all():
        raise ValueError("Training windows contain invalid numbers.")

    # One mean/std per channel. Shape (channels, 1) matches every window.
    # Axes 0 and 2 mean: average across training windows and their time samples.
    channel_mean = training_windows.mean(axis=(0, 2))[:, None]
    channel_std = training_windows.std(axis=(0, 2))[:, None]
    channel_std[channel_std < 1e-8] = 1e-8  # Avoid division by zero.

    # Apply the SAME training statistics to validation; never fit on validation.
    for dataset in [train_data, validation_data]:
        for index, window in enumerate(dataset.windows):
            if window.shape != training_windows.shape[1:] or not np.isfinite(window).all():
                raise ValueError("Check that all CNN windows have the same shape and finite values.")
            dataset.windows[index] = ((window - channel_mean) / channel_std).astype(np.float32)
    return train_data, validation_data, channel_mean, channel_std


def train_one_epoch(model, loader, optimizer, loss_function, device):
    """One pass over the training batches: predict, measure error, update weights."""
    model.train()
    for windows, labels in loader:
        windows = windows.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # Clear gradients left by the previous batch.
        probabilities = model(windows)
        loss = loss_function(probabilities, labels)
        if not torch.isfinite(loss).item():
            raise ValueError("Training loss became invalid. Review the data/settings.")
        loss.backward()       # Find how each weight affected the error.
        optimizer.step()      # Adjust weights to reduce that error.


def get_validation_loss(model, loader, loss_function, device):
    """Measure error on held-out validation windows without updating weights."""
    model.eval()
    total_loss = 0.0
    total_windows = 0
    with torch.no_grad():
        for windows, labels in loader:
            windows = windows.to(device)
            labels = labels.to(device)
            probabilities = model(windows)
            loss = loss_function(probabilities, labels)
            total_loss += loss.item() * len(windows)
            total_windows += len(windows)
    return total_loss / total_windows


def tune_cnn(feature_table, results_folder, channel_names, claimed_subject=None, epochs=5):
    """Try SIX settings and save the model with the lowest validation loss.

    epochs=5 keeps the introductory search short; it is not a claim of convergence.
    One study tunes one claimed person against the other enrolled people.
    """
    if epochs < 1:
        raise ValueError("Use at least one training epoch per trial.")
    if claimed_subject is None:
        subjects = sorted(feature_table["subject_id"].unique())
        print("Available subjects:", ", ".join(subjects))
        claimed_subject = input("Subject to tune (Enter for the first): ").strip() or subjects[0]

    train_data, validation_data, channel_mean, channel_std = prepare_tuning_data(
        feature_table, results_folder, claimed_subject
    )
    if len(channel_names) != len(channel_mean):
        raise ValueError("The channel list does not match the saved windows.")
    search_folder = os.path.join(results_folder, "optuna_cnn")
    os.makedirs(search_folder)  # Refuse to overwrite an earlier search.
    best_model_path = os.path.join(search_folder, "best_cnn.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Optuna training device:", device)
    print("Six trials;", epochs, "epochs per trial; fixed CNN architecture.")

    # A seed makes the suggested settings repeatable. Two initial trials explore
    # the range; later suggestions use results from the trials already completed.
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=2)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial):
        # Optuna asks: how well does ONE combination of settings work?
        learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.01, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        torch.manual_seed(42)  # Give each trial the same initial random seed.
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
        validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, num_workers=0)
        model = EEGCNN(len(channel_names)).to(device)
        loss_function = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            train_one_epoch(model, train_loader, optimizer, loss_function, device)
        validation_loss = get_validation_loss(model, validation_loader, loss_function, device)
        if not np.isfinite(validation_loss):
            raise ValueError("Validation loss is invalid. Review the data/settings.")
        print(f"Trial {trial.number + 1}/6: lr={learning_rate:.5f}, "
              f"batch={batch_size}, validation loss={validation_loss:.4f}")

        # Save a checkpoint only when this trial improves the best score so far.
        if trial.number == 0 or validation_loss < study.best_value:
            torch.save({
                "model_state": model.state_dict(),
                "channel_mean": torch.tensor(channel_mean),
                "channel_std": torch.tensor(channel_std),
                "channel_names": list(channel_names),
                "claimed_subject": claimed_subject,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
            }, best_model_path)
        return validation_loss  # Lower is better; test data is never consulted.

    study.optimize(objective, n_trials=6, n_jobs=1)

    rows = []
    for trial in study.trials:
        rows.append({"trial": trial.number + 1,
                     "learning_rate": trial.params["learning_rate"],
                     "batch_size": trial.params["batch_size"],
                     "validation_loss": trial.value, "state": trial.state.name})
    pd.DataFrame(rows).to_csv(os.path.join(search_folder, "trials.csv"), index=False)
    summary = {"claimed_subject": claimed_subject, "n_trials": 6,
               "epochs_per_trial": epochs, "best_trial": study.best_trial.number + 1,
               "best_parameters": study.best_params, "validation_loss": study.best_value,
               "objective": "window-averaged validation binary cross-entropy",
               "test_used": False, "device": str(device)}
    with open(os.path.join(search_folder, "best_parameters.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print("Best settings:", study.best_params)
    print("Best validation loss:", study.best_value)
    print("Saved best weights AND training normalization to:", best_model_path)
    print("No test evaluation, authentication accuracy, or SVM/CNN fusion is claimed.")
    return study
