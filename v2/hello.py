"""EEG authentication: recordings -> checks/features -> SVM -> optional CNN step.

Run from Ubuntu:  .venv/bin/python hello.py
Keep each session together by giving its recordings the same group_id.
Choose a CNN preview, or an optional six-trial Optuna training search.
The preview is untrained. Tuning uses training/validation groups, never test data.
"""

import os
import mne
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


def ask_settings():
    """Ask for the data, a NEW results folder, and preprocessing settings."""
    data_folder = input("Enter the folder containing your EEG recordings: ").strip().strip('"')
    if not os.path.isdir(data_folder):
        raise FileNotFoundError(data_folder)

    results_folder = input("Folder for NEW results (Enter for baseline_results): ").strip().strip('"') or "baseline_results"
    if os.path.exists(results_folder):
        raise FileExistsError("Choose a new results folder so earlier results stay unchanged.")
    os.makedirs(results_folder)

    low_frequency = float(input("Enter low frequency (Hz): "))
    high_frequency = float(input("Enter high frequency (Hz): "))
    window_seconds = float(input("Enter window time (seconds): "))
    artifact_limit_uv = float(input("Peak-to-peak review limit in microvolts (Enter for 150): ") or "150")

    # Reject invalid numbers before processing any recordings.
    if not all(np.isfinite([low_frequency, high_frequency, window_seconds, artifact_limit_uv])):
        raise ValueError("Enter finite numbers.")
    if low_frequency < 0 or low_frequency >= high_frequency or window_seconds <= 0 or artifact_limit_uv <= 0:
        raise ValueError("Check frequencies, window duration and review limit.")
    return data_folder, results_folder, low_frequency, high_frequency, window_seconds, artifact_limit_uv


def find_recordings(data_folder):
    """Read recordings.csv, or ask which person/session each file belongs to."""
    manifest = os.path.join(data_folder, "recordings.csv")
    if os.path.isfile(manifest):
        records = pd.read_csv(manifest, dtype=str, keep_default_na=False)
        if not {"path", "subject_id", "group_id"}.issubset(records.columns):
            raise ValueError("recordings.csv needs path, subject_id and group_id.")
    else:
        # Without a manifest, find recordings and ask for their identities.
        rows = []
        for root, directories, filenames in os.walk(data_folder):
            # Skip system and environment folders.
            for d in directories.copy():
                if d in ["Lib", "site-packages", ".git", "__pycache__"]:
                    directories.remove(d)
            for filename in sorted(filenames):
                if filename.lower().endswith((".edf", ".bdf", ".fif", ".fif.gz", ".set", ".vhdr", ".cnt", ".gdf")):
                    path = os.path.join(root, filename)
                    file_id = os.path.splitext(filename)[0]

                    # For a name such as S001R03, the subject is S001.
                    looks_like_srr = (
                        len(file_id) >= 5
                        and file_id[0] == "S"
                        and file_id[1:4].isdigit()
                        and file_id[4] == "R"
                    )
                    if looks_like_srr:
                        subject_id = file_id[:4]
                    else:
                        subject_id = input(f"True subject ID for {filename}: ").strip()

                    group_id = input(
                        f"Session/group ID for {filename} (Enter if this is an independent recording): "
                    ).strip() or file_id

                    rows.append({"path": os.path.abspath(path),
                                 "subject_id": subject_id,
                                 "group_id": group_id})
        records = pd.DataFrame(rows, columns=["path", "subject_id", "group_id"])

    if records.empty:
        raise ValueError("No supported EEG recordings found. MAT/CSV signals need their own layout/units described.")
    if (records[["path", "subject_id", "group_id"]] == "").any().any():
        raise ValueError("Every recording needs a path, true identity and group ID.")
    return records


def prepare_recordings(records, data_folder, results_folder, low_frequency, high_frequency, window_seconds, artifact_limit_uv):
    """Filter EEG, make windows, extract statistics, and save the same checks as before."""
    table = []           # Feature rows from all recordings.
    quality = []         # One quality summary per recording.
    window_checks = []   # Artifact flags for individual windows.
    channel_names = None  # The first recording sets the channel order.
    sampling_rate = None  # All recordings must use the same sampling rate.

    for recording_id, row in records.iterrows():
        path = row["path"]
        if not os.path.isabs(path):
            path = os.path.join(data_folder, path)
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

        # Load the recording and keep only EEG channels.
        if path.lower().endswith(".edf"):
            raw = mne.io.read_raw_edf(path, preload=True)
        else:
            raw = mne.io.read_raw(path, preload=True)
        raw.pick("eeg")
        print(raw.info)
        print(raw.ch_names)

        # Check for missing/duplicate channels and channels marked bad.
        if not raw.ch_names or len(raw.ch_names) != len(set(raw.ch_names)):
            raise ValueError("Missing or duplicate EEG channels.")
        if raw.info["bads"]:
            raise ValueError(f"Review channels marked bad: {raw.info['bads']}")

        # Every recording must have the same channels and sampling rate.
        if channel_names is None:
            channel_names = raw.ch_names.copy()
            sampling_rate = float(raw.info["sfreq"])
        if set(raw.ch_names) != set(channel_names):
            raise ValueError(f"EEG channels differ in {path}")
        raw.reorder_channels(channel_names)
        if not np.isclose(raw.info["sfreq"], sampling_rate):
            raise ValueError(f"Sampling rate differs in {path}")
        if high_frequency >= sampling_rate / 2:
            raise ValueError("High frequency must be below half the sampling frequency.")

        # Keep unfiltered signals for artifact checks and preview plots.
        raw_data = raw.get_data()
        if not np.isfinite(raw_data).all():
            raise ValueError(f"NaN or Inf values in {path}")

        # ── band-pass filter + non-overlapping windows ──
        filtered_raw = raw.copy()
        filtered_raw.filter(l_freq=low_frequency, h_freq=high_frequency)
        data = filtered_raw.get_data()
        win_size = int(filtered_raw.info["sfreq"] * window_seconds)  # Samples per window.
        if win_size < 1:
            raise ValueError("Window duration is too short.")

        start = 0
        windows = []
        while start + win_size <= data.shape[1]:  # Keep complete windows only.
            end = start + win_size
            windows.append(data[:, start:end])
            start += win_size
        if not windows:
            raise ValueError(f"No complete windows in {path}")

        # Three statistics per channel: mean, standard deviation, variance.
        features = []
        for window in windows:
            channel_means = np.mean(window, axis=1)  # Average over time.
            channel_stds = np.std(window, axis=1)    # How much the signal varies.
            channel_variances = np.var(window, axis=1)  # Squared standard deviation.
            features.append((channel_means, channel_stds, channel_variances))

        # Join (mean | std | variance) into one feature row per window.
        stat = []
        for channel_means, channel_stds, channel_variances in features:
            row_features = np.concatenate((channel_means, channel_stds, channel_variances))
            stat.append(row_features)
        feature_values = np.array(stat)

        # Keep feature columns in this order: all means, all stds, all variances.
        mean_cols = [f"mean_{ch}" for ch in channel_names]
        std_cols = [f"std_{ch}" for ch in channel_names]
        var_cols = [f"var_{ch}" for ch in channel_names]
        feature_columns = mean_cols + std_cols + var_cols
        recording_features = pd.DataFrame(feature_values, columns=feature_columns)
        recording_features.insert(0, "subject_id", row["subject_id"])
        recording_features.insert(1, "group_id", row["group_id"])
        recording_features.insert(2, "recording_id", recording_id)
        recording_features.insert(3, "window_id", np.arange(len(recording_features)))
        table.append(recording_features)

        # Flag possible artifacts for review; do not delete or repair signals.
        flagged = 0
        for window_id, window in enumerate(windows):
            raw_window = raw_data[:, window_id * win_size:(window_id + 1) * win_size]
            high_amplitude = bool((np.ptp(raw_window, axis=1) * 1e6 > artifact_limit_uv).any())
            flat_channel = bool((np.std(raw_window, axis=1) * 1e6 < 0.01).any())

            t0 = window_id * win_size / sampling_rate
            t1 = (window_id + 1) * win_size / sampling_rate
            annotated_bad = False
            for i in range(len(raw.annotations)):
                description = str(raw.annotations.description[i]).lower()
                annotation_start = raw.annotations.onset[i] - raw.first_time
                annotation_end = annotation_start + raw.annotations.duration[i]
                if description.startswith("bad"):
                    if annotation_start < t1 and annotation_end > t0:  # Overlapping times.
                        annotated_bad = True
                        break

            flag = high_amplitude or flat_channel or annotated_bad
            flagged += int(flag)
            window_checks.append({"recording_id": recording_id, "window_id": window_id,
                                  "high_amplitude": high_amplitude, "flat_channel": flat_channel,
                                  "annotated_bad": annotated_bad})

        # Save whole filtered windows for the CNN, rather than just statistics.
        np.save(os.path.join(results_folder, f"recording_{recording_id}_windows.npy"), np.asarray(windows))

        # Plot the first window before and after filtering.
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(win_size) / sampling_rate, raw_data[:, 0:win_size].T * 1e6)
        plt.ylabel("Raw EEG (microvolts)")
        plt.title(f"First window: {os.path.basename(path)}")
        plt.subplot(2, 1, 2)
        plt.plot(np.arange(win_size) / sampling_rate, windows[0].T * 1e6)
        plt.xlabel("Time (seconds)")
        plt.ylabel("EEG (microvolts)")
        plt.title(f"Filtered EEG: {low_frequency}-{high_frequency} Hz")
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, f"recording_{recording_id}_preview.png"))
        plt.close()

        quality.append({"recording_id": recording_id, "path": path, "subject_id": row["subject_id"],
                        "group_id": row["group_id"],
                        "channels": "|".join(channel_names), "sampling_hz": sampling_rate,
                        "duration_s": raw.n_times / sampling_rate, "windows": len(windows),
                        "flagged_windows": flagged, "discarded_tail_samples": data.shape[1] - start})


    # Save all feature/quality tables and print a short summary.
    feature_table = pd.concat(table, ignore_index=True)
    x = feature_table[feature_columns].to_numpy()
    if not np.isfinite(x).all():
        raise ValueError("Extracted features contain NaN or Inf.")
    feature_table.to_csv(os.path.join(results_folder, "features.csv"), index=False)
    pd.DataFrame(quality).to_csv(os.path.join(results_folder, "recording_checks.csv"), index=False)
    pd.DataFrame(window_checks).to_csv(os.path.join(results_folder, "window_checks.csv"), index=False)
    counts = feature_table.groupby("subject_id").agg(recordings=("recording_id", "nunique"),
                                           groups=("group_id", "nunique"), windows=("window_id", "size"))
    counts.to_csv(os.path.join(results_folder, "subject_counts.csv"))

    print("Missing feature values:", feature_table[feature_columns].isna().sum().sum())
    total_flagged = 0
    for q in quality:
        total_flagged += q["flagged_windows"]
    print("Potential artifact windows:", total_flagged)
    print(counts)
    print("Saved validation results to:", os.path.abspath(results_folder))
    print("At least 3 independent groups per person are required for the next step.")
    return channel_names, sampling_rate, total_flagged


def run_svm(results_folder, channel_names, total_flagged):
    """Keep sessions separate, fit on training data, choose thresholds on validation, then test."""
    feature_table = pd.read_csv(os.path.join(results_folder, "features.csv"), dtype={"subject_id": str, "group_id": str})

    # Rebuild the same feature column order used during preprocessing.
    feature_columns = []
    for stat_name in ["mean", "std", "var"]:
        for ch in channel_names:
            feature_columns.append(f"{stat_name}_{ch}")

    if not np.isfinite(feature_table[feature_columns].to_numpy()).all() or feature_table.isna().any().any():
        raise ValueError("Missing or invalid values: review Step 1.")
    if feature_table["subject_id"].nunique() < 2:
        raise SystemExit("Need at least two people for genuine-versus-impostor evaluation.")

    counts = feature_table.groupby("subject_id")["group_id"].nunique()
    if (counts < 3).any():
        message = ("SVM not fitted: need at least 3 independent recordings/session groups per person. Found: "
                   + str(counts.to_dict()))
        with open(os.path.join(results_folder, "baseline_status.txt"), "w", encoding="utf-8") as status_file:
            status_file.write(message)
        raise SystemExit(message)

    if total_flagged > 0:
        reviewed = input("Review recording previews/window_checks.csv. Have the artifact flags been investigated? Type yes to continue: ").strip().lower()
        if reviewed != "yes":
            raise SystemExit("Review artifacts before fitting the baseline. No windows were automatically removed.")

    # Split whole recording/session groups so related windows stay together.
    feature_table["split"] = ""
    for subject_id, group in feature_table.groupby("subject_id", sort=True):
        groups = sorted(group["group_id"].unique())
        # Reserve 20% of groups for testing, then 20% of the remainder for validation.
        train_groups, test_groups = train_test_split(groups, test_size=0.2, random_state=42)
        train_groups, val_groups = train_test_split(train_groups, test_size=0.2, random_state=42)
        train_rows = group.index[group["group_id"].isin(train_groups)]
        val_rows = group.index[group["group_id"].isin(val_groups)]
        test_rows = group.index[group["group_id"].isin(test_groups)]
        feature_table.loc[train_rows, "split"] = "train"
        feature_table.loc[val_rows, "split"] = "validation"
        feature_table.loc[test_rows, "split"] = "test"

    # A session shared by multiple people must also stay in just one split.
    if (feature_table.groupby("group_id")["split"].nunique() > 1).any():
        raise SystemExit("A shared session group crossed splits. Use a coordinated session protocol before evaluation.")
    assert (feature_table.groupby("recording_id")["split"].nunique() == 1).all()
    feature_table[["recording_id", "subject_id", "group_id", "split"]].drop_duplicates().to_csv(
        os.path.join(results_folder, "recording_split.csv"), index=False)

    # Learn scaling from training windows only, then apply it to the other splits.
    train = feature_table["split"] == "train"
    validation = feature_table["split"] == "validation"
    test = feature_table["split"] == "test"
    scaler = StandardScaler()
    x_train = scaler.fit_transform(feature_table.loc[train, feature_columns])
    x_val = scaler.transform(feature_table.loc[validation, feature_columns])
    x_test = scaler.transform(feature_table.loc[test, feature_columns])
    y_train = feature_table.loc[train, "subject_id"].to_numpy()

    # One SVM per enrolled person: genuine=1, other enrolled people=0.
    results = []
    all_scores = []
    plt.figure(figsize=(6, 5))
    for subject_id in sorted(feature_table["subject_id"].unique()):
        svm_y_train = (y_train == subject_id).astype(int)
        svm = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
        svm.fit(x_train, svm_y_train)

        # Average window scores to produce one score per recording.
        val_table = feature_table.loc[validation, ["recording_id", "subject_id"]].copy()
        val_table["score"] = svm.decision_function(x_val)
        val_table = val_table.groupby(["recording_id", "subject_id"], as_index=False)["score"].mean()
        val_y = (val_table["subject_id"] == subject_id).astype(int).to_numpy()

        # Choose the validation threshold where false acceptance/rejection are closest.
        far, true_acceptance, thresholds = roc_curve(val_y, val_table["score"], drop_intermediate=False)
        frr = 1 - true_acceptance
        finite = np.isfinite(thresholds)  # Exclude the infinite threshold.
        valid_thresholds = thresholds[finite]
        difference = np.abs(far[finite] - frr[finite])
        chosen = np.argmin(difference)
        threshold = float(valid_thresholds[chosen])

        # Freeze the threshold before scoring test recordings.
        test_table = feature_table.loc[test, ["recording_id", "subject_id"]].copy()
        test_table["score"] = svm.decision_function(x_test)
        test_table = test_table.groupby(["recording_id", "subject_id"], as_index=False)["score"].mean()
        test_y = (test_table["subject_id"] == subject_id).astype(int).to_numpy()
        accepted = (test_table["score"].to_numpy() >= threshold).astype(int)
        test_far = float(accepted[test_y == 0].mean())  # Impostors incorrectly accepted.
        test_frr = float((1 - accepted[test_y == 1]).mean())  # Genuine people rejected.

        # Approximate equal error rate for reporting; never tune on test data.
        test_roc_far, test_tpr, test_thresholds = roc_curve(test_y, test_table["score"], drop_intermediate=False)
        test_roc_frr = 1 - test_tpr
        k = np.argmin(np.abs(test_roc_far - test_roc_frr))
        eer_approx = float((test_roc_far[k] + test_roc_frr[k]) / 2)

        result = {}
        result["claimed_subject"] = subject_id
        result["validation_threshold"] = threshold
        result["test_genuine_recordings"] = int((test_y == 1).sum())
        result["test_impostor_recordings"] = int((test_y == 0).sum())
        result["FAR"] = test_far
        result["FRR"] = test_frr
        result["EER_approx"] = eer_approx
        result["ROC_AUC"] = roc_auc_score(test_y, test_table["score"])
        result["accuracy"] = accuracy_score(test_y, accepted)
        results.append(result)

        for split_name, scores in [("validation", val_table), ("test", test_table)]:
            scores = scores.copy()
            scores["claimed_subject"] = subject_id
            scores["genuine"] = (scores["subject_id"] == subject_id).astype(int)
            scores["threshold"] = threshold
            scores["accepted"] = (scores["score"] >= threshold).astype(int)
            scores["split"] = split_name
            all_scores.append(scores)
        plt.plot(test_roc_far, test_tpr, label=subject_id)

    metrics = pd.DataFrame(results)
    metrics.to_csv(os.path.join(results_folder, "svm_metrics.csv"), index=False)
    pd.concat(all_scores, ignore_index=True).to_csv(os.path.join(results_folder, "svm_recording_scores.csv"), index=False)
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False acceptance rate")
    plt.ylabel("True acceptance rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, "svm_test_roc.png"))
    plt.close()

    print(metrics.to_string(index=False))
    print("Macro means:", metrics[["FAR", "FRR", "EER_approx", "ROC_AUC", "accuracy"]].mean().to_dict())
    print("Rates are fractions (0.05 means 5%). Interpret alongside the recording counts.")
    return feature_table


class EEGDataset(Dataset):
    """One saved EEG window and its genuine/impostor label at a time.

    Dataset = the collection of examples; DataLoader = the batch maker.
    Each window already has PyTorch's order: (channels, time_samples).
    """

    def __init__(self, feature_rows, results_folder, claimed_subject):
        self.windows = []
        self.labels = []

        # A recording stays in the SAME split used by the SVM.
        recordings = feature_rows[["recording_id", "subject_id"]].drop_duplicates()
        for _, recording in recordings.iterrows():
            recording_id = recording["recording_id"]
            path = os.path.join(results_folder, f"recording_{recording_id}_windows.npy")
            windows = np.load(path)
            label = 1.0 if recording["subject_id"] == claimed_subject else 0.0

            for window in windows:
                self.windows.append(window)
                self.labels.append(label)

    def __len__(self):
        # DataLoader asks: how many examples are available?
        return len(self.windows)

    def __getitem__(self, index):
        # DataLoader asks: give me example number 'index'.
        window = torch.tensor(self.windows[index], dtype=torch.float32)
        label = torch.tensor([self.labels[index]], dtype=torch.float32)
        return window, label


def create_cnn_loaders(feature_table, results_folder, claimed_subject, batch_size=32):
    """Reuse the recording split; shuffle only the training examples."""
    train_rows = feature_table[feature_table["split"] == "train"]
    validation_rows = feature_table[feature_table["split"] == "validation"]
    test_rows = feature_table[feature_table["split"] == "test"]

    train_data = EEGDataset(train_rows, results_folder, claimed_subject)
    validation_data = EEGDataset(validation_rows, results_folder, claimed_subject)
    test_data = EEGDataset(test_rows, results_folder, claimed_subject)

    # num_workers=0 keeps loading simple: everything runs in this Python process.
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, validation_loader, test_loader


class EEGCNN(nn.Module):
    """The same CNN layers as the original Keras candidate, written step by step."""

    def __init__(self, number_of_channels):
        super().__init__()  # Let PyTorch keep track of this model's layers/weights.
        self.conv1 = nn.Conv1d(number_of_channels, 16, kernel_size=7, padding=3)
        self.batch_norm1 = nn.BatchNorm1d(16, eps=0.001, momentum=0.01)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=7, padding=3)
        self.batch_norm2 = nn.BatchNorm1d(32, eps=0.001, momentum=0.01)
        self.hidden = nn.Linear(32, 16)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, windows):
        # Input: (batch, EEG_channels, time_samples).
        values = self.relu(self.conv1(windows))
        values = self.batch_norm1(values)
        values = self.pool(values)
        values = self.relu(self.conv2(values))
        values = self.batch_norm2(values)

        # Global average pooling: average each learned feature across time.
        # Shape: (batch, 32, shorter_time) -> (batch, 32).
        values = values.mean(dim=2)
        values = self.relu(self.hidden(values))
        values = self.dropout(values)
        return self.sigmoid(self.classifier(values))  # Shape: (batch, 1).


def preview_cnn(feature_table, results_folder, channel_names, window_size):
    """Show the model and pass ONE batch through it; no training or evaluation."""
    if window_size < 2:
        raise ValueError("The CNN pooling layer needs at least 2 samples per window.")

    # CUDA runs the neural network on the NVIDIA GPU. CPU is the fallback.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        # One thread reduced small-window CPU latency in this laptop's benchmark.
        torch.set_num_threads(1)
    print("CNN device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Demonstrate one genuine-vs-impostor task, using the first enrolled person.
    claimed_subject = sorted(feature_table["subject_id"].unique())[0]
    train_loader, validation_loader, test_loader = create_cnn_loaders(
        feature_table, results_folder, claimed_subject
    )
    model = EEGCNN(len(channel_names)).to(device)
    # PyTorch defines loss/optimizer explicitly; Keras called this "compile".
    # These are ready for future training, but no learning happens in this preview.
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(model)
    print("Training setup:", type(loss_function).__name__, "/", type(optimizer).__name__, "lr=0.001")
    print("Preview claimed subject:", claimed_subject)
    print("Windows in train / validation / test:",
          len(train_loader.dataset), len(validation_loader.dataset), len(test_loader.dataset))

    model.eval()  # Disable dropout and use stored batch-normalization statistics.
    with torch.no_grad():  # A preview needs no gradients for learning.
        for windows, labels in train_loader:
            windows = windows.to(device)  # Put the input on the SAME device as the model.
            probabilities = model(windows)
            print("Input shape (batch, channels, time):", tuple(windows.shape))
            print("Output shape (batch, 1):", tuple(probabilities.shape))
            break

    print("CNN preview only: weights are untrained; no CNN accuracy is established.")
    # Future training must normalize channels using TRAINING data only, then
    # train one model per claimed person with BCELoss and Adam(lr=0.001).
    # Choose authentication thresholds on validation recordings, freeze them,
    # and average window scores per test recording, exactly as in the SVM.
    # Do not combine SVM/CNN scores before both have been evaluated separately.


def main():
    # Read this short function first: it is the complete project workflow.
    data_folder, results_folder, low_frequency, high_frequency, window_seconds, artifact_limit_uv = ask_settings()
    records = find_recordings(data_folder)
    channel_names, sampling_rate, total_flagged = prepare_recordings(
        records, data_folder, results_folder, low_frequency, high_frequency, window_seconds, artifact_limit_uv
    )
    feature_table = run_svm(results_folder, channel_names, total_flagged)

    cnn_action = input("CNN: type preview, tune, or press Enter to finish: ").strip().lower()
    if cnn_action in ["preview", "yes"]:
        preview_cnn(feature_table, results_folder, channel_names, int(sampling_rate * window_seconds))
    elif cnn_action == "tune":
        # Optuna is imported only for a search, never for a normal CNN prediction.
        from optuna_tuning import tune_cnn
        tune_cnn(feature_table, results_folder, channel_names)


if __name__ == "__main__":
    main()
