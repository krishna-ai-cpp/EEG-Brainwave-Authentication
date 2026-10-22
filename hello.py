# ============================================================
# EEG brain-wave authentication script.
# Idea (meeting me aise samjhao):
#   STEP 1 -> EEG files load karo, check karo, aur har window ka
#            simple feature set banao (mean / std / var per channel).
#   STEP 2 -> SVM se sikh Lo "asli banda (1) vs baaki log (0)".
#   STEP 3 -> chaho to CNN ka architecture dikhao (train nahi hua).
#
# Chalane ke liye: apna EEG data folder + ek NAYA results folder do.
# Ek hi session ki recordings ko SAME group_id dena.
# ============================================================

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


# ── STEP 1a: user se inputs lena ─────────────────────────────
folder = input("Enter the folder containing your EEG recordings: ").strip().strip('"')
if not os.path.isdir(folder):
    raise FileNotFoundError(folder)

output = input("Folder for NEW results (Enter for baseline_results): ").strip().strip('"') or "baseline_results"
if os.path.exists(output):
    raise FileExistsError("Choose a new results folder so earlier results stay unchanged.")
os.makedirs(output)

lf = float(input("Enter low frequency (Hz): "))
hf = float(input("Enter high frequency (Hz): "))
time = float(input("Enter window time (seconds): "))
artifact_limit_uv = float(input("Peak-to-peak review limit in microvolts (Enter for 150): ") or "150")

# thodi input validation (galat number pakad lete hain)
if not all(np.isfinite([lf, hf, time, artifact_limit_uv])):
    raise ValueError("Enter finite numbers.")
if lf < 0 or lf >= hf or time <= 0 or artifact_limit_uv <= 0:
    raise ValueError("Check frequencies, window duration and review limit.")


# ── STEP 1b: kaun-kaun si recordings hain, list banao ────────
# Agar recordings.csv diya hai to seedha wahi padho.
#   columns: path, subject_id, group_id  (same session = same group_id)
manifest = os.path.join(folder, "recordings.csv")
if os.path.isfile(manifest):
    records = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    if not {"path", "subject_id", "group_id"}.issubset(records.columns):
        raise ValueError("recordings.csv needs path, subject_id and group_id.")
else:
    # warna folder ko scan karo aur user se poocho kaun kaun si recording hai
    rows = []
    for root, directories, filenames in os.walk(folder):
        # ye system/venv folders chhod do
        for d in directories.copy():
            if d in ["Lib", "site-packages", ".git", "__pycache__"]:
                directories.remove(d)
        for filename in sorted(filenames):
            if filename.lower().endswith((".edf", ".bdf", ".fif", ".fif.gz", ".set", ".vhdr", ".cnt", ".gdf")):
                path = os.path.join(root, filename)
                file_id = os.path.splitext(filename)[0]

                # naam jaise "S001R03" hai to subject "S001" maan lo, warna poocho
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


# ── STEP 1c: har recording load + check karo, features banao ──
table = []           # har window ki feature row (badme jod dengi)
quality = []         # har recording ki quality info
window_checks = []   # har window ke artifact flags
channels = None      # pehli recording se channel list fix ho jaayegi
sfreq = None         # sampling rate bhi

for recording_id, row in records.iterrows():
    path = row["path"]
    if not os.path.isabs(path):
        path = os.path.join(folder, path)
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    # EEG load karo, sirf EEG channels rakho
    if path.lower().endswith(".edf"):
        raw = mne.io.read_raw_edf(path, preload=True)
    else:
        raw = mne.io.read_raw(path, preload=True)
    raw.pick("eeg")
    print(raw.info)
    print(raw.ch_names)

    # basic check: channel missing/duplicate nahi, koi bad mark nahi
    if not raw.ch_names or len(raw.ch_names) != len(set(raw.ch_names)):
        raise ValueError("Missing or duplicate EEG channels.")
    if raw.info["bads"]:
        raise ValueError(f"Review channels marked bad: {raw.info['bads']}")

    # sab recordings me same channels + same sampling rate hona chahiye
    if channels is None:
        channels = raw.ch_names.copy()
        sfreq = float(raw.info["sfreq"])
    if set(raw.ch_names) != set(channels):
        raise ValueError(f"EEG channels differ in {path}")
    raw.reorder_channels(channels)
    if not np.isclose(raw.info["sfreq"], sfreq):
        raise ValueError(f"Sampling rate differs in {path}")
    if hf >= sfreq / 2:
        raise ValueError("High frequency must be below half the sampling frequency.")

    # raw (unfiltered) data ek baar nikaal lo; baad me flags/preview me use hoga
    raw_data = raw.get_data()
    if not np.isfinite(raw_data).all():
        raise ValueError(f"NaN or Inf values in {path}")

    # ── band-pass filter + non-overlapping windows ──
    filtered_raw = raw.copy()
    filtered_raw.filter(l_freq=lf, h_freq=hf)
    data = filtered_raw.get_data()
    win_size = int(filtered_raw.info["sfreq"] * time)   # ek window me kitne samples
    if win_size < 1:
        raise ValueError("Window duration is too short.")

    start = 0
    windows = []
    while start + win_size <= data.shape[1]:           # poore data ko barabar kaato
        end = start + win_size
        windows.append(data[:, start:end])
        start += win_size
    if not windows:
        raise ValueError(f"No complete windows in {path}")

    # ── har window ke liye 3 simple stats: mean, std, var ──
    features = []
    for w in windows:
        avg = np.mean(w, axis=1)   # har channel ka average
        dev = np.std(w, axis=1)    # har channel ka standard deviation
        v = np.var(w, axis=1)      # har channel ka variance
        features.append((avg, dev, v))

    # (mean | std | var) jod kar har window ki ek row bana do
    stat = []
    for avg, dev, v in features:
        row_features = np.concatenate((avg, dev, v))
        stat.append(row_features)
    statics = np.array(stat)

    # column naam: mean_<ch>, std_<ch>, var_<ch> (isi order me)
    mean_cols = [f"mean_{ch}" for ch in channels]
    std_cols = [f"std_{ch}" for ch in channels]
    var_cols = [f"var_{ch}" for ch in channels]
    col = mean_cols + std_cols + var_cols
    df = pd.DataFrame(statics, columns=col)
    df.insert(0, "subject_id", row["subject_id"])
    df.insert(1, "group_id", row["group_id"])
    df.insert(2, "recording_id", recording_id)
    df.insert(3, "window_id", np.arange(len(df)))
    table.append(df)

    # ── artifact flags: sirf flag karo, kuch delete/repair mat karo ──
    flagged = 0
    for window_id, w in enumerate(windows):
        raw_window = raw_data[:, window_id * win_size:(window_id + 1) * win_size]
        high_amplitude = bool((np.ptp(raw_window, axis=1) * 1e6 > artifact_limit_uv).any())
        flat_channel = bool((np.std(raw_window, axis=1) * 1e6 < 0.01).any())

        t0 = window_id * win_size / sfreq
        t1 = (window_id + 1) * win_size / sfreq
        annotated_bad = False
        for i in range(len(raw.annotations)):
            description = str(raw.annotations.description[i]).lower()
            annotation_start = raw.annotations.onset[i] - raw.first_time
            annotation_end = annotation_start + raw.annotations.duration[i]
            if description.startswith("bad"):
                if annotation_start < t1 and annotation_end > t0:  # overlap hai kya
                    annotated_bad = True
                    break

        flag = high_amplitude or flat_channel or annotated_bad
        flagged += int(flag)
        window_checks.append({"recording_id": recording_id, "window_id": window_id,
                              "high_amplitude": high_amplitude, "flat_channel": flat_channel,
                              "annotated_bad": annotated_bad})

    # future CNN ke liye filtered windows save karo (3 stats ki jagah)
    np.save(os.path.join(output, f"recording_{recording_id}_windows.npy"), np.asarray(windows))

    # pehle window ka raw vs filtered preview image
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(win_size) / sfreq, raw_data[:, 0:win_size].T * 1e6)
    plt.ylabel("Raw EEG (microvolts)")
    plt.title(f"First window: {os.path.basename(path)}")
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(win_size) / sfreq, windows[0].T * 1e6)
    plt.xlabel("Time (seconds)")
    plt.ylabel("EEG (microvolts)")
    plt.title(f"Filtered EEG: {lf}-{hf} Hz")
    plt.tight_layout()
    plt.savefig(os.path.join(output, f"recording_{recording_id}_preview.png"))
    plt.close()

    quality.append({"recording_id": recording_id, "path": path, "subject_id": row["subject_id"],
                    "group_id": row["group_id"],
                    "channels": "|".join(channels), "sampling_hz": sfreq,
                    "duration_s": raw.n_times / sfreq, "windows": len(windows),
                    "flagged_windows": flagged, "discarded_tail_samples": data.shape[1] - start})


# ── STEP 1d: sab kuch CSV me save + chhota summary ──
fdf = pd.concat(table, ignore_index=True)
x = fdf[col].to_numpy()
if not np.isfinite(x).all():
    raise ValueError("Extracted features contain NaN or Inf.")
fdf.to_csv(os.path.join(output, "features.csv"), index=False)
pd.DataFrame(quality).to_csv(os.path.join(output, "recording_checks.csv"), index=False)
pd.DataFrame(window_checks).to_csv(os.path.join(output, "window_checks.csv"), index=False)
counts = fdf.groupby("subject_id").agg(recordings=("recording_id", "nunique"),
                                       groups=("group_id", "nunique"), windows=("window_id", "size"))
counts.to_csv(os.path.join(output, "subject_counts.csv"))

print("Missing feature values:", fdf[col].isna().sum().sum())
total_flagged = 0
for q in quality:
    total_flagged += q["flagged_windows"]
print("Potential artifact windows:", total_flagged)
print(counts)
print("Saved validation results to:", os.path.abspath(output))
print("At least 3 independent groups per person are required for the next step.")


# ══════════════════════════════════════════════════════════════
# STEP 2: Train and evaluate the SVM
# Same process me chalta hai; channels/sfreq/time memory me hi hain
# (JSON file read/write ki zarurat nahi).
# ══════════════════════════════════════════════════════════════
fdf = pd.read_csv(os.path.join(output, "features.csv"), dtype={"subject_id": str, "group_id": str})

# feature columns wapas bana lo: mean|std|var, channel order same
col = []
for stat_name in ["mean", "std", "var"]:
    for ch in channels:
        col.append(f"{stat_name}_{ch}")

if not np.isfinite(fdf[col].to_numpy()).all() or fdf.isna().any().any():
    raise ValueError("Missing or invalid values: review Step 1.")
if fdf["subject_id"].nunique() < 2:
    raise SystemExit("Need at least two people for genuine-versus-impostor evaluation.")

counts = fdf.groupby("subject_id")["group_id"].nunique()
if (counts < 3).any():
    message = ("SVM not fitted: need at least 3 independent recordings/session groups per person. Found: "
               + str(counts.to_dict()))
    with open(os.path.join(output, "baseline_status.txt"), "w", encoding="utf-8") as status_file:
        status_file.write(message)
    raise SystemExit(message)

if total_flagged > 0:
    reviewed = input("Review recording previews/window_checks.csv. Have the artifact flags been investigated? Type yes to continue: ").strip().lower()
    if reviewed != "yes":
        raise SystemExit("Review artifacts before fitting the baseline. No windows were automatically removed.")

# ── split: POORE recording/session group ko baanto, ek-ek window ko nahi ──
fdf["split"] = ""
for sid, group in fdf.groupby("subject_id", sort=True):
    groups = sorted(group["group_id"].unique())
    # pehle 20% test; phir baaki me se 20% validation
    train_groups, test_groups = train_test_split(groups, test_size=0.2, random_state=42)
    train_groups, val_groups = train_test_split(train_groups, test_size=0.2, random_state=42)
    train_rows = group.index[group["group_id"].isin(train_groups)]
    val_rows = group.index[group["group_id"].isin(val_groups)]
    test_rows = group.index[group["group_id"].isin(test_groups)]
    fdf.loc[train_rows, "split"] = "train"
    fdf.loc[val_rows, "split"] = "validation"
    fdf.loc[test_rows, "split"] = "test"

# shared session group (2 logon ki recording ek group me) ek hi side pe rahe
if (fdf.groupby("group_id")["split"].nunique() > 1).any():
    raise SystemExit("A shared session group crossed splits. Use a coordinated session protocol before evaluation.")
assert (fdf.groupby("recording_id")["split"].nunique() == 1).all()
fdf[["recording_id", "subject_id", "group_id", "split"]].drop_duplicates().to_csv(
    os.path.join(output, "recording_split.csv"), index=False)

# scale (scaler sirf training windows se fit)
train = fdf["split"] == "train"
validation = fdf["split"] == "validation"
test = fdf["split"] == "test"
s = StandardScaler()
x_train = s.fit_transform(fdf.loc[train, col])
x_val = s.transform(fdf.loc[validation, col])
x_test = s.transform(fdf.loc[test, col])
y_train = fdf.loc[train, "subject_id"].to_numpy()

# ── har enrolled person ke liye ek SVM (asli=1, baaki enrolled=0) ──
results = []
all_scores = []
plt.figure(figsize=(6, 5))
for sid in sorted(fdf["subject_id"].unique()):
    svm_y_train = (y_train == sid).astype(int)
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
    svm.fit(x_train, svm_y_train)

    # window ke scores ka average -> ek recording ka ek score
    val_table = fdf.loc[validation, ["recording_id", "subject_id"]].copy()
    val_table["score"] = svm.decision_function(x_val)
    val_table = val_table.groupby(["recording_id", "subject_id"], as_index=False)["score"].mean()
    val_y = (val_table["subject_id"] == sid).astype(int).to_numpy()

    # validation pe best threshold (jahan FAR aur FRR sabse paas hain)
    far, true_acceptance, thresholds = roc_curve(val_y, val_table["score"], drop_intermediate=False)
    frr = 1 - true_acceptance
    finite = np.isfinite(thresholds)          # infinite threshold ko chhodo
    valid_thresholds = thresholds[finite]
    difference = np.abs(far[finite] - frr[finite])
    chosen = np.argmin(difference)
    threshold = float(valid_thresholds[chosen])

    # threshold freeze, phir test pe lagao
    test_table = fdf.loc[test, ["recording_id", "subject_id"]].copy()
    test_table["score"] = svm.decision_function(x_test)
    test_table = test_table.groupby(["recording_id", "subject_id"], as_index=False)["score"].mean()
    test_y = (test_table["subject_id"] == sid).astype(int).to_numpy()
    accepted = (test_table["score"].to_numpy() >= threshold).astype(int)
    test_far = float(accepted[test_y == 0].mean())        # impostor galti se accept
    test_frr = float((1 - accepted[test_y == 1]).mean())  # asli banda reject

    # approx EER (sirf reporting; test threshold ko tune nahi karte)
    test_roc_far, test_tpr, test_thresholds = roc_curve(test_y, test_table["score"], drop_intermediate=False)
    test_roc_frr = 1 - test_tpr
    k = np.argmin(np.abs(test_roc_far - test_roc_frr))
    eer_approx = float((test_roc_far[k] + test_roc_frr[k]) / 2)

    result = {}
    result["claimed_subject"] = sid
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
        scores["claimed_subject"] = sid
        scores["genuine"] = (scores["subject_id"] == sid).astype(int)
        scores["threshold"] = threshold
        scores["accepted"] = (scores["score"] >= threshold).astype(int)
        scores["split"] = split_name
        all_scores.append(scores)
    plt.plot(test_roc_far, test_tpr, label=sid)

metrics = pd.DataFrame(results)
metrics.to_csv(os.path.join(output, "svm_metrics.csv"), index=False)
pd.concat(all_scores, ignore_index=True).to_csv(os.path.join(output, "svm_recording_scores.csv"), index=False)
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False acceptance rate")
plt.ylabel("True acceptance rate")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output, "svm_test_roc.png"))
plt.close()

print(metrics.to_string(index=False))
print("Macro means:", metrics[["FAR", "FRR", "EER_approx", "ROC_AUC", "accuracy"]].mean().to_dict())
print("Rates are fractions (0.05 means 5%). Interpret alongside the recording counts.")


# ══════════════════════════════════════════════════════════════
# STEP 3: Look at the CNN architecture (optional, NOT trained yet)
# ══════════════════════════════════════════════════════════════
show_cnn = input("Build the CNN candidate too? Type yes or press Enter to finish: ").strip().lower()
if show_cnn == "yes":
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D
    from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
    from tensorflow.keras.optimizers import Adam

    # win_size / n_channels memory wali values se hi bana lo
    win_size = int(sfreq * time)
    n_channels = len(channels)

    # CNN ek hi list me (aapke TensorFlow script jaisa style)
    nn = Sequential([
        Conv1D(16, kernel_size=7, padding="same", activation="relu", input_shape=(win_size, n_channels)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(32, kernel_size=7, padding="same", activation="relu"),
        BatchNormalization(),
        GlobalAveragePooling1D(),
        Dense(16, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])
    nn.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    nn.summary()

    # Next experiment, after approving the SVM protocol:
    # 1. Use the SAME recording_split.csv, not a new random window split.
    # 2. Load recording_N_windows.npy: shape (windows, channels, time).
    # 3. Transpose to (windows, time, channels) for Conv1D.
    # 4. Fit per-channel mean/std on TRAINING samples only.
    # 5. Train genuine=1 versus other enrolled people=0, as in the SVM.
    # 6. Use validation for EarlyStopping and the authentication threshold.
    # 7. Average window scores per recording and use the SAME test attempts.
    # 8. Report FAR, FRR, approximate EER and AUC before discussing fusion.
    # This is an architecture candidate, not a trained or evaluated CNN.
