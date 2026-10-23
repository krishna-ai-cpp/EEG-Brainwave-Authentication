# EEG Lab — Brainwave Authentication Research

A first-year B.Tech Artificial Intelligence project by Krishna at SRM University–AP: explore EEG preprocessing, quality checks, statistical features, and a recording-aware SVM authentication protocol.

## Updated UI demo

[Watch or download the demo video](https://github.com/krishna-ai-cpp/EEG-Brainwave-Authentication/blob/main/EEG-Lab-Demo-v2.mp4) · [Direct MP4](https://raw.githubusercontent.com/krishna-ai-cpp/EEG-Brainwave-Authentication/main/EEG-Lab-Demo-v2.mp4)

64 seconds, 1920×1080, 24 fps. An edited HyperFrames walkthrough of actual Render UI captures and the completed local BI2015a report: uploads, signal previews, quality flags, SVM final outputs, CNN architecture and downloads. Full decoding and representative frames were checked. The publicly accessible video matches SHA-256 `b77c9fba6e1426f7b216598cf2dbdaa01c118ba1aa45ce541111da97fdec391e`.

## Browser interface

The separately authorized `web_app.py` and `index.html` add a browser interface around the **unchanged** `hello.py`. No additional Python libraries are required. [Open the live workspace](https://eeg-brainwave-lab.onrender.com/).

The workspace starts empty: upload your own authorized, deidentified EDF, BDF or FIF recordings (1–12 files, up to 8 MB each and 20 MB total). Enter pseudonymous subject IDs and genuine session/group IDs. Recordings from the same session must share a group; separate task runs do not automatically count as independent sessions. No recordings are preloaded or downloaded by the application.

Files are transmitted to the Render server for processing. Temporary uploads are deleted when processing finishes. Generated outputs remain in a private temporary run directory for 20 minutes after completion, then are deleted. A server restart also makes run links unavailable. Run URLs contain random tokens; requests cannot list other runs. Use the local workflow for private biometric recordings. The free service accepts one analysis at a time, has a four-minute processing limit (excluding the review pause), and may need time to wake up.

The output panel appears before the upload form on narrow screens and scrolls into view when processing begins or finishes. A stopped SVM never hides completed preprocessing outputs. Every run reports whether the original script finished, stopped at a data guard, or failed.

## Complete script-to-interface mapping

| Original script input or output | Browser interface |
| --- | --- |
| Recording folder and `recordings.csv` | User uploads plus subject/group fields; the adapter writes a temporary manifest |
| New output directory | A unique temporary directory per run; previous results are never overwritten |
| Low/high cutoffs, window duration, amplitude review limit | Editable input fields, including the previously missing amplitude limit |
| Raw/filtered first-window PNGs | Signals tab with recording selector and discarded-tail counts |
| `features.csv` | Features tab, all columns, paginated rows and full CSV download |
| `recording_checks.csv`, `window_checks.csv` | Quality tab, including high-amplitude, flat-channel and annotation flags |
| `subject_counts.csv` | Subjects and Overview tabs; identities, groups, recordings and windows |
| Artifact-investigation prompt | Execution pauses after preprocessing outputs are visible; explicitly continue or stop after inspection |
| `baseline_status.txt`, other early exits | Prominent final status and retained available outputs; original console details |
| `recording_split.csv` | Splits tab with exact train/validation/test assignments |
| `svm_metrics.csv` and printed macro means | SVM tab: all metrics, thresholds, genuine/impostor counts and macro averages |
| `svm_recording_scores.csv` | SVM tab: validation/test scores, claimed/true identity, threshold and accepted/rejected decisions |
| `svm_test_roc.png` | SVM test ROC plot |
| Optional CNN prompt and model summary | Checkbox before running; CNN tab shows the original architecture summary if reached |
| `recording_N_windows.npy` and every other saved output | Files tab, individual downloads and complete ZIP download |
| Printed progress and errors | Console tab with temporary server paths redacted |

The adapter adds a run report and file/source SHA-256 fingerprints for traceability. It does not reimplement filtering, feature extraction, fitting, scoring, splitting or threshold selection. It passes inputs to the original subprocess and reads its outputs.

Browser constraints are explicit: windows are limited to 30 seconds; uploads support single-file EDF/BDF/FIF. The original Ubuntu command-line workflow additionally supports FIF.GZ, SET, VHDR, CNT and GDF, and larger datasets without these hosting limits. Native MAT/CSV recordings need a documented layout/units adapter; renaming them to EDF is not conversion.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python web_app.py
```

Open http://localhost:8080. On Render use a Python web service, build command `pip install -r requirements.txt`, start command `python web_app.py`, and health check `/health`. The adapter uses Render's `PORT` environment variable.

## What is implemented

- EEG loading and validation with MNE, band-pass filtering, and non-overlapping windows.
- Per-channel mean, standard deviation, and variance features.
- Review flags for amplitude, flat channels, and bad annotations; no automatic removal or repair.
- In the original script: recording/session group splits, training-only scaling, per-person SVMs, validation thresholds, and recording-level evaluation when data requirements are met.
- An optional CNN architecture candidate in the original script. It is **not trained or evaluated**.

## What this demo establishes

The UI exposes preprocessing plots, quality flags, and real SVM outputs when the original data requirements are met: at least two people and three independent recording/session groups per person. It preserves the original data and artifact-review guards. The optional CNN checkbox builds the original architecture after successful SVM evaluation; **the source does not train or evaluate the CNN**. No fusion model is implemented. A CNN build that exceeds the hosting runtime reports that limitation while retaining completed SVM results.

Synthetic integration checks verify that both SVM evaluation and CNN architecture construction can be invoked through the adapter. They do not establish real-world biometric performance. Real BI2015a recordings also verify preprocessing and the insufficient-session guard on Render; the complete published game-session run was performed in Ubuntu. No real-world accuracy or fusion improvement is claimed. Not for medical diagnosis or real access-control decisions.

The rebuilt adapter's local and hosted integration checks verify downloaded NPY windows against all mean/std/variance CSV values, recording acceptance decisions against frozen thresholds, output ZIP contents, both artifact-review choices, insufficient-subject/group guards, paginated tables, and rejection of unlisted paths and foreign origins. Actual browser checks verified all output tabs, real-recording plots, SVM results, the CNN architecture summary, and ZIP download. SVM results become visible while the optional CNN architecture is still building.

## BI2015 full-dataset verification

All 65 published MAT archives (9,830,228,468 bytes) passed publisher MD5 checksum verification locally in Ubuntu, outside this repository and the Render service. BI2015a completed successfully: 43 available subjects, 129 game recordings, 20,933 complete windows, 43 SVMs, and the optional CNN architecture summary. All 172 additional calibration recordings were also preprocessed separately and correctly stopped at the original independent-group requirement.

BI2015b completed preprocessing for all 44 subjects and 176 recordings: 46,856 complete windows, with 43,834 flagged. The original script then stopped with `SVM not fitted: need at least 3 independent recordings/session groups per person`; every person has one genuine session group. This expected guard preserves the published protocol. No SVM or CNN evaluation is claimed for BI2015b.

For both complete game datasets, every window's saved mean/std/variance features was numerically checked against its NPY array. BI2015a's 3,698 validation/test score rows were checked against frozen thresholds, per-subject FAR/FRR and recording counts, and group assignments were checked for split overlap. Original EEG and generated biometric arrays remain outside GitHub.

BI2015a macro test outputs: FAR 0.0110742, FRR 0.3953488, approximate EER 0.0085825, ROC AUC 0.9828350, accuracy 0.9799892. These are fractions. Each subject has only one genuine and 42 impostor test recordings: the high accuracy must be read alongside the **39.5% false-rejection rate** and class imbalance. **19,649 of 20,933 windows were flagged for high amplitude**. Quality tables and signal previews were inspected before explicitly continuing the unchanged pipeline; no windows were silently removed. This is an exploratory within-visit execution result, not validated authentication performance.

- [BI2015a (Zenodo)](https://zenodo.org/records/3266930): the study recruited 50 people, but the published dataset contains 43 subjects with usable recordings. The three game sessions are retained as documented by the [MOABB loader](https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/braininvaders.py). Additional before/after fixation and eyes-closed recordings are processed separately; they are not invented independent authentication sessions.
- [BI2015b (Zenodo)](https://zenodo.org/records/3268762): 22 pairs, 44 subjects, four runs in one session per subject. Its true grouping cannot satisfy the original script's three-group SVM requirement. A guard stop is an expected scientifically meaningful result, not a reason to relabel task runs as independent sessions.
- MAT archives are checksum-verified and converted to double-precision FIF using existing SciPy/MNE dependencies. The documented game recordings have 32 EEG channels per person at 512 Hz, with microvolts converted to volts by multiplying by `1e-6`. Pair recordings are split into the correct 32-channel subject blocks. No cropping or downsampling is used in the full local runs.
- Any BI2015a game-session evaluation is within the published visit/conditions, not evidence of cross-day biometric reliability. P300 game datasets are not automatically validated access-control datasets.

For the full local workflow run `python hello.py` with your own authorized data and a new results directory. Keep private recordings and derived biometric data out of Git.

To inspect a completed benchmark in the same UI, use `python web_app.py --run-summary /path/to/execution_summary.json --port 8082` and open the printed localhost URL. The summary specifies the original source SHA-256, output directory, settings and exit code. This mode binds only to localhost, preserves existing result files, and offers individual downloads instead of duplicating a multi-gigabyte run into a ZIP. It does not preload data in the public app.

## Public sample attribution

The updated demonstration uses public BI2015a recordings: Korczowski et al., Brain Invaders 2015a, [Zenodo record 3266930](https://zenodo.org/records/3266930), CC BY 4.0. It includes two complete calibration recordings on Render and the complete game-session evaluation locally. Recordings are not committed to this repository or preloaded in the app. Each upload run report includes file SHA-256 hashes.

The earlier prototype used manually selected public recordings from the [PhysioNet EEG Motor Movement/Imagery Dataset, version 1.0.0](https://physionet.org/content/eegmmidb/1.0.0/); its attribution is retained below.

Schalk, G. (2009). EEG Motor Movement/Imagery Dataset (version 1.0.0). PhysioNet. https://doi.org/10.13026/C28G6P. Data license: [Open Data Commons Attribution License v1.0](https://opendatacommons.org/licenses/by/1-0/).

Original publication: Schalk G, McFarland DJ, Hinterberger T, Birbaumer N, Wolpaw JR. BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. IEEE Transactions on Biomedical Engineering 51(6), 1034–1043, 2004.

PhysioNet reference: Goldberger AL et al. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101(23):e215–e220, 2000.

## Source preservation

`hello.py` SHA-256: `5d0b732f70042407649b43091d0005f84634edd6e52b2ee655765cc5125a4be9`.

Original Ubuntu project files, recordings, previous results, and environment were preserved. Only reviewed source and dependency files are published. `requirements-installed.txt` records the already installed Ubuntu environment; `requirements.txt` is the original dependency list. No credentials or private EEG artifacts are included.