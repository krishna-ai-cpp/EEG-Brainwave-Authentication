# EEG Lab — Brainwave Authentication Research

A first-year B.Tech Artificial Intelligence project by Krishna at SRM University–AP: explore EEG preprocessing, quality checks, statistical features, and a recording-aware SVM authentication protocol.

## Browser interface

The separately authorized `web_app.py` and `index.html` add a browser interface around the **unchanged** `hello.py`. No additional Python libraries are required. The public interface processes three fixed, public PhysioNet recordings and exposes actual plots and validation summaries. It accepts no personal EEG uploads, arbitrary file paths, or remote URLs. Temporary sample files and analysis outputs are removed after each run. Results are held in memory for up to 20 minutes and the most recent ten runs; requests cannot list other runs.

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

The public demo is a preprocessing demonstration, **not a trained authentication product**. Its sample is insufficient for the script's minimum of three independent recording/session groups per person. Runs from one subject are conservatively assigned to one group because separate task runs do not prove independent sessions. SVM fitting is deliberately stopped by the original data guard. No accuracy, FAR, FRR, EER, AUC, CNN performance, or fusion improvement is claimed. Quality flags require investigation. Not for medical diagnosis or real access-control decisions.

For the full local workflow run `python hello.py` with your own authorized data and a new results directory. Keep private recordings and derived biometric data out of Git.

## Public sample attribution

The browser retrieves `S001R01.edf`, `S001R03.edf`, and `S002R03.edf` directly from the [PhysioNet EEG Motor Movement/Imagery Dataset, version 1.0.0](https://physionet.org/content/eegmmidb/1.0.0/). Recordings are not committed to this repository. Each run report includes download URLs and SHA-256 hashes.

Schalk, G. (2009). EEG Motor Movement/Imagery Dataset (version 1.0.0). PhysioNet. https://doi.org/10.13026/C28G6P. Data license: [Open Data Commons Attribution License v1.0](https://opendatacommons.org/licenses/by/1-0/).

Original publication: Schalk G, McFarland DJ, Hinterberger T, Birbaumer N, Wolpaw JR. BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. IEEE Transactions on Biomedical Engineering 51(6), 1034–1043, 2004.

PhysioNet reference: Goldberger AL et al. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101(23):e215–e220, 2000.

## Source preservation

`hello.py` SHA-256: `5d0b732f70042407649b43091d0005f84634edd6e52b2ee655765cc5125a4be9`.

Original Ubuntu project files, recordings, previous results, and environment were preserved. Only reviewed source and dependency files are published. `requirements-installed.txt` records the already installed Ubuntu environment; `requirements.txt` is the original dependency list. No credentials or private EEG artifacts are included.
