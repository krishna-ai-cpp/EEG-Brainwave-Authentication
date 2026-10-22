# EEG Lab — Brainwave Authentication Research

A first-year B.Tech Artificial Intelligence project by Krishna at SRM University–AP: explore EEG preprocessing, quality checks, statistical features, and a recording-aware SVM authentication protocol.

## Browser interface

The separately authorized `web_app.py` and `index.html` add a browser interface around the **unchanged** `hello.py`. No additional Python libraries are required. [Open the live workspace](https://eeg-brainwave-lab.onrender.com/).

The workspace starts empty: upload your own authorized, deidentified EDF recordings (1–12 files, up to 8 MB each and 20 MB total). Enter pseudonymous subject IDs and genuine session/group IDs. Recordings from the same session must share a group; separate task runs do not automatically count as independent sessions. No recordings are preloaded or downloaded by the application.

Files are transmitted to the Render server for processing. Temporary uploads and analysis outputs are removed after each run. Results are held in memory for up to 20 minutes and the most recent ten runs; requests cannot list other runs. Use the local workflow for private biometric recordings. The free service accepts one analysis at a time and may need time to wake up.

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

Synthetic integration checks verify that both SVM evaluation and CNN architecture construction can be invoked through the adapter. They do not establish real-world biometric performance. Public EEG examples used in the video demonstrate preprocessing and the insufficient-session guard. No real-world accuracy or fusion improvement is claimed. Not for medical diagnosis or real access-control decisions.

For the full local workflow run `python hello.py` with your own authorized data and a new results directory. Keep private recordings and derived biometric data out of Git.

## Public sample attribution

The demonstration uses manually selected public recordings from the [PhysioNet EEG Motor Movement/Imagery Dataset, version 1.0.0](https://physionet.org/content/eegmmidb/1.0.0/). Recordings are not committed to this repository or preloaded in the app. Each run report includes uploaded file SHA-256 hashes.

Schalk, G. (2009). EEG Motor Movement/Imagery Dataset (version 1.0.0). PhysioNet. https://doi.org/10.13026/C28G6P. Data license: [Open Data Commons Attribution License v1.0](https://opendatacommons.org/licenses/by/1-0/).

Original publication: Schalk G, McFarland DJ, Hinterberger T, Birbaumer N, Wolpaw JR. BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. IEEE Transactions on Biomedical Engineering 51(6), 1034–1043, 2004.

PhysioNet reference: Goldberger AL et al. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101(23):e215–e220, 2000.

## Source preservation

`hello.py` SHA-256: `5d0b732f70042407649b43091d0005f84634edd6e52b2ee655765cc5125a4be9`.

Original Ubuntu project files, recordings, previous results, and environment were preserved. Only reviewed source and dependency files are published. `requirements-installed.txt` records the already installed Ubuntu environment; `requirements.txt` is the original dependency list. No credentials or private EEG artifacts are included.
