# Copilot instructions — Image-Classification

Purpose: Give an AI code agent the minimal, precise knowledge to be productive here.

## Quick summary

- Binary mood classifier (Happy vs Sad) using a Keras/TensorFlow CNN.
- Main package: `src/` (training, inference, GUI). Key files: `src/train.py`, `src/inference.py`, `src/cli.py`, `src/gui/app.py`.
- Data lives under `mood/` with class subfolders: `mood/happy/`, `mood/sad/`.
- Model path: `models/mood.h5`. TensorBoard logs: `m_logs/`.

## How to run (concrete commands)

- Install: `python -m pip install -r requirements.txt` (use Python 3.10)
- Run tests: `pytest -q` (CI uses ubuntu-latest + Python 3.10)
- Train (CLI): `python -m src.cli train --epochs 50`
- Predict (CLI): `python -m src.cli predict path/to/image.jpg`
- GUI (local): `python -m src.gui.app`
- TensorBoard: `tensorboard --logdir=m_logs`

## Big-picture architecture (what changes affect what)

- `src/train.py` builds and compiles the Keras model, loads dataset from `mood/`, and saves `models/mood.h5`.
- `src/inference.py` provides `MoodPredictor` which loads `models/mood.h5`, preprocesses images (OpenCV BGR→RGB, resize to 256×256, normalize to [0,1]) and exposes `predict_from_array` / `predict_from_file`.
- `src/gui/app.py` wraps `MoodPredictor` into a CustomTkinter GUI and calls `predictor.predict_from_*` for display (CustomTkinter is required to run the GUI).
- `src/cli.py` is a thin surface for `train` and `predict` operations (used by CI and local automation).
- Notebooks (`notebooks/train.ipynb`) are the canonical exploratory training flow — keep scriptable versions in `src/train.py` in sync.

## Project-specific patterns & gotchas (do not break these)

- Dataset loading: `tf.keras.utils.image_dataset_from_directory` is used and subsequent splitting is performed with Dataset `.take` / `.skip` using `total = len(data)` (this splits by batches, not raw samples). Changing `batch_size` affects split boundaries.
- Image size: 256×256 is used across training and inference — keep consistent preprocessing.
- Channel handling: inference expects BGR (OpenCV), converts to RGB, and ensures 3 channels (handles grayscale/alpha).
- Decision threshold: model outputs a single sigmoid prob; code maps `prob <= 0.5 -> 'Happy'`, else `'Sad'`. Confidence = probability distance from 0.5 (implemented as either `prob` or `1-prob`).
- Class weights: training uses `{0: 1.1, 1: 0.9}` in `train.train` — keep or adjust explicitly if rebalancing.
- Model save path: `models/mood.h5` — tests and GUI expect this exact path.
- TensorBoard logs: `m_logs/` created by training callback — useful for debugging training regressions.
- Tests can skip: `tests/test_inference.py` and `tests/test_dataset.py` skip if model missing or TF incompatible — CI currently runs CPU-only tests and may skip inference if `models/mood.h5` is not present.

## Integration points & external dependencies

- TensorFlow 2.16.x, Keras 3.x (model saving/loading issues can occur across minor TF/Keras versions).
- OpenCV used for image I/O and camera capture; CustomTkinter is required to run the GUI.
- Dockerfile updated to use the module entrypoint (`python -m src.gui.app`).

## Concrete examples for agents (do these when editing or adding features)

- When changing dataset or batch sizes, unit-test the splitting behavior: create a small dataset and assert `take/skip` splits expected number of batches.
- When modifying inference preprocessing, add tests that feed a small in-memory array to `predict_from_array` and assert output keys/shape and value ranges.
- If adding an API server, reuse `MoodPredictor` (single responsibility) and keep I/O (file reading/camera capture) outside the prediction class.

## Troubleshooting notes (common blocker patterns)

- Model load failures can be TF/Keras-version dependent (tests skip on such failures) — pin TF/Keras or add migration code when necessary.
- GUI may not start if `customtkinter` is missing; the code falls back to plain Tkinter.
- Dockerfile/README mismatch: if you use Docker, either provide `scripts/gui_app.py` or change `CMD` to `python -m src.gui.app`.

---

If anything is missing or you'd like the file to prefer a different format (more/less detail), tell me which parts to expand or prune and I will iterate. ✅
