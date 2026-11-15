# VibraSight

**Can a smartphone recover repeatable vibration frequency under documented recording conditions?**

A software-led feasibility prototype for maintenance spot checks, developed as an independent portfolio project.

## Run locally

Python 3.11 or 3.12 recommended.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. Start with the clearly labeled synthetic signal, then upload a real video. Processing runs locally; no video is committed or sent to an external analysis service.

To exercise the video upload workflow before collecting real footage, run `python generate_demo.py` and upload `synthetic_3hz.avi`. Use target `(130, 140, 80, 80)` and reference `(430, 140, 80, 80)` at 30 fps. Expected horizontal frequency: 3 Hz. This rendered video is synthetic evidence only.

```sh
python -m pytest -q
```

## First milestone

- Manual target and stationary-reference rectangles in original-image coordinates.
- Sparse Lucas–Kanade flow with forward/backward consistency checks.
- Median displacement of surviving original features; approximate camera translation removed using the reference.
- Tracking stops when either region retains fewer than four features or 40% of its initial features.
- Horizontal/vertical pixel motion, detrended RMS, Hann-windowed displacement PSD and dominant frequency.
- CSV export and known-signal demonstration.

## Measurement boundaries

30 fps has a theoretical Nyquist limit of 15 Hz, not a guaranteed usable bandwidth. Start with slow, clearly visible oscillation; higher-frequency motion may alias into the spectrum. Neither a peak nor a quality indicator proves the absence of aliasing.

The user must verify capture FPS and uniform frame spacing; decoder metadata alone is insufficient, especially for slow-motion or variable-frame-rate videos. Processing uses the first 1,800 frames at most. Displacement is projected image motion in pixels, not millimeters. Reference subtraction handles approximate translation only. Rotation, rolling shutter, changing perspective, exposure blur, texture loss, illumination and feature drift remain limitations. Retained-feature fraction is not calibrated confidence. The largest eligible spectral peak is descriptive, not proof of periodicity or a machinery fault.

## Evidence and next milestones

Automated checks cover noisy known-frequency signals, static input, aliasing, and a rendered video with target motion plus camera translation. These establish software behavior, **not physical accuracy**. No real recording or hardware validation has been collected yet.

Next: repeated physical recordings; second-recording comparison; timestamped ESP32-C3/MPU6050 CSV import; measured frequency agreement; a customer pilot proposal. Compare acceleration and optical **frequency** first, not raw amplitudes in incompatible units.

See [recording protocol](docs/recording-protocol.md) and [pilot hypothesis](docs/pilot-proposal.md).

## Repository workflow

Feature branches contain work in progress. `main` is reserved for verified milestones. Keep small commits, run relevant checks before merging, and keep raw private videos out of Git. Publish only data you have permission to share. Commit attribution uses the owner's GitHub noreply address.
