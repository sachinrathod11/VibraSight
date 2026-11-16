"""Validated timestamped recordings and descriptive band-peak comparisons."""
from dataclasses import dataclass
import csv
import io
import json

import numpy as np
import pandas as pd

from .signals import analyze

SCHEMAS = {
    "optical": ("x_px", "y_px"),
    "sensor": ("ax_m_s2", "ay_m_s2", "az_m_s2"),
}
MAX_ROWS = 200_000
MAX_JITTER_FRACTION = 0.02


@dataclass
class Recording:
    name: str
    kind: str
    time: np.ndarray
    channels: dict
    sample_rate: float
    max_jitter_fraction: float
    resampled: bool
    start_time_s: float

    @property
    def units(self):
        return "px" if self.kind == "optical" else "m/s²"


def load_csv(content: bytes, kind: str, name: str) -> Recording:
    """Reject gaps; interpolate only small timing jitter onto an equal grid.

    The 2% maximum interval deviation is a prototype input policy, not a
    validated instrument specification or anti-aliasing guarantee.
    """
    if kind not in SCHEMAS:
        raise ValueError("Recording type must be optical or sensor.")
    if len(content) > 10 * 1024 * 1024:
        raise ValueError("CSV must be at most 10 MB.")
    try:
        text = content.decode("utf-8-sig")
        header = next(row for row in csv.reader(io.StringIO(text)) if any(v.strip() for v in row))
        if len(set(header)) != len(header):
            raise ValueError("CSV column names must be unique.")
        frame = pd.read_csv(io.StringIO(text), nrows=MAX_ROWS + 1, skip_blank_lines=False)
    except (UnicodeError, StopIteration, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError("Cannot read CSV. Use UTF-8, comma-separated values with a header.") from exc
    required = ["time_s", *SCHEMAS[kind]]
    if not set(required).issubset(frame.columns):
        raise ValueError("Required columns: " + ", ".join(required))
    if not 32 <= len(frame) <= MAX_ROWS:
        raise ValueError(f"CSV must contain 32–{MAX_ROWS:,} samples.")
    try:
        data = frame[required].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Time and measurement columns must contain numbers.") from exc
    if not np.isfinite(data).all():
        raise ValueError("Time and measurement columns must be finite; missing samples are not dropped.")
    time = data[:, 0] - data[0, 0]
    intervals = np.diff(time)
    if np.any(intervals <= 0):
        raise ValueError("Timestamps must strictly increase; duplicates, resets and sorting are not allowed.")
    dt = time[-1] / (len(time) - 1)
    if not np.isfinite(dt) or not np.isfinite(1 / dt):
        raise ValueError("Timestamp range is not numerically usable; verify seconds and clock values.")
    jitter = float(np.max(np.abs(intervals - dt)) / dt)
    if jitter > MAX_JITTER_FRACTION + 1e-9:
        raise ValueError(f"Timing is irregular (maximum interval deviation {jitter:.1%}). Limit: 2%. Check dropped samples, timestamp units or clock resets; no FFT was computed.")
    uniform = np.linspace(0, time[-1], len(time))
    resampled = jitter > 1e-6
    channels = {
        column: np.interp(uniform, time, data[:, i + 1]) if resampled else data[:, i + 1].copy()
        for i, column in enumerate(SCHEMAS[kind])
    }
    return Recording(name, kind, uniform, channels, 1 / dt, jitter, resampled, float(data[0, 0]))


def compare(left: Recording, right: Recording, left_axis: str, right_axis: str,
            low_hz: float, high_hz: float, synthetic: bool = False):
    if left_axis not in left.channels or right_axis not in right.channels:
        raise ValueError("Select a channel present in each recording.")
    ceiling = 0.45 * min(left.sample_rate, right.sample_rate)
    if not np.isfinite([low_hz, high_hz]).all() or not 0 < low_hz < high_hz <= ceiling:
        raise ValueError(f"Select a positive band with lower < upper ≤ {ceiling:.3f} Hz (45% of the slower sample rate).")
    rows, spectra = [], []
    for slot, (rec, axis) in zip(("A", "B"), ((left, left_axis), (right, right_axis))):
        spectrum = analyze(rec.channels[axis], rec.sample_rate, low_hz)
        mask = (spectrum.frequencies >= low_hz) & (spectrum.frequencies <= high_hz)
        if np.count_nonzero(mask) < 3:
            raise ValueError("The comparison band must include at least three FFT bins for each recording. Widen the band or capture for longer.")
        f, psd = spectrum.frequencies[mask], spectrum.psd[mask]
        peak_index = int(np.argmax(psd))
        peak = float(f[peak_index]) if spectrum.rms > 1e-8 else None
        rows.append({
            "slot": slot, "name": rec.name, "kind": rec.kind, "channel": axis, "units": rec.units,
            "samples": len(rec.time), "duration_s": float(rec.time[-1]),
            "sample_rate_hz": rec.sample_rate, "max_interval_deviation_fraction": rec.max_jitter_fraction,
            "resampled_for_minor_jitter": rec.resampled, "original_start_time_s": rec.start_time_s,
            "band_peak_hz": peak, "fft_bin_spacing_hz": spectrum.bin_width_hz,
            "band_peak_at_edge": bool(peak is not None and peak_index in (0, len(f)-1)),
            "detrended_full_band_rms": spectrum.rms,
        })
        spectra.append((f, psd / np.max(psd) if np.max(psd) > 0 else np.zeros_like(psd)))
    a, b = [row["band_peak_hz"] for row in rows]
    result = {
        "provenance": "SYNTHETIC EXAMPLE — not physical validation" if synthetic else "USER-SUPPLIED CSV — provenance and acquisition settings require verification",
        "comparison": "operating-condition comparison" if right.kind == left.kind else "camera versus accelerometer frequency comparison",
        "band_hz": [float(low_hz), float(high_hz)], "recordings": rows,
        "signed_difference_hz_b_minus_a": float(b-a) if a is not None and b is not None else None,
        "absolute_difference_hz": float(abs(b-a)) if a is not None and b is not None else None,
        "interpretation": "Largest eligible spectral bins only. Bin spacing is not an accuracy bound. No automatic pass/fail tolerance or fault classification.",
        "limitations": [
            "Aliasing is not excluded by a shared frequency band.",
            "Signals are not synchronized; phase, coherence and waveform agreement are not assessed.",
            "Acceleration weights higher frequencies differently from displacement; dominant peaks may represent different components.",
            "Pixel amplitude depends on view geometry. No cross-recording amplitude ratio or cross-unit amplitude comparison is claimed.",
            "A peak can come from noise, camera movement or lighting. Frequency agreement alone does not validate measurement accuracy.",
        ],
    }
    return result, spectra


def report_markdown(result):
    """Portable engineer-facing text report with explicit provenance and limits."""
    def safe(value):
        return str(value).replace("|", "/").replace("\n", " ").replace("\r", " ").replace("<", "&lt;").replace(">", "&gt;")

    lines = ["# VibraSight measurement comparison", "", result["provenance"], "",
             f"Comparison: {result['comparison']}",
             f"Shared analysis band: {result['band_hz'][0]:.3f}–{result['band_hz'][1]:.3f} Hz", "",
             "| Slot | Recording | Channel | Rate (Hz) | Duration (s) | Band peak (Hz) | Bin spacing (Hz) |",
             "|---|---|---|---:|---:|---:|---:|"]
    for row in result["recordings"]:
        peak = f"{row['band_peak_hz']:.3f}" if row["band_peak_hz"] is not None else "Flat detrended signal"
        lines.append(f"| {row['slot']} | {safe(row['name'])} | {safe(row['channel'])} | {row['sample_rate_hz']:.3f} | {row['duration_s']:.3f} | {peak} | {row['fft_bin_spacing_hz']:.3f} |")
    for row in result["recordings"]:
        if row["band_peak_at_edge"]:
            lines.append("")
            lines.append(f"Caution: {safe(row['name'])} peak is at the selected band's edge; reconsider the band.")
    if "capture_context" in result:
        lines += ["", "## Capture context",
                  "Asset / experiment: " + safe(result["capture_context"]["asset_or_experiment"]),
                  "Notes: " + safe(result["capture_context"]["notes"])]
    delta = result["absolute_difference_hz"]
    lines += ["", f"Absolute peak-frequency difference: {delta:.3f} Hz" if delta is not None else "Frequency difference unavailable: a detrended signal is numerically flat.",
              "", result["interpretation"], "", "## Capture and timing"]
    for row in result["recordings"]:
        lines.append(f"- {safe(row['name'])}: {row['samples']} samples; maximum interval deviation {row['max_interval_deviation_fraction']:.3%}; minor-jitter resampling: {row['resampled_for_minor_jitter']}.")
    lines += ["", "## Limitations", *["- " + item for item in result["limitations"]], "",
              "## Next action", "Repeat capture under documented conditions, inspect tracking and static-target noise, and review reference mounting, axis alignment, sampling filters and timing before making a maintenance decision.", "",
              "## Machine-readable results", "```json", json.dumps(result, indent=2, allow_nan=False), "```", ""]
    return "\n".join(lines)
