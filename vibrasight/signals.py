"""Frequency estimates for uniformly sampled scalar measurements."""
from dataclasses import dataclass
import numpy as np
from scipy.signal import detrend, periodogram


@dataclass
class Spectrum:
    frequencies: np.ndarray
    psd: np.ndarray
    dominant_hz: float | None
    bin_width_hz: float
    rms: float

    @property
    def rms_px(self):
        """Compatibility alias for existing optical callers; rms uses input units."""
        return self.rms


def analyze(values, sample_rate: float, minimum_hz: float = 0.5) -> Spectrum:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 32 or not np.all(np.isfinite(x)):
        raise ValueError("Provide at least 32 finite, consecutive measurement samples.")
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")
    if not 0 < minimum_hz < sample_rate / 2:
        raise ValueError("Minimum frequency must be between zero and Nyquist.")
    centered = detrend(x, type="linear")
    f, p = periodogram(centered, fs=sample_rate, window="hann", scaling="density")
    eligible = (f >= minimum_hz) & (f < sample_rate / 2)
    rms = float(np.sqrt(np.mean(centered ** 2)))
    if not np.isfinite(rms) or not np.isfinite(p).all():
        raise ValueError("Measurement values exceed the numerical range; verify scaling and units.")
    peak = float(f[np.flatnonzero(eligible)[np.argmax(p[eligible])]]) if eligible.any() and rms > 1e-8 else None
    return Spectrum(f, p, peak, sample_rate / len(x), rms)
