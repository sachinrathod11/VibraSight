import cv2
import numpy as np
import pytest
from vibrasight.signals import analyze
from vibrasight.video import track


def test_known_frequency_with_noise_and_drift():
    t = np.arange(300) / 30
    x = 3*np.sin(2*np.pi*4*t) + .4*t + np.random.default_rng(7).normal(0, .1, len(t))
    result = analyze(x, 30)
    assert abs(result.dominant_hz - 4) <= result.bin_width_hz
    assert result.rms_px == pytest.approx(3/np.sqrt(2), abs=.1)


def test_static_and_invalid_inputs():
    assert analyze(np.zeros(100), 30).dominant_hz is None
    for values, fps in [([0]*10, 30), ([float("nan")]*100, 30), ([0]*100, 0)]:
        with pytest.raises(ValueError):
            analyze(values, fps)


def test_aliasing_is_not_mistaken_for_recovery():
    t = np.arange(300)/30
    # An 18 Hz signal aliases to 12 Hz at 30 samples/s; software cannot undo this.
    assert analyze(np.sin(2*np.pi*18*t), 30).dominant_hz == pytest.approx(12)


def test_video_recovers_motion_with_camera_translation(tmp_path):
    path = tmp_path / "known.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (240, 160))
    assert writer.isOpened()
    rng = np.random.default_rng(4)
    background = rng.integers(30, 170, (160, 240), dtype=np.uint8)
    patch = rng.integers(50, 255, (40, 40), dtype=np.uint8)
    expected = []
    for i in range(150):
        displacement = 3*np.sin(2*np.pi*3*i/30)
        expected.append(displacement)
        image = background.copy()
        image[50:90, 40:80] = patch
        # Move the target patch, then translate the whole scene independently.
        shifted = cv2.warpAffine(patch, np.float32([[1, 0, displacement], [0, 1, 0]]), (40, 40), borderMode=cv2.BORDER_REFLECT)
        image[50:90, 40:80] = shifted
        image = cv2.warpAffine(image, np.float32([[1, 0, 1.5*np.sin(2*np.pi*i/75)], [0, 1, 0]]), (240, 160), borderMode=cv2.BORDER_REFLECT)
        writer.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    writer.release()
    result = track(path, (45, 55, 30, 30), (140, 40, 65, 70))
    assert not result.stopped_early
    assert result.frame_count == 150
    assert analyze(result.displacement[:, 0], 30).dominant_hz == pytest.approx(3, abs=.2)
    assert np.sqrt(np.mean((result.displacement[:, 0] - expected)**2)) < .6


def test_rejects_overlapping_regions(tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        track(tmp_path / "unused.avi", (0, 0, 20, 20), (10, 10, 20, 20))
