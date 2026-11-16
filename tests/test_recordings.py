import json
import io
import numpy as np
import pandas as pd
import pytest

from vibrasight.comparison_ui import synthetic_csvs
from vibrasight.recordings import compare, load_csv, report_markdown


def pair(sensor=False):
    a, b = synthetic_csvs(sensor)
    return load_csv(a, "optical", "camera.csv"), load_csv(b, "sensor" if sensor else "optical", "reference.csv")


def test_different_sample_rates_compare_frequency_without_mixing_units():
    a, b = pair(True)
    result, spectra = compare(a, b, "x_px", "ax_m_s2", .5, 10, True)
    assert a.sample_rate == pytest.approx(30)
    assert b.sample_rate == pytest.approx(200)
    assert result["absolute_difference_hz"] == pytest.approx(0, abs=.01)
    assert result["recordings"][0]["band_peak_hz"] == pytest.approx(4, abs=.1)
    assert [row["units"] for row in result["recordings"]] == ["px", "m/s²"]
    assert b.start_time_s == 2  # no synchronization is inferred from clock origins
    for _, normalized in spectra:
        assert max(normalized) == pytest.approx(1)
    report = report_markdown(result)
    assert "SYNTHETIC" in report
    assert "not synchronized" in report
    json.dumps(result, allow_nan=False)


def test_changed_operating_condition_and_shared_band():
    a, b = pair()
    result, _ = compare(a, b, "x_px", "x_px", .5, 10)
    assert result["signed_difference_hz_b_minus_a"] == pytest.approx(1, abs=.01)
    with pytest.raises(ValueError, match="45%"):
        compare(a, b, "x_px", "x_px", .5, 14)
    with pytest.raises(ValueError, match="three FFT bins"):
        compare(a, b, "x_px", "x_px", 4, 4.01)
    with pytest.raises(ValueError, match="channel"):
        compare(a, b, "nonexistent", "x_px", .5, 10)


@pytest.mark.parametrize("fault", ["duplicate", "reverse", "gap", "nan", "bad_value", "missing_column", "short", "duplicate_header", "blank_row"])
def test_invalid_acquisition_data_is_rejected(fault):
    raw, _ = synthetic_csvs()
    frame = pd.read_csv(io.BytesIO(raw))
    if fault == "duplicate":
        frame.loc[10, "time_s"] = frame.loc[9, "time_s"]
    elif fault == "reverse":
        frame = frame.iloc[::-1]
    elif fault == "gap":
        frame = frame.drop(index=10)
    elif fault == "nan":
        frame.loc[10, "x_px"] = np.nan
    elif fault == "bad_value":
        frame["x_px"] = "not-a-number"
    elif fault == "missing_column":
        frame = frame.drop(columns="time_s")
    elif fault == "short":
        frame = frame.head(20)
    content = frame.to_csv(index=False).encode()
    if fault == "duplicate_header":
        content = content.replace(b"time_s,x_px,y_px", b"time_s,x_px,x_px", 1)
    elif fault == "blank_row":
        content = content.replace(b"\n", b"\n\n", 1)
    with pytest.raises(ValueError):
        load_csv(content, "optical", "invalid.csv")


def test_small_timestamp_jitter_is_resampled_and_recorded():
    time = np.arange(300)/30
    jittered = time.copy()
    jittered[1:-1] += .0001*np.sin(np.arange(298))
    frame = pd.DataFrame({"time_s": jittered, "x_px": np.sin(2*np.pi*4*jittered), "y_px": np.zeros(300)})
    recording = load_csv(frame.to_csv(index=False).encode(), "optical", "jitter.csv")
    assert recording.resampled
    assert 0 < recording.max_jitter_fraction < .02
    assert np.diff(recording.time) == pytest.approx(np.repeat(1/30, 299))
    result, _ = compare(recording, recording, "x_px", "x_px", .5, 10)
    assert result["recordings"][0]["resampled_for_minor_jitter"]
    assert result["recordings"][0]["band_peak_hz"] == pytest.approx(4)


def test_flat_sensor_is_not_reported_as_frequency_agreement():
    a, b = pair(True)
    b.channels["ax_m_s2"] = np.full_like(b.time, 9.81)
    result, spectra = compare(a, b, "x_px", "ax_m_s2", .5, 10)
    assert result["recordings"][1]["band_peak_hz"] is None
    assert result["absolute_difference_hz"] is None
    assert np.isfinite(spectra[1][1]).all()


def test_strong_out_of_band_sensor_peak_does_not_replace_shared_band_peak():
    a, b = pair(True)
    b.channels["ax_m_s2"] += 100*np.sin(2*np.pi*40*b.time)
    result, _ = compare(a, b, "x_px", "ax_m_s2", .5, 10)
    assert result["recordings"][1]["band_peak_hz"] == pytest.approx(4, abs=.1)
    assert result["absolute_difference_hz"] == pytest.approx(0, abs=.01)
