"""Run with: streamlit run app.py."""
import tempfile
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from vibrasight.signals import analyze
from vibrasight.video import first_frame, preview_regions, track
from vibrasight.comparison_ui import show_comparison

st.set_page_config(page_title="VibraSight", page_icon="📷", layout="wide")
st.title("VibraSight")
st.caption("Camera vibration feasibility • projected displacement • evidence before diagnosis")
st.info("Research prototype. Measurements apply only to documented capture conditions. No fault classification or certified accuracy.")
mode = st.radio("Measurement source", ["Synthetic signal", "Video recording", "Compare recordings", "Camera vs sensor"], horizontal=True)


def show_results(values, fps, label, quality=None):
    axis = st.radio("Displacement axis", ["Horizontal (x)", "Vertical (y)"], horizontal=True)
    selected = values[:, 0 if axis.startswith("Horizontal") else 1]
    minimum = st.number_input("Ignore frequencies below (Hz)", min_value=0.01, max_value=float(fps / 2 - .01), value=min(.5, float(fps / 4)))
    try:
        result = analyze(selected, fps, minimum)
    except ValueError as exc:
        st.error(str(exc))
        return
    a, b, c = st.columns(3)
    a.metric("Dominant frequency", f"{result.dominant_hz:.3f} Hz" if result.dominant_hz is not None else "No measurable motion")
    b.metric("FFT bin spacing", f"{result.bin_width_hz:.3f} Hz")
    c.metric("Detrended RMS", f"{result.rms_px:.3f} px")
    times = np.arange(len(values)) / fps
    data = pd.DataFrame({"time_s": times, "x_px": values[:, 0], "y_px": values[:, 1]})
    st.subheader("Relative displacement")
    st.line_chart(data.set_index("time_s"), x_label="Time (s)", y_label="Displacement (px)")
    st.subheader("Displacement power spectral density")
    st.line_chart(pd.DataFrame({"frequency_hz": result.frequencies, "PSD_px2_per_Hz": result.psd}).set_index("frequency_hz"), x_label="Frequency (Hz)", y_label="PSD (px²/Hz)")
    if quality is not None:
        st.caption(f"Minimum retained feature fraction: {np.min(quality):.0%}. This is a tracking indicator, not an accuracy probability.")
        data["retained_feature_fraction"] = quality
    st.download_button("Download motion CSV", data.to_csv(index=False), file_name="vibrasight_motion.csv", mime="text/csv")
    report = {
        "source": label,
        "axis": axis,
        "assumed_uniform_sample_rate_hz": fps,
        "sample_count": len(values),
        "minimum_frequency_hz": minimum,
        "dominant_frequency_hz": result.dominant_hz,
        "fft_bin_spacing_hz": result.bin_width_hz,
        "detrended_rms_px": result.rms_px,
        "method": "Linear detrend; Hann-windowed periodogram; largest eligible bin",
        "limitations": ["Projected pixel displacement", "Aliasing not excluded", "No physical accuracy or fault diagnosis claim"],
    }
    st.download_button("Download measurement summary", json.dumps(report, indent=2), file_name="vibrasight_summary.json", mime="application/json")
    st.caption(f"Source: {label}. Duration between first and last sample: {times[-1]:.2f} s. Hann-windowed, linearly detrended periodogram; no zero padding. Peak is a bin estimate, not a diagnostic label.")
    st.warning("A spectral peak alone cannot identify a machinery fault. Aliasing cannot be ruled out from this spectrum. Physical displacement requires separate calibration.")


if mode == "Synthetic signal":
    st.subheader("Known-signal demonstration")
    st.write("This simulated displacement checks signal processing; it does not validate camera tracking or industrial performance.")
    frequency = st.slider("Known frequency (Hz)", 1.0, 10.0, 4.0, .25)
    amplitude = st.slider("Peak displacement (px)", .5, 8., 3., .5)
    noise = st.slider("Noise standard deviation (px)", 0., 1., .1, .05)
    time = np.arange(300) / 30
    rng = np.random.default_rng(42)
    values = np.column_stack([amplitude * np.sin(2*np.pi*frequency*time) + rng.normal(0, noise, len(time)), rng.normal(0, noise, len(time))])
    show_results(values, 30., "Synthetic displacement, 30 samples/s")
elif mode == "Video recording":
    st.subheader("Analyze a recording")
    st.write("Use a fixed camera, good lighting, a textured moving target, and a stationary reference in the same scene. Regions are specified in original-image pixels. First version processes up to 1,800 frames.")
    upload = st.file_uploader("Video", type=["mp4", "mov", "avi"], max_upload_size=100)
    if upload:
        with tempfile.TemporaryDirectory(prefix="vibrasight-") as directory:
            path = Path(directory) / ("capture" + Path(upload.name).suffix.lower())
            path.write_bytes(upload.getvalue())
            try:
                frame, reported_fps = first_frame(path)
                height, width = frame.shape[:2]
                st.image(frame, caption=f"First frame: {width} × {height}. Decoder-reported FPS: {reported_fps:.3f}.")
                st.warning("Confirm the acquisition frame rate. Slow-motion playback FPS may differ from capture FPS. This version assumes uniformly spaced frames; variable-rate recordings must be checked or converted using verified timestamps before measurement.")
                fps = st.number_input("Confirmed acquisition frame rate (frames/s)", min_value=1., max_value=2000., value=reported_fps if 1 <= reported_fps <= 2000 else 30.)
                confirmed = st.checkbox("I have verified the acquisition rate and constant frame spacing")
                regions = []
                for name, defaults in [("Target", (0, 0, max(8, width//3), max(8, height//3))), ("Stationary reference", (width//2, height//2, max(8, width//3), max(8, height//3)))]:
                    st.markdown(f"**{name} region**")
                    cols = st.columns(4)
                    regions.append(tuple(int(col.number_input(f"{name} {field}", min_value=0 if field in ("x", "y") else 8, value=value, step=1)) for col, field, value in zip(cols, ["x", "y", "width", "height"], defaults)))
                valid_regions = True
                try:
                    selection, counts = preview_regions(frame, *regions)
                    st.image(selection, caption="Selection preview: green target; orange stationary reference. x increases rightward; y increases downward.")
                    st.caption(f"Candidate features: target {counts[0]}, reference {counts[1]}. Texture availability is not a measurement-accuracy estimate.")
                except ValueError as exc:
                    valid_regions = False
                    st.warning(str(exc))
                if st.button("Track recording", disabled=not confirmed or not valid_regions):
                    with st.spinner("Tracking target and reference…"):
                        result = track(path, *regions)
                    st.session_state["tracking"] = (upload.file_id, regions, fps, result)
                saved = st.session_state.get("tracking")
                if saved and saved[:3] == (upload.file_id, regions, fps):
                    result = saved[3]
                    st.image(result.preview, caption="Green: target. Orange: stationary reference.")
                    if result.stopped_early:
                        st.warning("Tracking lost sufficient features; results use only the consecutive valid segment before loss. Recapture if this segment is too short.")
                    if result.frame_count == 1800:
                        st.warning("Processing reached the 1,800-frame limit; later frames were not analyzed.")
                    show_results(result.displacement, fps, upload.name, result.retained_fraction)
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))

elif mode == "Compare recordings":
    show_comparison()
elif mode == "Camera vs sensor":
    show_comparison(sensor=True)

with st.expander("Reference sensor and interpretation"):
    st.write("Use Camera vs sensor to import a CSV with time_s, ax_m_s2, ay_m_s2, az_m_s2. Hardware validation is still required. Compare frequency first: acceleration amplitude and pixel displacement have different units. A stationary reference compensates approximate image translation, not general camera rotation, perspective, or rolling-shutter distortion.")
