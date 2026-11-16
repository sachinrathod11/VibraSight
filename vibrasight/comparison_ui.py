"""Streamlit screens for exported optical recordings and sensor references."""
import json
import numpy as np
import pandas as pd
import streamlit as st

from .recordings import SCHEMAS, compare, load_csv, report_markdown


def synthetic_csvs(sensor=False):
    rng = np.random.default_rng(2026)
    time = np.arange(300) / 30
    optical = pd.DataFrame({"time_s": time, "x_px": 3*np.sin(2*np.pi*4*time) + rng.normal(0, .03, len(time)), "y_px": rng.normal(0, .01, len(time))})
    if sensor:
        t = np.arange(2000) / 200
        # Assumed simulation scale: 1 mm/px. This is NOT a camera calibration.
        acceleration = -.003*(2*np.pi*4)**2*np.sin(2*np.pi*4*t)
        other = pd.DataFrame({"time_s": t + 2, "ax_m_s2": acceleration + rng.normal(0, .02, len(t)), "ay_m_s2": rng.normal(0, .02, len(t)), "az_m_s2": 9.81 + rng.normal(0, .02, len(t))})
    else:
        other = pd.DataFrame({"time_s": time, "x_px": 3*np.sin(2*np.pi*5*time) + rng.normal(0, .03, len(time)), "y_px": rng.normal(0, .01, len(time))})
    return optical.to_csv(index=False).encode(), other.to_csv(index=False).encode()


def show_comparison(sensor=False):
    st.subheader("Camera versus MPU6050 reference" if sensor else "Compare operating conditions")
    st.write("Analyze exported optical motion CSVs. Run video tracking first and download its motion CSV; use the same process for a second recording.")
    if sensor:
        st.write("Sensor CSV columns: time_s, ax_m_s2, ay_m_s2, az_m_s2. Use device acquisition timestamps in seconds and acceleration in m/s²; convert raw counts or g before upload. Do not use host Wi-Fi packet-arrival times as sample timestamps.")
    st.caption("Optical CSV columns: time_s, x_px, y_px. At least 32 samples; maximum 10 MB and 200,000 rows per file.")
    source = st.radio("Comparison input", ["Synthetic example", "Upload CSV files"], horizontal=True)
    synthetic = source == "Synthetic example"
    right_kind = "sensor" if sensor else "optical"
    if synthetic:
        st.warning("SYNTHETIC EXAMPLE — simulated data, not physical or industrial validation.")
        a, b = synthetic_csvs(sensor)
        names = ["synthetic_camera_4hz.csv", "synthetic_sensor_4hz.csv" if sensor else "synthetic_condition_b_5hz.csv"]
        st.download_button("Download example optical CSV", a, names[0], "text/csv")
        st.download_button("Download example reference CSV" if sensor else "Download example condition B CSV", b, names[1], "text/csv")
        st.caption("Expected peaks: 4 Hz camera / 4 Hz sensor. Signals are analyzed independently, not synchronized." if sensor else "Expected peaks: 4 Hz condition A / 5 Hz condition B.")
    else:
        cols = st.columns(2)
        first = cols[0].file_uploader("A — optical motion CSV", type=["csv"], max_upload_size=10)
        second = cols[1].file_uploader("B — accelerometer CSV" if sensor else "B — optical motion CSV", type=["csv"], max_upload_size=10)
        if first is None or second is None:
            st.info("Upload both files to validate their timestamps and select an analysis band.")
            return
        a, b, names = first.getvalue(), second.getvalue(), [first.name, second.name]
    try:
        left, right = load_csv(a, "optical", names[0]), load_csv(b, right_kind, names[1])
        columns = st.columns(2)
        for col, rec in zip(columns, (left, right)):
            col.write(f"**{rec.name}**")
            col.caption(f"{len(rec.time):,} samples · {rec.sample_rate:.3f} samples/s · {rec.time[-1]:.3f} s · maximum interval deviation {rec.max_jitter_fraction:.3%}")
            if rec.resampled:
                col.info("Small timestamp jitter detected. Values were interpolated onto an evenly spaced grid; the report records this.")
        left_axis = columns[0].selectbox("A channel", SCHEMAS["optical"])
        right_axis = columns[1].selectbox("B channel", SCHEMAS[right_kind])
        ceiling = .45 * min(left.sample_rate, right.sample_rate)
        if ceiling <= .02:
            raise ValueError("The sample rate is too low for the minimum comparison band.")
        controls = st.columns(2)
        low = controls[0].number_input("Band lower edge (Hz)", min_value=.001, max_value=ceiling-.001, value=min(.5, ceiling/4), format="%.3f")
        high = controls[1].number_input("Band upper edge (Hz)", min_value=.002, max_value=ceiling, value=min(10., ceiling), format="%.3f")
        st.caption(f"Band ceiling: {ceiling:.3f} Hz. The 45%-of-slower-rate guard band is an analysis policy; it does not rule out aliasing in either acquisition.")
        if not synthetic:
            prompt = "I confirm these recordings represent the same operating condition and physically corresponding axes/locations; they need not be synchronized." if sensor else "I recorded these conditions with documented settings and selected the intended motion axes."
            if not st.checkbox(prompt):
                st.info("Confirm the recording context before interpreting a paired comparison.")
                return
        asset = st.text_input("Asset or experiment (optional)", value="Synthetic bench example" if synthetic else "")
        notes = st.text_area("Capture notes (optional)", placeholder="Operating condition, phone mode, mounting, distance, reference sensor settings, what changed…")
        result, spectra = compare(left, right, left_axis, right_axis, low, high, synthetic)
        result["capture_context"] = {"asset_or_experiment": asset or "Not supplied", "notes": notes or "Not supplied", "paired_context_confirmed_by_user": not synthetic}
        a_row, b_row = result["recordings"]
        metrics = st.columns(3)
        for col, label, row in zip(metrics[:2], ["A band peak", "B band peak"], [a_row, b_row]):
            col.metric(label, f"{row['band_peak_hz']:.3f} Hz" if row["band_peak_hz"] is not None else "Flat signal")
        delta = result["absolute_difference_hz"]
        metrics[2].metric("Absolute frequency difference", f"{delta:.3f} Hz" if delta is not None else "Unavailable")
        st.dataframe(pd.DataFrame(result["recordings"])[["slot", "name", "channel", "units", "sample_rate_hz", "duration_s", "band_peak_hz", "fft_bin_spacing_hz"]], hide_index=True)
        for row in result["recordings"]:
            if row["band_peak_at_edge"]:
                st.warning(f"{row['name']}: the peak is at the selected band's edge. Reconsider the band before interpreting it.")
        st.subheader("Spectral shape within the shared band")
        plot = []
        for label, (f, psd) in zip(["A", "B"], spectra):
            plot.append(pd.DataFrame({"frequency_hz": f, "relative_PSD": psd, "recording": label}))
        st.line_chart(pd.concat(plot), x="frequency_hz", y="relative_PSD", color="recording", x_label="Frequency (Hz)", y_label="PSD / own band maximum (unitless)")
        st.caption("Each spectrum is normalized separately. Curves compare shape, not amplitude. No frequency agreement tolerance has been validated.")
        st.subheader("Independent time histories")
        cols = st.columns(2)
        for col, rec, axis in zip(cols, (left, right), (left_axis, right_axis)):
            col.line_chart(pd.DataFrame({"elapsed_s": rec.time, axis: rec.channels[axis]}).set_index("elapsed_s"), x_label="Elapsed time (s; not synchronized)", y_label=rec.units)
        st.warning(result["interpretation"] + " Acceleration can emphasize different frequency components from displacement. Matching peaks alone do not establish physical accuracy.")
        st.download_button("Download comparison report (.md)", report_markdown(result), "vibrasight_comparison.md", "text/markdown")
        st.download_button("Download comparison results (.json)", json.dumps(result, indent=2, allow_nan=False), "vibrasight_comparison.json", "application/json")
        with st.expander("Timing policy and limitations"):
            st.write("CSV timestamps must strictly increase. Every sample interval must lie within 2% of the mean interval. Small accepted jitter is resampled linearly; larger jitter or gaps are rejected. This is a prototype policy, not a certified accuracy bound. No samples are silently removed.")
            for item in result["limitations"]:
                st.write("• " + item)
    except ValueError as exc:
        st.error(str(exc))
