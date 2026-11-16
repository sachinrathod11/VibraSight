from streamlit.testing.v1 import AppTest
from pathlib import Path


def test_synthetic_demo_and_video_screen():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    assert not app.exception
    assert app.metric[0].value == "4.000 Hz"
    app.slider[0].set_value(6.0).run()
    assert not app.exception
    assert app.metric[0].value == "6.000 Hz"
    app.radio[0].set_value("Video recording").run()
    assert not app.exception


def test_comparison_screens_and_axis_selection():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    app.radio[0].set_value("Compare recordings").run()
    assert not app.exception
    assert [metric.value for metric in app.metric] == ["4.000 Hz", "5.000 Hz", "1.000 Hz"]
    app.radio[0].set_value("Camera vs sensor").run()
    assert not app.exception
    assert [metric.value for metric in app.metric] == ["4.000 Hz", "4.000 Hz", "0.000 Hz"]
    app.selectbox[1].set_value("az_m_s2").run()
    assert not app.exception
    app.radio[1].set_value("Upload CSV files").run()
    assert not app.exception
    assert not app.metric  # old demo results must not remain visible for missing uploads
