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
