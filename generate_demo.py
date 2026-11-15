"""Generate a labeled synthetic VIDEO for exercising the upload workflow.

Usage: python generate_demo.py
Output is ignored by Git. This is not a physical measurement.
"""
import cv2
import numpy as np


def main():
    writer = cv2.VideoWriter("synthetic_3hz.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 360))
    if not writer.isOpened():
        raise RuntimeError("MJPG video encoder unavailable")
    rng = np.random.default_rng(10)
    target = rng.integers(40, 240, (100, 100), dtype=np.uint8)
    reference = rng.integers(40, 240, (100, 100), dtype=np.uint8)
    try:
        for i in range(300):
            image = np.full((360, 640), 30, dtype=np.uint8)
            image[130:230, 120:220] = target
            image[130:230, 420:520] = reference
            shift = 4*np.sin(2*np.pi*3*i/30)
            image[130:230, 120:220] = cv2.warpAffine(target, np.float32([[1, 0, shift], [0, 1, 0]]), (100, 100), borderMode=cv2.BORDER_REFLECT)
            frame = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.putText(frame, "SYNTHETIC | 3 Hz | 30 fps | 4 px peak", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, .7, (220, 220, 220), 1)
            writer.write(frame)
    finally:
        writer.release()
    print("Created synthetic_3hz.avi (simulated, not physical evidence).")
    print("Target ROI: x=130 y=140 width=80 height=80")
    print("Reference ROI: x=430 y=140 width=80 height=80")


if __name__ == "__main__":
    main()
