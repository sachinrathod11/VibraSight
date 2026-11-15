"""Sparse optical flow with fixed target and stationary-reference feature sets."""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class Tracking:
    displacement: np.ndarray
    retained_fraction: np.ndarray
    preview: np.ndarray
    reported_fps: float
    frame_count: int
    stopped_early: bool


def first_frame(path):
    cap = cv2.VideoCapture(str(path))
    try:
        ok, frame = cap.read()
        if not ok:
            raise ValueError("Cannot decode this video. Try MP4/H.264.")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()


def _features(gray, roi):
    x, y, w, h = map(int, roi)
    if x < 0 or y < 0 or w < 8 or h < 8 or x + w > gray.shape[1] or y + h > gray.shape[0]:
        raise ValueError("Each region must fit inside the image and be at least 8 × 8 pixels.")
    mask = np.zeros_like(gray)
    mask[y:y+h, x:x+w] = 255
    points = cv2.goodFeaturesToTrack(gray, 80, 0.01, 5, mask=mask)
    if points is None or len(points) < 4:
        raise ValueError("Not enough texture in a region. Include a high-contrast edge or marker.")
    return points


def track(path, target, reference, max_frames=1800):
    tx, ty, tw, th = target
    rx, ry, rw, rh = reference
    if max(tx, rx) < min(tx+tw, rx+rw) and max(ty, ry) < min(ty+th, ry+rh):
        raise ValueError("Target and stationary-reference regions must not overlap.")
    cap = cv2.VideoCapture(str(path))
    try:
        ok, frame = cap.read()
        if not ok:
            raise ValueError("Cannot decode video.")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        groups = [_features(gray, roi) for roi in (target, reference)]
        origins = [p.copy() for p in groups]
        initial = [len(p) for p in groups]
        preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for roi, color in ((target, (40, 220, 120)), (reference, (255, 170, 40))):
            x, y, w, h = map(int, roi)
            cv2.rectangle(preview, (x, y), (x+w, y+h), color, 2)
        positions, quality = [[0., 0.]], [1.]
        stopped = False
        while len(positions) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            next_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            shifts = []
            for i in range(2):
                new, status, _ = cv2.calcOpticalFlowPyrLK(gray, next_gray, groups[i], None)
                if new is None:
                    stopped = True
                    break
                back, back_status, _ = cv2.calcOpticalFlowPyrLK(next_gray, gray, new, None)
                if back is None:
                    stopped = True
                    break
                good = (status.ravel() == 1) & (back_status.ravel() == 1)
                good &= np.linalg.norm(back[:, 0] - groups[i][:, 0], axis=1) < 1.0
                groups[i], origins[i] = new[good], origins[i][good]
                if len(groups[i]) < max(4, int(np.ceil(initial[i] * .4))):
                    stopped = True
                    break
                shifts.append(np.median(groups[i][:, 0] - origins[i][:, 0], axis=0))
            if stopped:
                break
            positions.append(shifts[0] - shifts[1])
            quality.append(min(len(groups[i]) / initial[i] for i in range(2)))
            gray = next_gray
        return Tracking(np.asarray(positions), np.asarray(quality), preview, fps, len(positions), stopped)
    finally:
        cap.release()
