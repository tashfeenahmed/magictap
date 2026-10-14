"""Body signals from the forward-facing camera.

The camera can't see the fingertip on the glass, but it does see how the
shoulders, elbows, forearms and any visible part of the hand are arranged, plus
the body's silhouette and where the arm leaves the frame. Those change
systematically with where on the screen the user is reaching, so we turn each
frame into a feature vector and let a per-user model (model.py) learn the rest.
"""

from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from .config import HAND_MODEL, MASK_H, MASK_W, POSE_MODEL

# Pose landmarks used: nose, ears, shoulders, elbows, wrists, pinky/index/thumb.
KP_IDS = [0, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
ARM_IDS = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
L_SHOULDER, R_SHOULDER = 11, 12
N_KP_FEATURES = len(KP_IDS) * 4 + len(ARM_IDS) * 2 + 2 * 5
N_MASK_FEATURES = MASK_W * MASK_H


@dataclass
class BodyObs:
    t: float
    pose: np.ndarray | None = None  # (33, 4): x, y (0..1 image), z, visibility
    hands: list = field(default_factory=list)  # (21, 2) arrays, 0..1 image coords
    mask: np.ndarray | None = None  # (MASK_H, MASK_W) person silhouette, 0..1
    motion: float = 0.0  # silhouette change per second
    features: np.ndarray | None = None
    gaze: object = None  # GazeObs, filled in by App


def _ensure(model):
    path, url = model
    if not path.exists():
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return str(path)


def features(pose: np.ndarray, hands: list, mask: np.ndarray) -> np.ndarray:
    absolute = pose[KP_IDS].ravel()
    mid = (pose[L_SHOULDER, :2] + pose[R_SHOULDER, :2]) / 2
    width = max(float(np.linalg.norm(pose[L_SHOULDER, :2] - pose[R_SHOULDER, :2])), 0.05)
    relative = ((pose[ARM_IDS, :2] - mid) / width).ravel()
    hand_feats = np.zeros(10)
    for k, h in enumerate(sorted(hands, key=lambda h: h[8, 0])[:2]):
        hand_feats[k * 5:(k + 1) * 5] = (1.0, h[8, 0], h[8, 1], h[0, 0], h[0, 1])
    return np.concatenate([absolute, relative, hand_feats, mask.ravel()]).astype(np.float32)


class BodyTracker:
    def __init__(self, use_hands: bool = True):
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_ensure(POSE_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            output_segmentation_masks=True,
            min_pose_detection_confidence=0.5, min_tracking_confidence=0.5,
        ))
        self.hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_ensure(HAND_MODEL)),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.4, min_tracking_confidence=0.4,
        )) if use_hands else None
        self._last_ms = -1
        self._prev_mask = None
        self._prev_t = None

    def process(self, frame: np.ndarray, t: float) -> BodyObs:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ms = max(int(t * 1000), self._last_ms + 1)  # must strictly increase
        self._last_ms = ms
        obs = BodyObs(t)

        res = self.pose.detect_for_video(image, ms)
        if not res.pose_landmarks:
            self._prev_mask = None
            return obs
        obs.pose = np.array([[p.x, p.y, p.z, p.visibility] for p in res.pose_landmarks[0]], np.float32)
        full = np.squeeze(res.segmentation_masks[0].numpy_view())
        obs.mask = cv2.resize(full, (MASK_W, MASK_H), interpolation=cv2.INTER_AREA)

        if self.hand is not None:
            hres = self.hand.detect_for_video(image, ms)
            obs.hands = [np.array([[p.x, p.y] for p in lms], np.float32) for lms in hres.hand_landmarks]

        if self._prev_mask is not None:
            dt = max(t - self._prev_t, 1e-3)
            obs.motion = float(np.abs(obs.mask - self._prev_mask).mean() / dt)
        self._prev_mask, self._prev_t = obs.mask, t
        obs.features = features(obs.pose, obs.hands, obs.mask)
        return obs

    def close(self):
        self.pose.close()
        if self.hand is not None:
            self.hand.close()


_POSE_BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (15, 17), (15, 19), (15, 21),
               (16, 18), (16, 20), (16, 22), (11, 23), (12, 24), (7, 0), (8, 0)]
_HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 13),
               (13, 17), (0, 17), (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16),
               (17, 18), (18, 19), (19, 20)]


def draw_body(frame: np.ndarray, obs: BodyObs) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    if obs.mask is not None:
        m = cv2.resize(obs.mask, (w, h), interpolation=cv2.INTER_LINEAR)
        tint = np.zeros_like(out)
        tint[:] = (255, 120, 0)
        out = np.where(m[..., None] > 0.5, cv2.addWeighted(out, 0.6, tint, 0.4, 0), out)
    lw = max(2, w // 400)
    if obs.pose is not None:
        p = obs.pose[:, :2] * (w, h)
        for a, b in _POSE_BONES:
            if obs.pose[a, 3] > 0.3 and obs.pose[b, 3] > 0.3:
                cv2.line(out, tuple(map(int, p[a])), tuple(map(int, p[b])), (0, 255, 255), lw, cv2.LINE_AA)
        for i in ARM_IDS:
            # dim = estimated but not visible (e.g. arm leaving the frame)
            color = (0, 0, 255) if obs.pose[i, 3] > 0.5 else (80, 80, 160)
            cv2.circle(out, tuple(map(int, p[i])), 2 * lw, color, -1)
    for hand in obs.hands:
        q = hand * (w, h)
        for a, b in _HAND_BONES:
            cv2.line(out, tuple(map(int, q[a])), tuple(map(int, q[b])), (0, 255, 0), lw, cv2.LINE_AA)
    return out
