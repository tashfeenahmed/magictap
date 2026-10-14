"""Eye-gaze estimation from the built-in camera.

Pipeline (the data normalisation of Zhang et al. 2018, as used by ptgaze):
  1. MediaPipe face mesh (468 points + irises).
  2. Head pose: fit a 3D face template with solvePnP.
  3. Warp the face as if seen by a virtual camera 60 cm away, looking straight
     at it (224x224), which removes most head-pose variation.
  4. A ResNet-18 trained on the ETH-XGaze dataset predicts gaze pitch/yaw.
  5. Rotate back to camera coordinates and intersect the gaze ray with the
     camera plane.

Those numbers (plus eye crops) are the inputs; calibration learns the
per-user mapping to screen points, which also absorbs the unknown camera
intrinsics and screen geometry.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks.python import BaseOptions, vision
from scipy.spatial.transform import Rotation

from .config import FACE_MODEL, GAZE_WEIGHTS

ASSETS = Path(__file__).parent / "assets"
FACE_TEMPLATE = np.load(ASSETS / "face_model_mediapipe_468.npy")  # meters, from ptgaze (MIT)
REYE, LEYE, NOSE = [33, 133], [362, 263], [240, 460]
NORM_SIZE, NORM_FOCAL, NORM_DIST = 224, 960.0, 0.6
PATCH_W, PATCH_H = 24, 12
N_GEO_FEATURES = 17
N_PATCH_FEATURES = 2 * PATCH_W * PATCH_H
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class GazeObs:
    features: np.ndarray  # geometric features + eye patches
    hit: np.ndarray  # gaze ray hit on the camera plane, meters (rough, uncalibrated)
    blink: bool
    landmarks: np.ndarray  # (478, 2) pixels


def _ensure(model):
    path, url = model
    if not path.exists():
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return str(path)


def _load_gaze_net():
    import safetensors.torch
    import timm

    path, (repo, filename) = GAZE_WEIGHTS
    if not path.exists():
        import shutil

        from huggingface_hub import hf_hub_download

        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(hf_hub_download(repo, filename), path)
    net = timm.create_model("resnet18", num_classes=2)
    net.load_state_dict(safetensors.torch.load_file(str(path)))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return net.eval().to(device), device


def camera_matrix(w: int, h: int, diag_fov_deg: float = 110.0) -> np.ndarray:
    """Approximate intrinsics; per-user calibration absorbs the error."""
    f = 0.5 * np.hypot(w, h) / np.tan(np.radians(diag_fov_deg / 2))
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], float)


def _eye_patch(gray, lm, a, b):
    """Grey, contrast-normalised crop around one eye, aligned to its corners."""
    p, q = lm[a], lm[b]
    centre = (p + q) / 2
    width = np.linalg.norm(q - p) * 1.5
    angle = np.degrees(np.arctan2(q[1] - p[1], q[0] - p[0]))
    M = cv2.getRotationMatrix2D(tuple(centre), angle, PATCH_W / max(width, 1))
    M[:, 2] += (PATCH_W / 2 - centre[0], PATCH_H / 2 - centre[1])
    patch = cv2.warpAffine(gray, M, (PATCH_W, PATCH_H), flags=cv2.INTER_AREA)
    patch = cv2.equalizeHist(patch).astype(np.float32) / 255.0
    return patch.ravel()


def _eye_openness(lm):
    r = np.linalg.norm(lm[159] - lm[145]) / max(np.linalg.norm(lm[33] - lm[133]), 1)
    l = np.linalg.norm(lm[386] - lm[374]) / max(np.linalg.norm(lm[362] - lm[263]), 1)
    return (r + l) / 2


def _iris_ratios(lm):
    out = []
    for iris, a, b, top, bot in ((468, 33, 133, 159, 145), (473, 362, 263, 386, 374)):
        ax = lm[b] - lm[a]
        w = max(np.linalg.norm(ax), 1)
        out += [float((lm[iris] - lm[a]) @ ax / w ** 2), float((lm[iris][1] - (lm[top][1] + lm[bot][1]) / 2) / w)]
    return out


class GazeEstimator:
    def __init__(self):
        self.face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_ensure(FACE_MODEL)),
            running_mode=vision.RunningMode.VIDEO, num_faces=1,
        ))
        self.net, self.device = _load_gaze_net()
        self.K = None
        self._last_ms = -1

    def process(self, frame: np.ndarray, t: float) -> GazeObs | None:
        h, w = frame.shape[:2]
        if self.K is None or self.K[0, 2] != w / 2:
            self.K = camera_matrix(w, h)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ms = max(int(t * 1000), self._last_ms + 1)
        self._last_ms = ms
        res = self.face.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)
        if not res.face_landmarks:
            return None
        lm = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], np.float64)

        # Head pose from the 3D template.
        ok, rvec, tvec = cv2.solvePnP(FACE_TEMPLATE, lm[:468], self.K, np.zeros(5), np.zeros(3),
                                      np.array([0, 0, 0.5]), useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        rvec, tvec = rvec.reshape(3), tvec.reshape(3)
        head = Rotation.from_rotvec(rvec)
        model3d = FACE_TEMPLATE @ head.as_matrix().T + tvec
        centre = model3d[REYE + LEYE + NOSE].mean(axis=0)

        # Normalising rotation: look straight at the face centre, keep head x-axis.
        z = centre / np.linalg.norm(centre)
        y = np.cross(z, head.as_matrix()[:, 0])
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        R = np.vstack([x, y, z])
        S = np.diag([1, 1, NORM_DIST / np.linalg.norm(centre)])
        Kn = np.array([[NORM_FOCAL, 0, NORM_SIZE / 2], [0, NORM_FOCAL, NORM_SIZE / 2], [0, 0, 1]])
        warp = Kn @ S @ R @ np.linalg.inv(self.K)
        face = cv2.warpPerspective(rgb, warp, (NORM_SIZE, NORM_SIZE))

        tensor = torch.from_numpy(((face.astype(np.float32) / 255 - _MEAN) / _STD).transpose(2, 0, 1))[None]
        with torch.no_grad():
            pitch, yaw = self.net(tensor.to(self.device)).cpu().numpy()[0]
        g_norm = -np.array([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)])
        g = g_norm @ R  # back to camera coordinates
        hit = centre[:2] - centre[2] / g[2] * g[:2] if g[2] < -1e-3 else np.array([np.nan, np.nan])
        hit = np.nan_to_num(np.clip(hit, -1, 1))

        geo = np.r_[g, pitch, yaw, centre, rvec, hit, _iris_ratios(lm)].astype(np.float32)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        patches = np.r_[_eye_patch(gray, lm, 33, 133), _eye_patch(gray, lm, 362, 263)].astype(np.float32)
        return GazeObs(np.r_[geo, patches], hit, _eye_openness(lm) < 0.12, lm.astype(np.float32))

    def close(self):
        self.face.close()


def draw_gaze(img: np.ndarray, g: GazeObs | None) -> None:
    if g is None:
        return
    for i in (468, 473):
        cv2.circle(img, tuple(map(int, g.landmarks[i])), 3, (0, 0, 255) if g.blink else (0, 255, 0), -1)
