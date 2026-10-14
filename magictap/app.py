"""Shared fullscreen app shell: camera + hand tracker + canvas + drawing helpers."""

import time

import cv2
import numpy as np

from .camera import Camera, VideoSource, find_camera
from .config import CAMERA_HEIGHT, CAMERA_NAME, CAMERA_WIDTH
from .body import BodyTracker, draw_body
from .gaze import GazeEstimator, draw_gaze
from .screen import main_canvas, open_fullscreen

FONT = cv2.FONT_HERSHEY_SIMPLEX


def text(img, s, org, scale=0.7, color=(255, 255, 255), thickness=2, center=False):
    if center:
        (w, h), _ = cv2.getTextSize(s, FONT, scale, thickness)
        org = (int(org[0] - w / 2), int(org[1] + h / 2))
    org = tuple(map(int, org))
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thickness, cv2.LINE_AA)


def pt(p):
    return int(round(p[0])), int(round(p[1]))


class App:
    def __init__(self, camera=None, video=None, window="Magic Tap", fullscreen=True):
        self.window = window
        self.canvas = main_canvas()
        if video:
            self.cam = VideoSource(video)
            self.camera_index = -1
        else:
            self.camera_index = find_camera(CAMERA_NAME) if camera is None else camera
            self.cam = Camera(self.camera_index, CAMERA_WIDTH, CAMERA_HEIGHT)
        self.tracker = BodyTracker()
        self.gaze = GazeEstimator()
        self.seq = 0
        self.fps = 0.0
        self.frame = None
        self.obs = None
        self.show_inset = True
        self.show_gaze = True
        self._last = time.monotonic()
        if fullscreen:
            open_fullscreen(window)
        else:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def step(self):
        self.frame, t, self.seq = self.cam.read(self.seq)
        self.obs = self.tracker.process(self.frame, t)
        self.obs.gaze = self.gaze.process(self.frame, t)
        now = time.monotonic()
        self.fps = 0.9 * self.fps + 0.1 / max(now - self._last, 1e-3)
        self._last = now
        return self.obs

    @property
    def camera_size(self):
        return self.frame.shape[1], self.frame.shape[0]

    def inset(self, img, avoid=None, width=400):
        """Draw the annotated camera view in the corner farthest from `avoid`."""
        if not self.show_inset or self.frame is None:
            return
        view = draw_body(self.frame, self.obs)
        draw_gaze(view, self.obs.gaze)
        h = int(view.shape[0] * width / view.shape[1])
        view = cv2.resize(view, (width, h))
        H, W = img.shape[:2]
        corners = [(10, 10), (W - width - 10, 10), (10, H - h - 10), (W - width - 10, H - h - 10)]
        if avoid is not None:
            corners.sort(key=lambda c: -np.hypot(c[0] + width / 2 - avoid[0], c[1] + h / 2 - avoid[1]))
        x, y = corners[0]
        img[y:y + h, x:x + width] = view
        cv2.rectangle(img, (x, y), (x + width, y + h), (90, 90, 90), 1)

    def show(self, img) -> int:
        cv2.imshow(self.window, img)
        return cv2.waitKey(1) & 0xFF

    def wait_key(self, img) -> int:
        cv2.imshow(self.window, img)
        return cv2.waitKey(0) & 0xFF

    def close(self):
        self.cam.close()
        self.tracker.close()
        self.gaze.close()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


def _framing_checks(obs):
    """What the camera needs to see for body-based touch to work."""
    pose = obs.pose
    visible = lambda i: pose is not None and pose[i, 3] > 0.5 and 0 < pose[i, 1] < 1
    return [
        ("Body detected", pose is not None),
        ("Both shoulders in view", visible(11) and visible(12)),
        ("Elbows in view", visible(13) or visible(14)),
        ("Eyes tracked", obs.gaze is not None and not obs.gaze.blink),
    ]


def wait_until_ready(app: App, title: str, lines: list[str]) -> None:
    """Live camera view with a framing checklist; returns when the user presses Space."""
    c = app.canvas
    while True:
        obs = app.step()
        img = c.blank(18)
        view = draw_body(app.frame, obs)
        draw_gaze(view, obs.gaze)
        vh = int(c.height * 0.62)
        vw = int(view.shape[1] * vh / view.shape[0])
        x0, y0 = (c.width - vw) // 2, 90
        img[y0:y0 + vh, x0:x0 + vw] = cv2.resize(view, (vw, vh))
        text(img, title, (c.width / 2, 45), 1.0, center=True)
        y = y0 + vh + 40
        checks = _framing_checks(obs)
        status = "   ".join(f"[{'OK' if ok else '--'}] {name}" for name, ok in checks)
        all_ok = all(ok for _, ok in checks)
        text(img, status, (c.width / 2, y), 0.65, (0, 220, 0) if all_ok else (0, 180, 255), center=True)
        for n, line in enumerate(lines):
            text(img, line, (c.width / 2, y + 40 + 30 * n), 0.6, (200, 200, 200), 1, center=True)
        text(img, "Press SPACE when you're ready   (Esc to quit)",
             (c.width / 2, c.height - 40), 0.8, (0, 220, 255), center=True)
        key = app.show(img)
        if key == ord(" "):
            return
        if key == 27:
            raise KeyboardInterrupt
