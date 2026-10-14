"""Fullscreen canvas and its mapping to global screen coordinates (points).

On notched MacBooks a fullscreen window sits below the notch, so the canvas is
sized to the safe area and offset by the top inset.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Canvas:
    width: int
    height: int
    offset_x: float  # canvas (0, 0) in global screen points
    offset_y: float

    def to_global(self, pt) -> tuple[float, float]:
        return float(pt[0]) + self.offset_x, float(pt[1]) + self.offset_y

    def blank(self, value: int = 0) -> np.ndarray:
        return np.full((self.height, self.width, 3), value, np.uint8)


def main_canvas() -> Canvas:
    import AppKit

    screen = AppKit.NSScreen.mainScreen()
    frame = screen.frame()
    top = 0.0
    if hasattr(screen, "safeAreaInsets"):
        top = float(screen.safeAreaInsets().top)
    return Canvas(int(frame.size.width), int(frame.size.height - top), 0.0, top)


def open_fullscreen(name: str) -> None:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def verify_canvas(name: str, canvas: Canvas) -> float | None:
    """Show markers, screenshot them, and return the worst placement error in points.

    Needs Screen Recording permission; returns None if the screenshot fails.
    """
    import subprocess
    import tempfile

    margin = 80
    expected = [(margin, margin), (canvas.width - margin, margin),
                (margin, canvas.height - margin), (canvas.width - margin, canvas.height - margin)]
    img = canvas.blank()
    for x, y in expected:
        img[y - 8:y + 8, x - 8:x + 8] = 255
    for _ in range(40):  # let the fullscreen animation finish
        cv2.imshow(name, img)
        cv2.waitKey(25)
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        if subprocess.run(["screencapture", "-x", "-m", f.name]).returncode != 0:
            return None
        shot = cv2.imread(f.name, cv2.IMREAD_GRAYSCALE)
    if shot is None:
        return None
    scale = shot.shape[1] / (canvas.width + 2 * canvas.offset_x)
    n, _, stats, centers = cv2.connectedComponentsWithStats((shot > 200).astype(np.uint8))
    found = [centers[i] / scale for i in range(1, n)
             if abs(stats[i, 2] / scale - 16) < 4 and abs(stats[i, 3] / scale - 16) < 4]
    worst = 0.0
    for p in expected:
        g = np.array(canvas.to_global(p))
        if not found:
            return None
        worst = max(worst, min(float(np.linalg.norm(g - f)) for f in found))
    return worst
