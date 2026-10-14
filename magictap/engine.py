"""Headless Magic Tap engine: camera -> gaze points, microphone -> taps.

Lighter than the button board: no body tracker and no window. Used by
`magictap serve` and `magictap mouse`.
"""

import threading
import time

from .calibration import Calibration
from .camera import Camera, find_camera
from .config import CAMERA_HEIGHT, CAMERA_NAME, CAMERA_WIDTH
from .fusion import GazeBuffer, to_global
from .gaze import GazeEstimator
from .model import CellModel
from .recenter import apply_correction, correction_hint, load_correction
from .tapcal import TapDetector


class Engine:
    """Runs in a background thread. Callbacks: on_tap(t, samples), on_ignored(t, reason), on_gaze(t, xy)."""

    def __init__(self, camera=None, on_tap=None, on_ignored=None, on_gaze=None):
        self.calib = Calibration.load()
        self.model = CellModel.load()
        self.screen = (self.calib.canvas_size[0] + self.calib.canvas_offset[0],
                       self.calib.canvas_size[1] + self.calib.canvas_offset[1])
        self.camera_index = (self.calib.camera_index if self.calib.camera_index >= 0 else find_camera(CAMERA_NAME)) \
            if camera is None else camera
        self.on_tap, self.on_ignored, self.on_gaze = on_tap, on_ignored, on_gaze
        self.gaze = GazeBuffer()
        self.correction, _ = load_correction()  # session re-centre (affine), applied to every gaze point
        hint = correction_hint()
        if hint:
            print(hint, flush=True)
        self._stop = threading.Event()
        self.ready = threading.Event()  # set once the first camera frame has been processed
        self._thread = None

    def start(self):
        print("Starting camera, microphone and models (about 10 s)...", flush=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _run(self):
        try:
            self._loop()
        except Exception:
            import traceback

            traceback.print_exc()
            print("Magic Tap engine stopped because of the error above. Try `magictap doctor`.", flush=True)

    def _loop(self):
        cam = Camera(self.camera_index, CAMERA_WIDTH, CAMERA_HEIGHT)
        est = GazeEstimator()
        taps = TapDetector()
        pending = []  # taps waiting for the end of their gaze window
        seq = 0
        try:
            while not self._stop.is_set():
                frame, t, seq = cam.read(seq)
                g = est.process(frame, t)
                xy = None
                if g is not None and not g.blink:
                    canvas_xy = self.model.gaze.predict(g.features[None])[0] + self.model.offset
                    canvas_xy = apply_correction(self.correction, canvas_xy)
                    xy = to_global(canvas_xy, self.calib.canvas_offset)
                self.gaze.add(t, xy)
                if not self.ready.is_set():
                    self.ready.set()
                    print("Magic Tap is ready: look at something and tap the screen.", flush=True)
                if xy is not None and self.on_gaze:
                    self.on_gaze(t, xy)
                for tt, ok, score, reason, _ in taps.poll():
                    if ok:
                        pending.append(tt)
                    elif self.on_ignored:
                        self.on_ignored(tt, reason)
                now = time.monotonic()
                for tt in [p for p in pending if now >= p + 0.06]:
                    pending.remove(tt)
                    if self.on_tap:
                        self.on_tap(tt, self.gaze.fixation(tt))
        finally:
            taps.close()
            est.close()
            cam.close()
