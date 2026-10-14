"""Live mode: drive the real macOS mouse pointer from inferred touches."""

import cv2
import Quartz

from .app import App, text
from .body import draw_body
from .calibration import Calibration
from .model import TouchModel
from .touch import TouchEngine


def accessibility_ok() -> bool:
    from ApplicationServices import AXIsProcessTrusted

    return bool(AXIsProcessTrusted())


def _post(kind, x, y):
    ev = Quartz.CGEventCreateMouseEvent(None, kind, (x, y), Quartz.kCGMouseButtonLeft)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)


def run_live(camera=None, preview=True, move_cursor=True):
    if not accessibility_ok():
        raise SystemExit(
            "Accessibility permission is needed to move the mouse.\n"
            "System Settings > Privacy & Security > Accessibility: enable your terminal app, then retry."
        )
    calib = Calibration.load()
    cells = calib.report.get("mode") == "cells"
    if cells:
        from .model import CellModel
        from .touch import CellTouchEngine

        model = CellModel.load()
        engine = CellTouchEngine(model)
    else:
        model = TouchModel.load()
        engine = TouchEngine(model, calib.canvas_size)
    app = App(camera=calib.camera_index if camera is None else camera, window="holo live", fullscreen=False)
    ox, oy = calib.canvas_offset
    paused, pressed = True, False
    print("Live mode starts PAUSED. Press P in the preview window to start/pause, Q (or Ctrl+C) to stop.")
    try:
        while True:
            obs = app.step()
            tf = engine.update(obs)
            if not paused:
                for ev in tf.events:
                    kind = ev[0]
                    p = model.centre(ev[1]) if cells and ev[1] is not None else ev[1]
                    if kind == "down":
                        _post(Quartz.kCGEventLeftMouseDown, p[0] + ox, p[1] + oy)
                        pressed = True
                    elif kind == "up" and pressed:
                        x, y = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
                        _post(Quartz.kCGEventLeftMouseUp, x, y)
                        pressed = False
                if move_cursor and not cells and tf.state != "touch" and tf.gaze is not None and not tf.events:
                    _post(Quartz.kCGEventMouseMoved, tf.gaze[0] + ox, tf.gaze[1] + oy)  # cursor follows your eyes
            if preview:
                view = draw_body(app.frame, obs)
                view = cv2.resize(view, (640, int(640 * view.shape[0] / view.shape[1])))
                label = f"cell {tf.cell}" if cells else ""
                text(view, f"{'PAUSED (P to start)' if paused else tf.state} {label}  p={tf.p_touch:.2f}  "
                           f"fps {app.fps:.0f}", (10, view.shape[0] - 12), 0.55)
                key = app.show(view)
                if key in (ord("q"), 27):
                    break
                if key == ord("p"):
                    paused = not paused
    except KeyboardInterrupt:
        pass
    finally:
        if pressed:
            x, y = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
            _post(Quartz.kCGEventLeftMouseUp, x, y)
        app.close()
