"""Setup view: what the camera sees and which body signals are detected."""

import time

import cv2

from .app import App, text
from .body import draw_body
from .gaze import draw_gaze
from .config import MODEL_PATH, RECORDINGS_DIR


def run_preview(camera=None, video=None):
    model = None
    if MODEL_PATH.exists():
        from .model import TouchModel

        model = TouchModel.load()
    app = App(camera=camera, video=video, window="holo preview", fullscreen=False)
    writer = None
    print("Preview: R record clip, Q quit. Blue = body silhouette, yellow = skeleton, green = hands.")
    try:
        while True:
            try:
                obs = app.step()
            except EOFError:
                break
            view = draw_body(app.frame, obs)
            draw_gaze(view, obs.gaze)
            if writer is not None:
                writer.write(app.frame)
            view = cv2.resize(view, (1280, int(1280 * view.shape[0] / view.shape[1])))
            status = "no body" if obs.pose is None else f"hands {len(obs.hands)}  motion {obs.motion:.2f}"
            status += "  eyes: " + ("-" if obs.gaze is None else "blink" if obs.gaze.blink else "ok")
            if model is not None and obs.features is not None:
                _, p = model.predict_body(obs.features)
                status += f"  touch prob {p:.2f}"
            if model is not None and obs.gaze is not None and not obs.gaze.blink:
                gx, gy = model.predict_gaze(obs.gaze.features)
                status += f"  looking at ({gx:.0f}, {gy:.0f})"
            text(view, f"{status}  fps {app.fps:.0f}  {'REC' if writer else ''}", (12, view.shape[0] - 14), 0.7)
            key = app.show(view)
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                if writer is None:
                    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
                    path = RECORDINGS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.mp4"
                    h, w = app.frame.shape[:2]
                    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
                    print("Recording to", path)
                else:
                    writer.release()
                    writer = None
                    print("Recording stopped.")
    finally:
        if writer is not None:
            writer.release()
        app.close()
