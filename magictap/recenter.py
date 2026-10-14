"""Re-centre: a ~15 s check at the start of a session that removes calibration drift.

Seating, lid angle and lighting change between sessions, which shifts and stretches
the gaze estimate. Looking at (and touching) 6 dots at the 3×2 cell centres gives
6 pairs of (where Magic Tap thought you looked, where the dot really was). An
affine correction fitted to those pairs is applied to every gaze point afterwards.
"""

import json
import random
import time

import numpy as np

from .config import DATA_DIR

CORRECTION_PATH = DATA_DIR / "recenter.json"
MAX_AGE_HOURS = 4.0
MIN_AFFINE_POINTS = 4


# ------------------------------------------------------------------ pure logic
def fit_correction(points, targets):
    """2×3 matrix M so that [x, y, 1] @ M.T ≈ target. Affine with ≥4 points, else a median shift."""
    P, T = np.asarray(points, float), np.asarray(targets, float)
    if len(P) >= MIN_AFFINE_POINTS:
        A = np.c_[P, np.ones(len(P))]
        M, *_ = np.linalg.lstsq(A, T, rcond=None)  # (3, 2)
        return M.T, "affine"
    shift = np.median(T - P, axis=0) if len(P) else np.zeros(2)
    return np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]]), "shift"


def apply_correction(M, xy):
    """Apply a 2×3 correction to one point (x, y) or an (N, 2) array."""
    if M is None:
        return xy
    a = np.asarray(xy, float)
    single = a.ndim == 1
    a = np.atleast_2d(a)
    out = a @ M[:, :2].T + M[:, 2]
    return out[0] if single else out


def cell_centres(width, height, cols=3, rows=2):
    return [((c + 0.5) * width / cols, (r + 0.5) * height / rows) for r in range(rows) for c in range(cols)]


def save_correction(M, kind, residuals, path=CORRECTION_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"matrix": np.asarray(M).tolist(), "kind": kind, "created": time.time(),
                                "residuals_pt": [round(float(r), 1) for r in residuals]}, indent=1))


def load_correction(path=CORRECTION_PATH):
    """(matrix or None, age in hours or None)."""
    if not path.exists():
        return None, None
    d = json.loads(path.read_text())
    return np.array(d["matrix"], float), (time.time() - d["created"]) / 3600


def correction_hint():
    """A one-line warning if the session correction is missing or stale, else None."""
    M, age = load_correction()
    if M is None:
        return "No re-centre yet: run `magictap recenter` (15 s) for best accuracy."
    if age > MAX_AGE_HOURS:
        return f"Last re-centre was {age:.0f} h ago: run `magictap recenter` (15 s) for best accuracy."
    return None


# ------------------------------------------------------------------ the 15-second session
def run_recenter(camera=None, seed=None):
    import cv2

    from .app import text
    from .calibration import Calibration
    from .camera import Camera, find_camera
    from .config import CAMERA_HEIGHT, CAMERA_NAME, CAMERA_WIDTH
    from .fusion import select_fixation
    from .gaze import GazeEstimator
    from .model import CellModel
    from .screen import main_canvas, open_fullscreen

    window = "Magic Tap - re-centre"
    calib = Calibration.load()
    model = CellModel.load()
    canvas = main_canvas()
    cam_index = (calib.camera_index if calib.camera_index >= 0 else find_camera(CAMERA_NAME)) if camera is None else camera
    cam = Camera(cam_index, CAMERA_WIDTH, CAMERA_HEIGHT)
    est = GazeEstimator()
    taps = None
    try:
        from .tapcal import TAP_MODEL_PATH, TapDetector

        if TAP_MODEL_PATH.exists():
            taps = TapDetector()
    except (Exception, SystemExit):
        taps = None
    open_fullscreen(window)
    dots = cell_centres(canvas.width, canvas.height)
    order = list(range(len(dots)))
    random.Random(seed).shuffle(order)
    items, seq = [], 0

    def step():
        nonlocal seq
        frame, t, seq = cam.read(seq)
        g = est.process(frame, t)
        xy = None
        if g is not None and not g.blink:
            xy = tuple(model.gaze.predict(g.features[None])[0] + model.offset)
        items.append((t, xy))
        del items[:-120]
        return t

    def edge_text(img, s):  # small, at the screen edge: anything readable mid-screen pulls the eyes
        text(img, s, (canvas.width / 2, canvas.height - 22), 0.5, (150, 150, 150), 1, center=True)

    points, targets = [], []
    try:
        while True:  # start screen: no camera preview, nothing in the middle
            step()
            img = canvas.blank(10)
            text(img, "Re-centre (15 s): look at each dot, touch it, then press SPACE. Space to start, Esc to quit.",
                 (canvas.width / 2, 40), 0.6, (200, 200, 200), 1, center=True)
            cv2.imshow(window, img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                raise KeyboardInterrupt
            if k == ord(" "):
                break
        for n, i in enumerate(order):
            x, y = dots[i]
            shown = step()
            got = None
            while got is None:
                t = step()
                img = canvas.blank(10)
                pulse = 1 + 0.15 * np.sin((t - shown) * 6)
                cv2.circle(img, (int(x), int(y)), int(34 * pulse), (255, 210, 120), 2, cv2.LINE_AA)
                cv2.circle(img, (int(x), int(y)), 9, (255, 230, 170), -1, cv2.LINE_AA)
                edge_text(img, f"re-centre {n + 1}/{len(order)}  -  look at the dot, touch it, press Space")
                cv2.imshow(window, img)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    raise KeyboardInterrupt
                if k == ord(" ") and t - shown > 0.3:
                    got = t
                if taps is not None:
                    for tt, ok, *_ in taps.poll():
                        if ok and tt - shown > 0.3:
                            got = tt
            end = time.monotonic() + 0.08
            while time.monotonic() < end:
                step()
            samples = select_fixation(items, got)
            if len(samples) >= 3:
                points.append(np.median(np.array(samples), axis=0))
                targets.append((x, y))
            flash_end = time.monotonic() + 0.3
            while time.monotonic() < flash_end:
                step()
                img = canvas.blank(10)
                cv2.circle(img, (int(x), int(y)), 34, (80, 200, 80), -1, cv2.LINE_AA)
                cv2.imshow(window, img)
                cv2.waitKey(1)
        M, kind = fit_correction(points, targets)
        corrected = apply_correction(M, np.array(points)) if points else np.zeros((0, 2))
        before = [float(np.hypot(*(np.asarray(p) - t))) for p, t in zip(points, targets)]
        after = [float(np.hypot(*(c - np.asarray(t)))) for c, t in zip(corrected, targets)]
        save_correction(M, kind, after)
        img = canvas.blank(10)
        for (tx, ty), p, c in zip(targets, points, corrected):
            cv2.circle(img, (int(tx), int(ty)), 30, (120, 120, 120), 2, cv2.LINE_AA)
            cv2.circle(img, (int(p[0]), int(p[1])), 6, (90, 90, 200), -1, cv2.LINE_AA)
            cv2.circle(img, (int(c[0]), int(c[1])), 7, (90, 220, 90), -1, cv2.LINE_AA)
            text(img, f"{np.hypot(*(c - np.array((tx, ty)))):.0f} pt", (tx + 40, ty - 30), 0.5, (170, 170, 170), 1)
        msg = (f"Re-centred ({kind}, {len(points)} dots): gaze error {np.median(before):.0f} pt -> {np.median(after):.0f} pt."
               f"  Grey = dot, red = before, green = after.  Press any key.") if points else "No gaze captured - try again."
        text(img, msg, (canvas.width / 2, canvas.height - 22), 0.55, (200, 200, 200), 1, center=True)
        print(msg.split("  Grey")[0])
        cv2.imshow(window, img)
        cv2.waitKey(0)
    except KeyboardInterrupt:
        print("Re-centre cancelled.")
    finally:
        if taps is not None:
            taps.close()
        est.close()
        cam.close()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
