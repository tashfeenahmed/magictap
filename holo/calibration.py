"""Guided, self-paced calibration.

  1. Framing  - live camera view + checklist, press Space.
  2. Eyes     - look at dots (no touching). A separate set of check dots that
                the gaze model never trains on gives an honest accuracy score.
  3. Rest     - hands away from the screen, natural postures ("not touching").
  4. Touch    - look at a dot, touch it, press Space with the other hand. Gives
                body-posture data for WHEN, and gaze-while-touching for WHERE.
  5. Train + score.

Everything is recorded to data/calibration_runs/<time>/ (full-res video plus
every frame's features) so new methods can be tested offline on the same data.
"""

import json
import random
import time
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .app import App, pt, text, wait_until_ready
from .config import CALIBRATION_PATH, DATA_DIR, HOLD_TIME, REST_TIME
from .gaze import draw_gaze
from .model import TouchModel

RUNS_DIR = DATA_DIR / "calibration_runs"
SETTLE_TIME = 0.8  # --auto: must be still this long before recording starts
GAZE_GRID = (5, 4)  # eye-calibration dots
GAZE_DOT_TIME = 1.6  # seconds each eye dot is shown
GAZE_SETTLE = 0.6  # ignore the first part while the eyes move to the dot
HEAD_DOT_TIME = 4.0  # head-movement dots stay longer


@dataclass
class Calibration:
    canvas_size: tuple
    canvas_offset: tuple
    camera_index: int
    camera_size: tuple
    created: str = ""
    report: dict = field(default_factory=dict)
    targets: list = field(default_factory=list)

    def save(self, path=CALIBRATION_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path=CALIBRATION_PATH) -> "Calibration":
        if not path.exists():
            raise SystemExit("No calibration found. Run `uv run holo calibrate` first.")
        d = json.loads(path.read_text())
        for k in ("canvas_size", "canvas_offset", "camera_size"):
            d[k] = tuple(d[k])
        return cls(**d)


def cell_index(xy, grid, canvas):
    cols, rows = grid
    c = int(np.clip(xy[0] // (canvas.width / cols), 0, cols - 1))
    r = int(np.clip(xy[1] // (canvas.height / rows), 0, rows - 1))
    return r * cols + c


def draw_grid(img, grid, highlight=None, colour=(70, 70, 70), fill=(40, 60, 70)):
    if not grid:
        return
    cols, rows = grid
    h, w = img.shape[:2]
    if highlight is not None and highlight is not False:
        r, c = divmod(int(highlight), cols)
        cv2.rectangle(img, (int(c * w / cols), int(r * h / rows)), (int((c + 1) * w / cols), int((r + 1) * h / rows)),
                      fill, -1)
    for c in range(1, cols):
        cv2.line(img, (int(c * w / cols), 0), (int(c * w / cols), h), colour, 2)
    for r in range(1, rows):
        cv2.line(img, (0, int(r * h / rows)), (w, int(r * h / rows)), colour, 2)


def jittered_cell_targets(canvas, cols, rows, rng, spread=0.5):
    """One target per cell, at a random spot within the central `spread` of the cell."""
    out = []
    cw, ch = canvas.width / cols, canvas.height / rows
    for r in range(rows):
        for c in range(cols):
            out.append(np.array([(c + 0.5 + rng.uniform(-spread, spread) / 2) * cw,
                                 (r + 0.5 + rng.uniform(-spread, spread) / 2) * ch]))
    return out


def grid_targets(w, h, cols, rows, margin=0.08):
    xs = np.linspace(w * margin, w * (1 - margin), cols)
    ys = np.linspace(h * margin, h * (1 - margin), rows)
    return [np.array([x, y]) for y in ys for x in xs]


class Abort(Exception):
    pass


class Recorder:
    """Full-resolution video plus every frame's features and labels."""

    def __init__(self, out_dir):
        self.dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.video = None
        self.rows = []

    def add(self, frame, obs, phase, dot, target):
        if self.video is None:
            h, w = frame.shape[:2]
            self.video = cv2.VideoWriter(str(self.dir / "camera.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
        self.video.write(frame)
        g = obs.gaze
        self.rows.append((obs.t, phase, dot, -1 if target is None else target[0], -1 if target is None else target[1],
                          obs.motion, obs.features, None if g is None else g.features, bool(g is not None and g.blink)))

    def save_dataset(self, **arrays):
        flat = {}
        for k, v in arrays.items():
            if isinstance(v, tuple):  # (X, Y, groups)
                flat.update({f"{k}_X": v[0], f"{k}_Y": v[1], f"{k}_groups": v[2]})
            else:
                flat[k] = v
        np.savez_compressed(self.dir / "dataset.npz", **flat)

    def close(self):
        if self.video is not None:
            self.video.release()
        if not self.rows:
            return
        nb = next((len(r[6]) for r in self.rows if r[6] is not None), 1)
        ng = next((len(r[7]) for r in self.rows if r[7] is not None), 1)
        np.savez_compressed(
            self.dir / "frames.npz",
            t=np.array([r[0] for r in self.rows]), phase=np.array([r[1] for r in self.rows]),
            dot=np.array([r[2] for r in self.rows]), target=np.array([r[3:5] for r in self.rows], float),
            motion=np.array([r[5] for r in self.rows]),
            body=np.array([r[6] if r[6] is not None else np.full(nb, np.nan) for r in self.rows], np.float32),
            gaze=np.array([r[7] if r[7] is not None else np.full(ng, np.nan) for r in self.rows], np.float32),
            blink=np.array([r[8] for r in self.rows]))


class Calibrator:
    def __init__(self, app: App, recorder: Recorder, auto: bool = False):
        self.app, self.rec = app, recorder
        self.c = app.canvas
        self.auto = auto  # also start recording automatically when the user holds still
        self.paused = False
        self.flash = None
        self.rest_motion = []
        self.still_thr = 0.05
        self.samples_per_dot = 2
        self.grid = None  # (cols, rows) to draw cell outlines during touches

    # -- frame loop helpers -------------------------------------------------

    def frame(self, phase, dot=-1, target=None):
        """Grab + process one frame and add it to the recording."""
        obs = self.app.step()
        self.rec.add(self.app.frame, obs, phase, dot, target)
        return obs

    def show(self, img, avoid=None):
        self.app.inset(img, avoid=avoid)
        if self.paused:
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (self.c.width, self.c.height), (0, 0, 0), -1)
            img = cv2.addWeighted(img, 0.35, overlay, 0.65, 0)
            text(img, "PAUSED - press P to continue", (self.c.width / 2, self.c.height / 2), 1.1, center=True)
        key = self.app.show(img)
        if key == 27:
            raise Abort
        if key in (ord("p"), ord("P")):
            self.paused = not self.paused
        if key in (ord("i"), ord("I")):
            self.app.show_inset = not self.app.show_inset
        return key

    def lines(self, img, lines, y=None, scale=0.75):
        y = self.c.height / 2 - 20 * len(lines) if y is None else y
        for n, (s, color) in enumerate(lines):
            text(img, s, (self.c.width / 2, y + 36 * n), scale if n == 0 else scale * 0.8, color, 2 if n == 0 else 1,
                 center=True)

    def screen(self, lines, key_hint="Press SPACE to continue"):
        """An instruction screen that waits for Space."""
        while True:
            self.frame("instructions")
            img = self.c.blank(18)
            self.lines(img, [(s, (255, 255, 255) if n == 0 else (210, 210, 210)) for n, s in enumerate(lines)])
            text(img, key_hint + "     (P pause, Esc quit)", (self.c.width / 2, self.c.height - 50), 0.75,
                 (0, 220, 255), center=True)
            if self.show(img) == ord(" ") and not self.paused:
                return

    def timed(self, seconds, phase, title, subtitle, on_obs=None):
        elapsed, last_t = 0.0, None
        while elapsed < seconds:
            obs = self.frame(phase)
            if last_t is not None and not self.paused:
                elapsed += obs.t - last_t
            last_t = obs.t
            if on_obs and not self.paused:
                on_obs(obs)
            img = self.c.blank(18)
            self.lines(img, [(title, (255, 255, 255)), (subtitle, (210, 210, 210)),
                             (f"{max(seconds - elapsed, 0):.0f} s", (0, 220, 255))])
            w = self.c.width // 2
            cv2.rectangle(img, (w // 2, self.c.height // 2 + 90), (int(w / 2 + w * elapsed / seconds), self.c.height // 2 + 100),
                          (0, 200, 0), -1)
            self.body_status(img)
            self.show(img)

    def body_status(self, img):
        obs = self.app.obs
        if obs.pose is None:
            s, col = "Can't see you - sit in front of the camera", (0, 0, 255)
        else:
            s, col = f"I can see you   (hands visible: {len(obs.hands)})", (0, 180, 0)
        text(img, s, (self.c.width / 2, self.c.height - 20), 0.55, col, 1, center=True)

    # -- calibration steps --------------------------------------------------

    def rest(self, rest_X, n):
        self.screen([f"Not touching ({n})",
                     "For 8 seconds, keep your hands away from the screen.",
                     "Type a bit, rest your hands, lean back, shift around - any natural posture.",
                     "This teaches holo what you look like when you're NOT touching."])
        self.timed(REST_TIME, "rest", "Don't touch the screen", "Move naturally: type, rest, lean, shift around.",
                   lambda o: (o.features is not None and rest_X.append(o.features),
                              self.rest_motion.append(o.motion)))

    def gaze_dots(self, points, title, out, group_offset, dot_time=GAZE_DOT_TIME, hint=None):
        """Show each dot for `dot_time`; collect gaze features after the eyes settle."""
        show_inset, self.app.show_inset = self.app.show_inset, False  # the camera view would draw the eyes
        try:
            for k, p in enumerate(points):
                shown = None
                while True:
                    obs = self.frame("gaze", group_offset + k, p)
                    if self.paused:
                        shown = None
                    else:
                        shown = shown or obs.t
                    age = obs.t - shown if shown else 0.0
                    g = obs.gaze
                    if shown and age >= GAZE_SETTLE and g is not None and not g.blink:
                        out.append((g.features, p, group_offset + k))
                    if shown and age >= dot_time:
                        break
                    img = self.c.blank(18)
                    r = int(8 + 26 * max(0.0, 1 - age / 0.4))  # shrink to draw the eye in
                    cv2.circle(img, pt(p), r, (0, 220, 255), -1, cv2.LINE_AA)
                    cv2.circle(img, pt(p), 3, (0, 0, 0), -1, cv2.LINE_AA)
                    text(img, f"{title}   ({k + 1}/{len(points)})", (self.c.width / 2, 30), 0.6, (170, 170, 170), 1,
                         center=True)
                    if hint:
                        text(img, hint, (self.c.width / 2, 60), 0.6, (0, 220, 255), 1, center=True)
                    if g is None:
                        text(img, "Can't see your eyes - face the screen", (self.c.width / 2, 90), 0.6, (0, 0, 255),
                             1, center=True)
                    self.show(img)
        finally:
            self.app.show_inset = show_inset

    def dots(self, targets, round_name, posture_hint):
        show_inset, self.app.show_inset = self.app.show_inset, False  # the camera view would draw the eyes
        try:
            return self._dots(targets, round_name, posture_hint)
        finally:
            self.app.show_inset = show_inset

    def _dots(self, targets, round_name, posture_hint):
        samples, hold_motion, reach_motion = [], [], []
        i = 0
        while i < len(targets):
            result = self.one_dot(targets, i, round_name, posture_hint, samples, reach_motion)
            if result == "back":
                while samples and samples[-1][0] == i - 1:
                    samples.pop()
                i = max(i - 1, 0)
                continue
            for frames, motions in result:
                samples.append((i, frames))
                hold_motion += motions
            i += 1
        return samples, hold_motion, reach_motion

    def one_dot(self, targets, i, round_name, posture_hint, samples, reach_motion):
        """Touch the dot, press Space -> record. Repeat with a shifted posture, then lift."""
        tgt = targets[i]
        stage, captures = "wait", []
        shown_t = still_since = hold_start = done_t = None
        frames, motions = [], []
        while True:
            obs = self.frame(stage, i, tgt)
            t = obs.t
            shown_t = shown_t or t
            if self.paused and stage == "hold":
                stage, frames, motions = "wait", [], []

            if stage == "wait" and not self.paused:
                reach_motion.append(obs.motion)
                if self.auto and obs.features is not None and obs.motion < self.still_thr and t - shown_t > 1.0:
                    still_since = still_since or t
                    if t - still_since >= SETTLE_TIME:
                        stage, hold_start, frames, motions = "hold", t, [], []
                else:
                    still_since = None
            elif stage == "hold" and not self.paused:
                g = obs.gaze
                frames.append((obs.features, g.features if g is not None and not g.blink else None))
                if obs.features is not None:
                    motions.append(obs.motion)
                if t - hold_start >= HOLD_TIME:
                    if sum(f[0] is not None or f[1] is not None for f in frames) >= 5:
                        captures.append((frames, motions))
                    else:
                        self.flash = ("Couldn't see you during that capture - try again", t)
                    if len(captures) >= self.samples_per_dot:
                        stage, done_t = "done", t
                    else:
                        stage, shown_t, still_since = "wait", t, None
            elif stage == "done":
                reach_motion.append(obs.motion)
                if t - done_t >= 1.0:
                    return captures

            img = self.c.blank(18)
            draw_grid(img, self.grid, highlight=self.grid and cell_index(tgt, self.grid, self.c))
            for j, _ in samples:
                cv2.circle(img, pt(targets[j]), 6, (0, 140, 0), -1)
            color = {"wait": (0, 220, 255), "hold": (0, 255, 0), "done": (0, 200, 0)}[stage]
            cv2.circle(img, pt(tgt), 40, (80, 80, 80), 2, cv2.LINE_AA)
            cv2.circle(img, pt(tgt), 10, color, -1, cv2.LINE_AA)
            for n in range(self.samples_per_dot):  # capture progress pips
                filled = n < len(captures)
                cv2.circle(img, (int(tgt[0] - 12 + 24 * n), int(tgt[1] + 58)), 6, (0, 200, 0) if filled else (90, 90, 90),
                           -1 if filled else 1, cv2.LINE_AA)
            if stage == "hold":
                cv2.ellipse(img, pt(tgt), (40, 40), -90, 0, 360 * (t - hold_start) / HOLD_TIME, (0, 255, 0), 6,
                            cv2.LINE_AA)
            # Keep text small and at the screen edge away from the dot: anything readable
            # near the middle pulls your eyes off the dot and spoils the gaze data.
            head = f"{round_name}: dot {i + 1} of {len(targets)}"
            if self.samples_per_dot > 1:
                head += f"  (capture {min(len(captures) + 1, self.samples_per_dot)} of {self.samples_per_dot})"
            msg, col = {
                "wait": ("look at the dot, touch it, press SPACE" if not captures else
                         "keep touching, shift posture a little, press SPACE", (0, 200, 255)),
                "hold": ("recording - keep touching and looking at the dot", (0, 220, 0)),
                "done": ("got it - lift your hand", (0, 200, 0)),
            }[stage]
            if self.flash and t - self.flash[1] < 2.0:
                msg, col = self.flash[0], (0, 0, 255)
            y = self.c.height - 22 if tgt[1] < self.c.height / 2 else 22
            text(img, f"{head}   -   {msg}      [B] redo  [P] pause  [Esc] quit", (self.c.width / 2, y), 0.5, col, 1,
                 center=True)
            key = self.show(img, avoid=tgt)
            if key in (ord("b"), ord("B")):
                return "back"
            if key == ord(" ") and stage == "wait" and not self.paused:
                stage, hold_start, frames, motions = "hold", t, [], []


def _message(app, lines):
    img = app.canvas.blank(18)
    for n, line in enumerate(lines):
        text(img, line, (app.canvas.width / 2, app.canvas.height / 2 + 34 * n), 0.8, center=True)
    app.show(img)
    cv2.waitKey(30)


def _arrays(samples, with_target=True):
    """[(features, target, group)] -> X, Y, groups."""
    if not samples:
        return np.zeros((0, 1)), np.zeros((0, 2)), np.zeros(0, int)
    X, Y, G = zip(*samples)
    return np.array(X), np.array(Y, float), np.array(G)


def eye_calibration(cal, c, rng):
    """Look-only dots, head-movement dots, then separate check dots. Returns (train, check) samples."""
    # 1. Gaze: training dots, then separate check dots it never trains on first.
    cal.screen(["Part 1: Eyes",
                "A dot appears in different places. Look straight at it until it moves on.",
                "Keep your head natural - no need to hold it still. Blinking is fine.",
                "Hands can stay on the keyboard."])
    train_pts = grid_targets(c.width, c.height, *GAZE_GRID)
    rng.shuffle(train_pts)
    check_pts = grid_targets(c.width, c.height, 3, 3, margin=0.2)
    check_pts = [p + np.array([c.width * 0.04, -c.height * 0.04]) for p in check_pts]  # off the training grid
    rng.shuffle(check_pts)
    gaze_train, gaze_check = [], []
    cal.gaze_dots(train_pts, "Look at the dot", gaze_train, group_offset=0)
    cal.screen(["Part 1b: Eyes + head movement",
                "Keep your eyes on the dot while you SLOWLY move your head and body:",
                "lean left and right, forward and back - like you do when reaching for the screen.",
                "This teaches holo that moving your head isn't the same as looking somewhere else."])
    head_pts = grid_targets(c.width, c.height, 3, 2, margin=0.15)
    rng.shuffle(head_pts)
    cal.gaze_dots(head_pts, "Eyes on the dot, move your head slowly", gaze_train, group_offset=500,
                  dot_time=HEAD_DOT_TIME, hint="Keep looking at the dot - lean left/right, forward/back")
    cal.screen(["Part 1c: Accuracy check",
                "A few more dots - just look at them, normal posture.",
                "Holo doesn't learn from these; they measure how accurate your eye tracking is."])
    cal.gaze_dots(check_pts, "Checking accuracy - look at the dot", gaze_check, group_offset=1000)

    return gaze_train, gaze_check


def run_calibration(camera=None, seed=None, auto=False, touch_samples=2):
    app = App(camera=camera, window="holo calibrate")
    run_dir = RUNS_DIR / time.strftime("%Y%m%d-%H%M%S")
    rec = Recorder(run_dir)
    cal = Calibrator(app, rec, auto)
    cal.samples_per_dot = touch_samples
    c = app.canvas
    rng = random.Random(seed)
    try:
        wait_until_ready(app, "Calibration - check your position", [
            "Sit the way you'll use it. Tilt the lid until everything in the checklist shows OK.",
            "Keep the lid angle and your seat position the same afterwards.",
            "About 4 minutes, at your pace. You can pause (P) any time.",
        ])
        cal.screen(["How calibration works (3 parts)",
                    "1. EYES (~1.5 min): look at dots as they appear - once still, once moving your head. No touching.",
                    "2. NOT TOUCHING (8 s): move naturally with your hands away from the screen.",
                    f"3. TOUCH (9 dots): look at the dot, touch it, press SPACE with your other hand.",
                    "Holo mainly uses where you look to know WHERE you touch,",
                    "and your arm and shoulders to know WHEN you touch."])

        gaze_train, gaze_check = eye_calibration(cal, c, rng)

        # 2. Not touching.
        rest_X = []
        cal.rest(rest_X, "part 2")
        if cal.rest_motion:
            cal.still_thr = float(np.percentile(cal.rest_motion, 40))

        # 3. Touch.
        touch_pts = grid_targets(c.width, c.height, 3, 3, margin=0.12)
        order = list(range(len(touch_pts)))
        rng.shuffle(order)
        cal.screen(["Part 3: Touch",
                    "LOOK at the yellow dot, touch it with your finger, then press SPACE with your other hand.",
                    "Keep looking at the dot while you press SPACE (not at the keyboard).",
                    f"{'Then shift your posture a little and press SPACE again. ' if touch_samples > 1 else ''}"
                    "When the dot turns green, lift your hand."])
        samples, hold_m, reach_m = cal.dots([touch_pts[k] for k in order], "Touch", "the way you normally would")
        body_touch, touch_gaze = [], []
        for j, frames in samples:
            k = order[j]
            for body_f, gaze_f in frames:
                if body_f is not None:
                    body_touch.append((body_f, touch_pts[k], k))
                if gaze_f is not None:
                    touch_gaze.append((gaze_f, touch_pts[k], 2000 + k))
        if len(gaze_train) < 50 or len(gaze_check) < 20:
            raise SystemExit("Too few eye-tracking samples (was your face visible?). Please run calibration again.")

        rec.save_dataset(gaze_train=_arrays(gaze_train), gaze_check=_arrays(gaze_check),
                         touch_gaze=_arrays(touch_gaze), body_touch=_arrays(body_touch),
                         rest_X=np.array(rest_X), hold_motion=np.array(hold_m), reach_motion=np.array(reach_m),
                         canvas=np.array([c.width, c.height]))
        _message(app, ["Training your models..."])
        model = TouchModel().fit(_arrays(gaze_train), _arrays(gaze_check), _arrays(touch_gaze), _arrays(body_touch),
                                 np.array(rest_X), hold_m, reach_m, (c.width, c.height),
                                 progress=lambda s: _message(app, ["Training your models...", s]))
        model.save()
        calib = Calibration(canvas_size=(c.width, c.height), canvas_offset=(c.offset_x, c.offset_y),
                            camera_index=app.camera_index, camera_size=app.camera_size,
                            created=time.strftime("%Y-%m-%d %H:%M:%S"), report=_jsonable(model.report),
                            targets=[t.tolist() for t in touch_pts])
        calib.save()
        (run_dir / "report.json").write_text(json.dumps(_jsonable(model.report), indent=2))
        lines = report_lines(model.report)
        print("\n".join(lines))
        print(f"Recording and data saved to {run_dir}")
        _show_summary(app, model.report, lines)
    except (Abort, KeyboardInterrupt):
        print(f"Calibration cancelled. Partial recording in {run_dir}")
    finally:
        rec.close()
        app.close()


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def _fmt(name, s):
    return (f"{name}: median {s['median']:.0f} pt, 90% within {s['p90']:.0f} pt  |  "
            f"left/mid/right {100 * s['col']:.0f}%, top/mid/bottom {100 * s['row']:.0f}%, 3x3 cell {100 * s['region']:.0f}%")


def report_lines(r):
    lines = [_fmt("Eyes, on check dots", r["gaze_validation"])]
    if "gaze_at_touch" in r:
        lines.append(_fmt("Eyes, while touching", r["gaze_at_touch"]))
    if "body" in r:
        lines.append(_fmt("Body only, while touching", r["body"]))
    if "fused_at_touch" in r:
        lines.append(_fmt("Eyes + body combined", r["fused_at_touch"]))
    lines.append(f"(Guessing the centre would be off by {r['final']['baseline']:.0f} pt; chance is 33% / 33% / 11%.)")
    if r.get("touch_vs_rest_accuracy") is not None:
        lines.append(f"Touching vs. not touching recognised {100 * r['touch_vs_rest_accuracy']:.0f}% of the time")
    lines.append(f"Verdict: {r['verdict']}")
    return lines


def _show_summary(app, r, lines):
    img = app.canvas.blank(18)
    w, h = app.canvas.width, app.canvas.height
    for key, colour in (("gaze_validation", (255, 200, 0)), ("final", (0, 220, 0))):
        s = r.get(key)
        if not s:
            continue
        for g, p in s["pred"].items():
            t = s["truth"][g]
            cv2.circle(img, pt(t), 10, colour, 2, cv2.LINE_AA)
            cv2.line(img, pt(t), pt(p), colour, 1, cv2.LINE_AA)
            cv2.circle(img, pt(p), 4, colour, -1)
    y = h / 2 - 20 * len(lines)
    for line in lines + ["Circle = real dot, line = holo's estimate (blue: eye check dots, green: touch dots). "
                         "Press any key."]:
        text(img, line, (w / 2, y), 0.55, center=True)
        y += 28
    app.wait_key(img)


# --------------------------------------------------------------------------
# Cell mode
# --------------------------------------------------------------------------

EVAL_GRIDS = [(4, 3), (3, 3), (3, 2), (2, 2)]


def run_cell_calibration(camera=None, cols=4, rows=3, rounds=3, seed=None):
    from .model import joint_features

    app = App(camera=camera, window="holo calibrate")
    run_dir = RUNS_DIR / time.strftime("%Y%m%d-%H%M%S")
    rec = Recorder(run_dir)
    cal = Calibrator(app, rec)
    cal.samples_per_dot = 1
    cal.grid = (cols, rows)
    c = app.canvas
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    try:
        wait_until_ready(app, "Calibration - check your position", [
            "Sit the way you'll use it. Tilt the lid until everything in the checklist shows OK.",
            "Keep the lid angle and your seat position the same afterwards.",
            f"About {2 + rounds * cols * rows // 9} minutes, at your pace. You can pause (P) any time.",
        ])
        cal.screen(["How calibration works (3 parts)",
                    "1. EYES (~1.5 min): look at dots as they appear - no touching.",
                    "2. NOT TOUCHING (8 s): move naturally with your hands away from the screen.",
                    f"3. TOUCH: the screen is split into {cols} x {rows} cells; touch each cell {rounds} times in rounds.",
                    "   For each: LOOK at the dot, touch it, press SPACE with your other hand, lift when it turns green.",
                    "Holo combines where you look with your arm posture to tell the cells apart."])
        gaze_train, gaze_check = eye_calibration(cal, c, rng)
        rest_X = []
        cal.rest(rest_X, "1")
        frames_X, cells, captures, round_ids, hold_m, reach_m = [], [], [], [], [], []
        cap_id = 0
        for rd in range(rounds):
            targets = jittered_cell_targets(c, cols, rows, nprng)
            order = list(range(len(targets)))
            rng.shuffle(order)
            cal.screen([f"Touch round {rd + 1} of {rounds}",
                        "LOOK at the dot, touch it, then press SPACE with your other hand.",
                        "Touch naturally - the dot moves around inside each cell between rounds.",
                        "When the dot turns green, lift your hand."])
            samples, h, rm = cal.dots([targets[k] for k in order], f"Round {rd + 1}", "the way you normally would")
            hold_m += h
            reach_m += rm
            for j, frames in samples:
                tgt = targets[order[j]]
                for body_f, gaze_f in frames:
                    if body_f is None and gaze_f is None:
                        continue
                    frames_X.append(joint_features(gaze_f, body_f))
                    cells.append(tgt)
                    captures.append(cap_id)
                    round_ids.append(rd)
                cap_id += 1
            if rd == 0 and rounds > 1:
                cal.rest(rest_X, "2")

        X, T = np.array(frames_X), np.array(cells)
        captures, round_ids = np.array(captures), np.array(round_ids)
        rec.save_dataset(gaze_train=_arrays(gaze_train), gaze_check=_arrays(gaze_check),
                         X=X, targets=T, captures=captures, rounds=round_ids, rest_X=np.array(rest_X),
                         hold_motion=np.array(hold_m), reach_motion=np.array(reach_m), canvas=np.array([c.width, c.height]),
                         grid=np.array([cols, rows]))

        model, report = train_cells(_arrays(gaze_train), _arrays(gaze_check), X, T, captures, round_ids,
                                    np.array(rest_X), hold_m, reach_m, (c.width, c.height), (cols, rows),
                                    progress=lambda s: _message(app, ["Training...", s]))
        calib = Calibration(canvas_size=(c.width, c.height), canvas_offset=(c.offset_x, c.offset_y),
                            camera_index=app.camera_index, camera_size=app.camera_size,
                            created=time.strftime("%Y-%m-%d %H:%M:%S"), report=_jsonable(report), targets=[])
        calib.save()
        (run_dir / "report.json").write_text(json.dumps(_jsonable(report), indent=2))
        lines = cell_report_lines(report)
        print("\n".join(lines))
        print(f"Recording and data saved to {run_dir}")
        _show_cell_summary(app, model, lines)
    except (Abort, KeyboardInterrupt):
        print(f"Calibration cancelled. Partial recording in {run_dir}")
    finally:
        rec.close()
        app.close()


def train_cells(gaze_train, gaze_check, X, T, captures, rounds, rest_X, hold_m, reach_m, canvas_size, grid,
                progress=lambda s: None):
    """Fit the eye mapper and cell models for several grid sizes; save and return the finest reliable one."""
    from .model import CellModel, fit_gaze_mapper

    cols, rows = grid
    mapper, mapper_name, eye_score, eye_candidates = fit_gaze_mapper(gaze_train, gaze_check, canvas_size, progress)
    grids = [g for g in EVAL_GRIDS if g[0] <= cols and g[1] <= rows]
    reports, models = {}, {}
    for gc, gr in grids:
        m = CellModel(gc, gr, canvas_size)
        m.fit(mapper, X, T, captures, rounds, rest_X, hold_m, reach_m,
              progress=lambda s, g=(gc, gr): progress(f"{g[0]}x{g[1]} grid: {s}"))
        reports[f"{gc}x{gr}"], models[(gc, gr)] = m.report, m
    reliable = [g for g in grids if (reports[f"{g[0]}x{g[1]}"]["accepted_accuracy"] or 0) >= 0.95
                and reports[f"{g[0]}x{g[1]}"]["accepted_rate"] >= 0.8]
    chosen = max(reliable, key=lambda g: g[0] * g[1]) if reliable else max(
        grids, key=lambda g: (reports[f"{g[0]}x{g[1]}"]["accuracy"], g[0] * g[1]))
    model = models[chosen]
    model.save()
    report = {"mode": "cells", "chosen_grid": list(chosen), "reliable": bool(reliable), "grids": reports,
              "eyes": {"model": mapper_name, "check_dots": eye_score, "candidates": eye_candidates}}
    return model, report


def retrain(run_dir):
    """Retrain from a saved cell-calibration recording (no new calibration needed)."""
    from pathlib import Path

    run_dir = Path(run_dir)
    d = np.load(run_dir / "dataset.npz")
    if "X" not in d:
        raise SystemExit(f"{run_dir} is not a cell-mode calibration recording.")
    t = lambda k: (d[f"{k}_X"], d[f"{k}_Y"], d[f"{k}_groups"])
    canvas = tuple(int(v) for v in d["canvas"])
    model, report = train_cells(t("gaze_train"), t("gaze_check"), d["X"], d["targets"], d["captures"], d["rounds"],
                                d["rest_X"], list(d["hold_motion"]), list(d["reach_motion"]), canvas,
                                tuple(int(v) for v in d["grid"]), progress=print)
    old = Calibration.load() if CALIBRATION_PATH.exists() else None
    calib = Calibration(canvas_size=canvas, canvas_offset=old.canvas_offset if old else (0.0, 0.0),
                        camera_index=old.camera_index if old else 0, camera_size=old.camera_size if old else (0, 0),
                        created=time.strftime("%Y-%m-%d %H:%M:%S"), report=_jsonable(report), targets=[])
    calib.save()
    (run_dir / "report.json").write_text(json.dumps(_jsonable(report), indent=2))
    print("\n".join(cell_report_lines(report)))


def cell_report_lines(report):
    e = report["eyes"]["check_dots"]
    lines = [f"Eyes on check dots: median {e['median']:.0f} pt, 3x3 cell right {100 * e['region']:.0f}% of the time",
             "Which cell you touched (each round predicted by a model that never saw it):"]
    for name, r in report["grids"].items():
        acc = r["accepted_accuracy"]
        lines.append(f"  {name} grid ({r['model']}): {100 * r['accuracy']:.0f}% right on every touch; "
                     f"with the confidence check it clicks on {100 * r['accepted_rate']:.0f}% of touches "
                     f"and those are {100 * acc:.0f}% right" if acc is not None else f"  {name}: no confident clicks")
    best = report["grids"][f"{report['chosen_grid'][0]}x{report['chosen_grid'][1]}"]
    lines.append(f"Touching vs. not touching recognised {100 * best['touch_vs_rest_accuracy']:.0f}% of the time")
    g = f"{report['chosen_grid'][0]}x{report['chosen_grid'][1]}"
    lines.append(f"Using the {g} grid (finest grid that's >=95% right on >=80% of touches)." if report["reliable"]
                 else f"No grid reached >=95% on >=80% of touches yet; using the most accurate one ({g}).")
    return lines


def _show_cell_summary(app, model, lines):
    img = app.canvas.blank(18)
    cw, ch = app.canvas.width / model.cols, app.canvas.height / model.rows
    for cell, acc in model.report["per_cell"].items():
        r, c = divmod(int(cell), model.cols)
        colour = (0, 120, 0) if acc >= 0.95 else (0, 120, 160) if acc >= 0.75 else (0, 0, 140)
        cv2.rectangle(img, (int(c * cw) + 4, int(r * ch) + 4), (int((c + 1) * cw) - 4, int((r + 1) * ch) - 4), colour, -1)
        text(img, f"{100 * acc:.0f}%", ((c + 0.5) * cw, (r + 0.5) * ch - 60), 0.8, center=True)
    y = app.canvas.height / 2 - 14 * len(lines)
    for line in lines + ["Cell colour = how often that cell was recognised. Press any key."]:
        text(img, line, (app.canvas.width / 2, y), 0.55, center=True)
        y += 28
    app.wait_key(img)
