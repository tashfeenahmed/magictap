"""Button board: big on-screen buttons on the best grid Magic Tap has learned, with live
right/wrong feedback and a summary of what calibration found.

Guided mode (default): Magic Tap asks you to touch a button and marks the result.
Free mode (F): touch any button; press Y if Magic Tap got it right, N if not.
"""

import collections
import json
import random
import time

import cv2
import numpy as np

from .app import App, text, wait_until_ready
from .calibration import Calibration
from .config import SESSIONS_DIR
from .fusion import MIN_CONFIDENCE, select_fixation, vote
from .model import CellModel
from .model import joint_features
from .recenter import apply_correction, correction_hint, load_correction
from .tapcal import TAP_MODEL_PATH, TapDetector
from .touch import CellTouchEngine

TAP_WINDOW = (-0.3, 0.05)  # fallback window around a tap when there's no clear fixation
MIN_TAP_CONFIDENCE = MIN_CONFIDENCE  # below this, say "not sure" instead of risking a wrong button

LABELS = "ABCDEFGHIJKL"
BAR = 0  # buttons must match the calibrated cells exactly, so text is drawn over them


def _summary_lines(calib, model):
    r = calib.report
    g = f"{model.cols}x{model.rows}"
    gr = r["grids"][g]
    eyes = r["eyes"]["check_dots"]["median"]
    return [f"What Magic Tap learned:  eyes accurate to ~{eyes:.0f} pt  |  touch vs. no touch {100 * gr['touch_vs_rest_accuracy']:.0f}%"
            f"  |  {g} buttons: {100 * gr['accuracy']:.0f}% right on test touches ({gr['model']})"]


def run_buttons(camera=None, seed=None):
    calib = Calibration.load()
    if calib.report.get("mode") != "cells":
        raise SystemExit("Buttons need a cell calibration: run `magictap calibrate` (or `magictap retrain`).")
    model = CellModel.load()
    correction, _ = load_correction()  # session re-centre, applied to every gaze point
    hint = correction_hint()
    app = App(camera=calib.camera_index if camera is None else camera, window="Magic Tap - buttons")
    app.show_inset = False  # a camera view on screen would pull your eyes off the buttons
    c = app.canvas
    use_taps = TAP_MODEL_PATH.exists()
    taps = TapDetector() if use_taps else None
    engine = None if use_taps else CellTouchEngine(model)
    frames = []  # (t, joint features, raw gaze point or None, corrected screen point or None)
    learner = Learner(model, taps) if use_taps else None
    ignored, stages = collections.Counter(), collections.Counter()
    heard = None  # last rejected sound: (message, until)
    n = model.cols * model.rows
    rng = random.Random(seed)
    bw, bh = c.width / model.cols, (c.height - BAR) / model.rows

    def rect(cell):
        r, col = divmod(cell, model.cols)
        return int(col * bw) + 8, int(BAR + r * bh) + 8, int((col + 1) * bw) - 8, int(BAR + (r + 1) * bh) - 8

    # The model's cells cover the whole canvas; map a touch to a button by the cell it predicts.
    guided, ask = True, rng.randrange(n)
    tally = {"right": 0, "wrong": 0, "not_sure": 0}
    log, result, pending = [], None, None  # result: (cell, colour, message, until)
    try:
        key = wait_until_ready(app, "Button board", [
            *([hint + "  Press R to re-centre now."] if hint else ["Re-centred recently. Press R to re-centre again (15 s)."]),
            f"{n} big buttons ({model.cols} x {model.rows}), sized from what calibration found works best.",
            "Look at a button and TAP the screen on it: the tap sound says WHEN, your eyes say WHICH button."
            if use_taps else "Look at a button and touch it, holding still briefly.",
            "Guided: touch the button Magic Tap asks for; it marks right / wrong.",
            "Free (press F): touch any button, then press Y if Magic Tap got it right or N if not.",
        ], keys=(ord("r"), ord("R")))
        if key in (ord("r"), ord("R")):
            if taps is not None:
                taps.close()
            app.close()
            from .recenter import run_recenter

            run_recenter(camera)
            return run_buttons(camera, seed)
        while True:
            obs = app.step()
            if use_taps:
                cf, events = _tap_step(obs, model, taps, frames, c, correction)
                for msg in [e[3] for e in events if e[0] == "ignored"]:
                    heard = (msg, obs.t + 1.2)
                    ignored[msg] += 1
                stages["sounds heard"] += len(events)
                events = [e for e in events if e[0] != "ignored"]
                if guided and learner:
                    for e in events:  # you were asked to tap `ask`, so each tap teaches Magic Tap
                        learner.add(ask, feat=e[3], gaze=e[4])
            else:
                cf = engine.update(obs)
                events = cf.events
            for kind, cell, conf, *_ in events:
                if kind == "reject":
                    tally["not_sure"] += 1
                    result = (None, (90, 90, 90), "Not sure - didn't click. Try again.", obs.t + 1.2)
                elif kind == "down":
                    name = LABELS[cell]
                    if guided:
                        ok = cell == ask
                        tally["right" if ok else "wrong"] += 1
                        log.append({"asked": LABELS[ask], "detected": name, "confidence": conf, "correct": ok})
                        msg = f"Detected {name}  -  {'RIGHT' if ok else f'WRONG (you were asked for {LABELS[ask]})'}"
                        result = (cell, (0, 170, 0) if ok else (0, 0, 200), msg, obs.t + 1.5)
                        ask = rng.choice([k for k in range(n) if k != ask])
                    else:
                        pending = cell
                        result = (cell, (200, 140, 0), f"Detected {name} - was that right?  Y = yes, N = no",
                                  obs.t + 30)

            img = c.blank(18)
            for cell in range(n):
                x0, y0, x1, y1 = rect(cell)
                fill = (45, 45, 45)
                if guided and cell == ask:
                    fill = (0, 140, 190)
                if result and result[0] == cell and obs.t < result[3]:
                    fill = result[1]
                elif cf.state == "hover" and cf.cell == cell:
                    cv2.rectangle(img, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), (255, 160, 60), 3)
                cv2.rectangle(img, (x0, y0), (x1, y1), fill, -1)
                text(img, LABELS[cell], ((x0 + x1) / 2, (y0 + y1) / 2), 3.0, (255, 255, 255), 6, center=True)
            text(img, _summary_lines(calib, model)[0], (c.width / 2, 22), 0.5, (200, 200, 200), 1, center=True)
            done = tally["right"] + tally["wrong"]
            score = f"{tally['right']}/{done} right" if done else "no touches yet"
            mode = f"GUIDED: touch {LABELS[ask]}" if guided else "FREE: touch any button"
            status = f"{mode}     {score}   (not sure: {tally['not_sure']})     [F] guided/free  [R] reset  [Esc] quit"
            text(img, status, (c.width / 2, 52), 0.6, (0, 220, 255), 1, center=True)
            if result and obs.t < result[3]:
                text(img, result[2], (c.width / 2, c.height - 24), 0.8, (255, 255, 255), 2, center=True)
            elif heard and obs.t < heard[1]:
                text(img, f"Heard a sound - ignored ({heard[0]})", (c.width / 2, c.height - 24), 0.6,
                     (150, 150, 150), 1, center=True)

            key = app.show(img)
            if key == 27:
                break
            if key in (ord("f"), ord("F")):
                guided, pending, result = not guided, None, None
            elif key in (ord("r"), ord("R")):
                tally = {"right": 0, "wrong": 0, "not_sure": 0}
                log = []
            elif pending is not None and key in (ord("y"), ord("Y"), ord("n"), ord("N")):
                ok = key in (ord("y"), ord("Y"))
                tally["right" if ok else "wrong"] += 1
                log.append({"asked": None, "detected": LABELS[pending], "correct": ok})
                result = (pending, (0, 170, 0) if ok else (0, 0, 200), "Marked RIGHT" if ok else "Marked WRONG",
                          obs.t + 1.0)
                pending = None
    except KeyboardInterrupt:
        pass
    finally:
        if learner is not None:
            print(learner.finish())
        if taps is not None:
            taps.close()
        if log:
            out = SESSIONS_DIR / f"buttons-{time.strftime('%Y%m%d-%H%M%S')}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"grid": [model.cols, model.rows], "tally": tally, "touches": log,
                                       "sounds_heard": stages["sounds heard"], "ignored": dict(ignored)}, indent=2))
            print(f"Sounds heard: {stages['sounds heard']}; ignored: {dict(ignored) or 'none'}")
            print(f"{tally['right']}/{tally['right'] + tally['wrong']} right, {tally['not_sure']} 'not sure'. "
                  f"Saved to {out}")
        app.close()


class _Live:
    def __init__(self, state, cell, p_touch=0.0):
        self.state, self.cell, self.p_touch = state, cell, p_touch


def _tap_step(obs, model, taps, frames, canvas, correction=None):
    """Keep recent frames; turn accepted taps into button presses using where the eyes settled."""
    g = obs.gaze.features if obs.gaze is not None and not obs.gaze.blink else None
    raw = model.gaze.predict(g[None])[0] if g is not None else None  # before the touch-shift correction
    point = apply_correction(correction, raw + model.offset) if raw is not None else None
    frames.append((obs.t, joint_features(g, obs.features), raw, point))
    while frames and obs.t - frames[0][0] > 2.0:
        frames.pop(0)
    recent = [p for t, _, _, p in frames if p is not None and obs.t - t < 0.3]
    live_cell = model.cell_of(np.median(recent, axis=0)) if recent else None

    events = []
    for t, accepted, score, reason, feat in taps.poll():
        if not accepted:
            events.append(("ignored", None, score, reason))
            continue
        looks = select_fixation([(ft, p) for ft, _, _, p in frames], t)
        if len(looks) < 2:
            events.append(("ignored", None, score, "couldn't see your eyes"))
            continue
        gx, gy = np.median(np.array(looks), axis=0)
        if not (-100 < gx < canvas.width + 100 and -100 < gy < canvas.height + 100):
            events.append(("ignored", None, score, "you weren't looking at the screen"))
            continue
        cell, conf = vote([model.cell_of(p) for p in looks])
        raws = [r for ft, _, r, _ in frames if r is not None and t + TAP_WINDOW[0] <= ft <= t + TAP_WINDOW[1]]
        raw_med = np.median(raws, axis=0) if raws else None
        events.append(("down" if conf >= MIN_TAP_CONFIDENCE else "reject", cell, conf, feat, raw_med))
    return _Live("hover" if live_cell is not None else "none", live_cell), events


class Learner:
    """Learns from guided taps: you were asked for a button, so every tap is a labelled example.

    - Tap sound: each heard tap joins the tap model (most recent 40 kept), refitted every 5.
    - Eyes: the touch shift (gaze offset while touching) is updated from where you looked vs.
      the button you were asked for, blended with the calibration value (worth 10 taps).
    """

    PRIOR_WEIGHT = 10
    MAX_SHIFT_SAMPLES = 40

    def __init__(self, model, taps):
        self.model, self.taps = model, taps
        self.prior = np.array(model.offset, float)
        self.residuals, self.new_taps, self.pending = [], 0, []

    def add(self, asked, feat, gaze):
        self.pending.append(feat)
        self.new_taps += 1
        if len(self.pending) >= 5:
            self.taps.model.learn_taps(self.pending)
            self.pending = []
        if self.model.method == "eyes + touch shift" and gaze is not None:
            self.residuals = (self.residuals + [self.model.centre(asked) - gaze])[-self.MAX_SHIFT_SAMPLES:]
            n = len(self.residuals)
            self.model.offset = (self.PRIOR_WEIGHT * self.prior + np.sum(self.residuals, axis=0)) / (self.PRIOR_WEIGHT + n)

    def finish(self):
        if self.pending:
            self.taps.model.learn_taps(self.pending)
        if not self.new_taps:
            return "Nothing new learned."
        self.taps.save()
        self.model.save()
        return (f"Learned from {self.new_taps} guided taps. Touch shift "
                f"{np.round(self.prior).tolist()} -> {np.round(self.model.offset).tolist()} pt. Saved.")
