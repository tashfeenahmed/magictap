"""Tap calibration: learn what *your* screen taps sound like vs. everyday noise.

Segments (each starts with Space):
  taps     - tap dots on the screen (firm, then gentle)
  typing   - type anything
  talking  - talk or read aloud
  noises   - knock the desk, clap, tap the palm rest, click the trackpad, put things down
  media    - (optional) play music or a video
Every sound in the noise segments is a "must not trigger" example. The report
gives the tap hit rate and false alarms, both measured on data the classifier
never trained on.
"""

import json
import random
import time

import cv2
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .app import text
from .audio import Mic, realign_clip, seconds_since_input, tap_features
from .config import DATA_DIR
from .screen import main_canvas, open_fullscreen

TAP_MODEL_PATH = DATA_DIR / "tap_model.joblib"
WINDOW = "Magic Tap - taps"
KEY_GATE_S = 0.15  # ignore sounds this close to a key press / trackpad click


class Abort(Exception):
    pass


def _show(img):
    cv2.imshow(WINDOW, img)
    key = cv2.waitKey(15) & 0xFF
    if key == 27:
        raise Abort
    return key


def _screen(canvas, lines):
    while True:
        img = canvas.blank(18)
        for n, s in enumerate(lines):
            text(img, s, (canvas.width / 2, canvas.height / 2 - 60 + 38 * n), 0.9 if n == 0 else 0.7,
                 (255, 255, 255) if n == 0 else (210, 210, 210), 2 if n == 0 else 1, center=True)
        text(img, "Press SPACE to start   (Esc quit)", (canvas.width / 2, canvas.height - 50), 0.75, (0, 220, 255),
             center=True)
        if _show(img) == ord(" "):
            return


def _collect(mic, out, label, keyed):
    for t, clip in mic.onsets():
        out.append({"t": t, "label": label, "clip": clip,
                    "key_near": keyed(t)})


def _keyed(t):
    """Was there a key press / trackpad click within KEY_GATE_S of the sound at time t?"""
    since = seconds_since_input()
    return since < (time.monotonic() - t) + KEY_GATE_S


def _timed(canvas, mic, events, label, seconds, title, hint):
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        _collect(mic, events, label, _keyed)
        img = canvas.blank(18)
        text(img, title, (canvas.width / 2, canvas.height / 2 - 40), 1.0, center=True)
        text(img, hint, (canvas.width / 2, canvas.height / 2 + 5), 0.7, (210, 210, 210), 1, center=True)
        left = seconds - (time.monotonic() - t0)
        text(img, f"{left:.0f} s   sounds heard: {sum(e['label'] == label for e in events)}",
             (canvas.width / 2, canvas.height / 2 + 50), 0.7, (0, 220, 255), center=True)
        if _show(img) in (ord("s"), ord("S")):
            return


def _taps(canvas, mic, events, n_firm=14, n_gentle=8, seed=None):
    rng = random.Random(seed)
    for k in range(n_firm + n_gentle):
        gentle = k >= n_firm
        p = (rng.uniform(0.1, 0.9) * canvas.width, rng.uniform(0.12, 0.9) * canvas.height)
        shown = time.monotonic()
        mic.onsets()  # drop anything heard before the dot appeared
        got = None
        while got is None:
            for t, clip in mic.onsets():
                if t > shown + 0.15:
                    got = (t, clip)
                    break
            img = canvas.blank(18)
            cv2.circle(img, (int(p[0]), int(p[1])), 34, (0, 220, 255), -1, cv2.LINE_AA)
            msg = "Tap the dot GENTLY (a light tap)" if gentle else "Tap the dot with your fingertip"
            if time.monotonic() - shown > 4:
                msg = "Didn't hear it - tap a little firmer (or press S to skip)"
            y = canvas.height - 30 if p[1] < canvas.height / 2 else 30
            text(img, f"{msg}   ({k + 1}/{n_firm + n_gentle})", (canvas.width / 2, y), 0.6, (200, 200, 200), 1,
                 center=True)
            if _show(img) in (ord("s"), ord("S")):
                break
        if got:
            events.append({"t": got[0], "label": "tap", "clip": got[1], "key_near": _keyed(got[0]),
                           "gentle": gentle})
            img = canvas.blank(18)
            cv2.circle(img, (int(p[0]), int(p[1])), 34, (0, 200, 0), -1, cv2.LINE_AA)
            _show(img)
            time.sleep(0.4)
            mic.onsets()  # ignore the echo/second bounce of the same tap


def _rate(flags):
    return float(np.mean(flags)) if len(flags) else 0.0


MAX_LEARNED_TAPS = 40  # taps learned from the button screen (most recent kept)
MIN_THRESHOLD = 0.35


class TapModel:
    """Tap-vs-noise classifier plus a novelty check ("does this sound like any of your taps?")."""

    CANDIDATES = {
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000)),
        "trees": lambda: ExtraTreesClassifier(400, min_samples_leaf=2, random_state=0, n_jobs=-1),
    }

    def __init__(self):
        self.name, self.clf, self.threshold = None, None, 0.5
        self.X = self.y = None
        self.learned = []  # tap feature vectors learned from the button screen
        self.ref = self.scale = None
        self.novelty_threshold = np.inf
        self.report = {}

    # -- novelty: distance to the nearest known tap, in standardised feature space
    @staticmethod
    def _novelty_fit(taps):
        mu, sd = taps.mean(0), taps.std(0) + 1e-6
        z = (taps - mu) / sd
        d = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        nn = d.min(1)
        return (mu, sd), z, float(np.percentile(nn, 95) * 1.5)

    @staticmethod
    def _novelty(scale, ref, X):
        z = (X - scale[0]) / scale[1]
        return np.sqrt(((z[:, None, :] - ref[None, :, :]) ** 2).sum(-1)).min(1)

    def fit(self, X, y, labels, keyed, gentle, progress=print):
        fold = np.zeros(len(X), int)
        for lab in np.unique(labels):
            idx = np.where(labels == lab)[0]
            fold[idx] = np.arange(len(idx)) * 4 // max(len(idx), 1)
        best = None
        for name, make in self.CANDIDATES.items():
            prob, novel = np.zeros(len(X)), np.zeros(len(X), bool)
            for k in range(4):
                tr = fold != k
                if len(np.unique(y[tr])) < 2:
                    continue
                prob[~tr] = make().fit(X[tr], y[tr]).predict_proba(X[~tr])[:, 1]
                scale, ref, nthr = self._novelty_fit(X[tr & y])
                novel[~tr] = self._novelty(scale, ref, X[~tr]) > nthr
            # Threshold: catch as many held-out taps as possible with zero held-out false alarms
            # after the keyboard and novelty checks.
            neg = ~y & ~keyed & ~novel
            thr = max(prob[neg].max() + 1e-3 if neg.any() else 0, MIN_THRESHOLD)
            caught = (prob >= thr) & ~novel
            hit = _rate(caught[y])
            progress(f"{name}: catches {hit:.0%} of taps with no false alarms")
            if best is None or hit > best[2]:
                best = (name, make, hit, thr, prob, novel)
        name, make, hit, thr, prob, novel = best
        caught = (prob >= thr) & ~novel
        self.report = {
            "model": name, "threshold": float(thr), "taps": int(y.sum()), "tap_catch_rate": hit,
            "gentle_catch_rate": _rate(caught[y & gentle]),
            "taps_rejected_as_unusual": int((y & novel).sum()), "noise": {}}
        for lab in np.unique(labels[~y]):
            s = labels == lab
            self.report["noise"][lab] = {
                "sounds": int(s.sum()), "blocked_by_keyboard_gate": int((s & keyed).sum()),
                "blocked_as_unusual": int((s & ~keyed & novel).sum()),
                "would_trigger_without_gate": int((s & (prob >= thr)).sum()),
                "false_alarms": int((s & ~keyed & caught).sum())}
        self.name, self.threshold, self.X, self.y = name, float(thr), X, y
        self._refit()
        return self

    def _refit(self):
        X = np.vstack([self.X] + ([np.array(self.learned)] if self.learned else []))
        y = np.r_[self.y, np.ones(len(self.learned), bool)]
        self.clf = self.CANDIDATES[self.name]().fit(X, y)
        self.scale, self.ref, self.novelty_threshold = self._novelty_fit(X[y])

    def learn_taps(self, feats):
        """Add confirmed taps (e.g. from guided buttons) and refit."""
        self.learned = (self.learned + [np.asarray(f) for f in feats])[-MAX_LEARNED_TAPS:]
        self._refit()

    def judge(self, feat):
        """(accepted, score, reason)"""
        score = float(self.clf.predict_proba(feat[None])[0, 1])
        if score < self.threshold:
            return False, score, "not a tap"
        if self._novelty(self.scale, self.ref, feat[None])[0] > self.novelty_threshold:
            return False, score, "unusual sound"
        return True, score, ""


def train(events, progress=print, realign=False):
    """Fit a TapModel from recorded events; returns the model (with .report)."""
    clips = [realign_clip(e["clip"]) if realign else e["clip"] for e in events]
    X = np.array([tap_features(c) for c in clips])
    y = np.array([e["label"] == "tap" for e in events])
    labels = np.array([e["label"] for e in events])
    keyed = np.array([bool(e["key_near"]) for e in events])
    gentle = np.array([bool(e.get("gentle")) for e in events])
    return TapModel().fit(X, y, labels, keyed, gentle, progress)


def retrain_taps(run_dir=None):
    """Retrain from saved tap sounds (realigning clips recorded before exact alignment)."""
    from pathlib import Path

    runs = sorted((DATA_DIR / "tap_runs").glob("*/sounds.npz"))
    path = Path(run_dir) / "sounds.npz" if run_dir else (runs[-1] if runs else None)
    if path is None or not path.exists():
        raise SystemExit("No saved tap recordings. Run `magictap taps` first.")
    d = np.load(path)
    events = [{"clip": c, "label": str(l), "key_near": bool(k), "gentle": bool(g)}
              for c, l, k, g in zip(d["clips"], d["labels"], d["keyed"], d["gentle"])]
    model = train(events, realign=True)
    joblib.dump(model, TAP_MODEL_PATH)
    print("\n".join(tap_report_lines(model.report)))
    return model


def run_tap_calibration(seed=None):
    canvas = main_canvas()
    open_fullscreen(WINDOW)
    mic = Mic()
    events = []
    out_dir = DATA_DIR / "tap_runs" / time.strftime("%Y%m%d-%H%M%S")
    try:
        _screen(canvas, ["Tap training (~2 min)",
                         "Holo will learn what YOUR taps on the screen sound like,",
                         "and what must NOT count: typing, talking, knocks, claps, background noise.",
                         "Step 1: tap the dots (firm taps, then gentle ones)."])
        _taps(canvas, mic, events, seed=seed)
        _screen(canvas, ["Step 2: Typing (15 s)", "Type anything on the keyboard, normally."])
        _timed(canvas, mic, events, "typing", 15, "Type anything", "Normal typing on the keyboard")
        _screen(canvas, ["Step 3: Talking (10 s)", "Talk, or read this aloud: 'the quick brown fox jumps over the lazy dog'"])
        _timed(canvas, mic, events, "talking", 10, "Talk", "Say anything, any volume")
        _screen(canvas, ["Step 4: Other sounds (20 s)",
                         "Knock on the desk, clap, tap the palm rest or laptop body, click the trackpad,",
                         "put a cup or phone down, tap the table - anything that ISN'T tapping the screen."])
        _timed(canvas, mic, events, "noises", 20, "Make everyday noises", "Desk knocks, claps, palm rest taps, things set down")
        _screen(canvas, ["Step 5 (optional): Media (15 s)",
                         "Play some music or a video out loud now, then press SPACE. Press S to skip."])
        _timed(canvas, mic, events, "media", 15, "Media playing", "Leave it playing - press S to skip")
        _timed(canvas, mic, events, "quiet", 4, "Quiet please", "Just sit still for a moment")
    except Abort:
        print("Tap training cancelled.")
        mic.close()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return
    mic.close()
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "sounds.npz", clips=np.array([e["clip"] for e in events]),
                        labels=np.array([e["label"] for e in events]), keyed=np.array([e["key_near"] for e in events]),
                        gentle=np.array([bool(e.get("gentle")) for e in events]))
    taps = sum(e["label"] == "tap" for e in events)
    if taps < 8:
        raise SystemExit(f"Only {taps} taps were heard - try again in a quieter moment, tapping a bit firmer.")
    model = train(events)
    report = model.report
    report["detector"] = dict(mic.counts)
    joblib.dump(model, TAP_MODEL_PATH)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    lines = tap_report_lines(report)
    print("\n".join(lines))
    img = canvas.blank(18)
    for n, s in enumerate(lines + ["", "Press any key."]):
        text(img, s, (canvas.width / 2, canvas.height / 2 - 120 + 34 * n), 0.65, center=True)
    cv2.imshow(WINDOW, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def tap_report_lines(r):
    lines = [f"Screen taps recognised: {100 * r['tap_catch_rate']:.0f}% "
             f"(gentle taps {100 * r['gentle_catch_rate']:.0f}%)  - tested on taps the model didn't train on"]
    if r.get("taps_rejected_as_unusual"):
        lines.append(f"({r['taps_rejected_as_unusual']} real taps were rejected as 'unusual sound')")
    for lab, n in r["noise"].items():
        extra = []
        if n["blocked_by_keyboard_gate"]:
            extra.append(f"{n['blocked_by_keyboard_gate']} blocked by the keyboard check")
        if n.get("blocked_as_unusual"):
            extra.append(f"{n['blocked_as_unusual']} blocked as unusual")
        gate = f" ({', '.join(extra)})" if extra else ""
        lines.append(f"{lab}: {n['sounds']} sounds heard{gate} -> false taps: {n['false_alarms']}")
    return lines


class TapDetector:
    """Live: mic onsets -> accepted taps (with reasons for anything rejected)."""

    def __init__(self):
        if not TAP_MODEL_PATH.exists():
            raise SystemExit("No tap model yet. Run `magictap taps` first.")
        from . import legacy_unpickle_aliases

        legacy_unpickle_aliases()
        self.model = joblib.load(TAP_MODEL_PATH)
        if not isinstance(self.model, TapModel):
            raise SystemExit("Tap model is from an older version. Run `magictap taps --retrain`.")
        self.mic = Mic()

    def poll(self):
        """[(time, accepted, score, reason, features)] for sounds heard since the last call."""
        out = []
        for t, clip in self.mic.onsets():
            feat = tap_features(clip)
            if _keyed(t):
                out.append((t, False, 0.0, "keyboard/trackpad", feat))
                continue
            ok, score, reason = self.model.judge(feat)
            out.append((t, ok, score, reason, feat))
        return out

    def save(self):
        joblib.dump(self.model, TAP_MODEL_PATH)

    def close(self):
        self.mic.close()
