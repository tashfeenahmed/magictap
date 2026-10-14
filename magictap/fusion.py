"""Pure logic shared by `magictap serve`, `magictap mouse` and the button board (no camera, mic or sockets).

A tap at time t0 is placed where the eyes last *settled* before it: walking back
from the tap over [t0 - 0.8 s, t0 + 0.05 s], the most recent stable run of gaze
samples (a fixation). That skips the saccade from the previous target, which a
fixed window can still contain. If there's no clear fixation, it falls back to
[t0 - 0.30 s, t0 + 0.05 s]. Registered targets are hit-tested against the
median of those samples.
"""

import json
from collections import deque

import numpy as np

TAP_WINDOW = (-0.30, 0.05)
FIXATION_LOOKBACK = 0.8  # seconds before the tap searched for a fixation
FIXATION_TOL = 60.0  # points: samples within this of the run's median belong to it
FIXATION_MIN = 4  # samples (~0.13 s at 30 fps) for a run to count as a fixation
MIN_SAMPLES = 2
MIN_CONFIDENCE = 0.6  # below this, say "not sure" instead of clicking
OFFSCREEN_MARGIN = 100  # points


def select_fixation(items, t0, lookback=FIXATION_LOOKBACK, after=TAP_WINDOW[1], tol=FIXATION_TOL,
                    min_len=FIXATION_MIN, fallback=TAP_WINDOW):
    """Gaze samples from the most recent fixation before t0.

    items: iterable of (t, (x, y) or None). Walk back from the tap, growing the newest
    run of samples while each one stays within `tol` of the run's median; the first run
    with at least `min_len` samples is the fixation. Falls back to the fixed window.
    """
    items = [(t, xy) for t, xy in items if xy is not None]
    pts = [xy for t, xy in items if t0 - lookback <= t <= t0 + after]
    i = len(pts)
    P = np.array(pts, float) if pts else np.zeros((0, 2))
    while i > 0:
        j = i - 1
        while j > 0 and np.hypot(*(P[j - 1] - np.median(P[j - 1:i], axis=0))) < tol:
            j -= 1
        if i - j >= min_len:
            return [tuple(p) for p in P[j:i]]
        i = j
    return [xy for t, xy in items if t0 + fallback[0] <= t <= t0 + fallback[1]]


def vote(cells):
    """Most common cell and the share of samples that agree with it."""
    if not cells:
        return None, 0.0
    win = max(set(cells), key=cells.count)
    return win, cells.count(win) / len(cells)


class GazeBuffer:
    """Recent gaze points (global screen points) with timestamps."""

    def __init__(self, keep_s=2.0):
        self.keep_s = keep_s
        self.items = deque()

    def add(self, t, xy):
        self.items.append((t, None if xy is None else (float(xy[0]), float(xy[1]))))
        while self.items and t - self.items[0][0] > self.keep_s:
            self.items.popleft()

    def window(self, t0, window=TAP_WINDOW):
        return [xy for t, xy in self.items if xy is not None and t0 + window[0] <= t <= t0 + window[1]]

    def fixation(self, t0):
        return select_fixation(self.items, t0)

    def latest(self):
        for t, xy in reversed(self.items):
            if xy is not None:
                return t, xy
        return None


def to_global(canvas_xy, canvas_offset):
    return canvas_xy[0] + canvas_offset[0], canvas_xy[1] + canvas_offset[1]


def contains(target, xy):
    return target["x"] <= xy[0] <= target["x"] + target["w"] and target["y"] <= xy[1] <= target["y"] + target["h"]


def hit_test(xy, targets):
    """The registered target containing xy (the smallest one if they overlap), else None."""
    hits = [tg for tg in targets if contains(tg, xy)]
    return min(hits, key=lambda tg: tg["w"] * tg["h"]) if hits else None


def confidence(samples, target):
    """Share of window gaze samples that fall inside the target."""
    if not samples or target is None:
        return 0.0
    return sum(contains(target, s) for s in samples) / len(samples)


def on_screen(xy, screen, margin=OFFSCREEN_MARGIN):
    return -margin <= xy[0] <= screen[0] + margin and -margin <= xy[1] <= screen[1] + margin


def resolve_tap(t0, samples, screen, targets):
    """Turn a heard tap + the gaze samples around it into a protocol message."""
    if len(samples) < MIN_SAMPLES:
        return ignored_msg("eyes not on screen", t0)
    point = tuple(float(v) for v in np.median(np.array(samples), axis=0))
    if not on_screen(point, screen):
        return ignored_msg("eyes not on screen", t0)
    target = hit_test(point, targets) if targets else None
    conf = confidence(samples, target) if target else float(
        np.mean([np.hypot(s[0] - point[0], s[1] - point[1]) < 150 for s in samples]))
    return tap_msg(point, target["id"] if target else None, conf, t0)


# ------------------------------------------------------------------ protocol messages
def hello_msg(version, screen):
    return {"type": "hello", "version": version, "screen": [int(screen[0]), int(screen[1])]}


def tap_msg(xy, target_id, conf, t):
    return {"type": "tap", "x": round(xy[0], 1), "y": round(xy[1], 1), "target": target_id,
            "confidence": round(float(conf), 3), "t": round(float(t), 3)}


def ignored_msg(reason, t=None):
    m = {"type": "ignored", "reason": reason}
    if t is not None:
        m["t"] = round(float(t), 3)
    return m


def gaze_msg(xy):
    return {"type": "gaze", "x": round(xy[0], 1), "y": round(xy[1], 1)}


def error_msg(text):
    return {"type": "error", "message": text}


def parse_client(raw):
    """Client message -> ("targets", [...]) | ("subscribe", bool) | ("error", text)."""
    try:
        m = json.loads(raw)
    except (TypeError, ValueError):
        return "error", "not JSON"
    kind = m.get("type") if isinstance(m, dict) else None
    if kind == "targets":
        out = []
        for tg in m.get("targets", []):
            try:
                out.append({"id": tg["id"], "x": float(tg["x"]), "y": float(tg["y"]),
                            "w": float(tg["w"]), "h": float(tg["h"])})
            except (KeyError, TypeError, ValueError):
                return "error", "each target needs id, x, y, w, h"
        return "targets", out
    if kind == "subscribe":
        return "subscribe", bool(m.get("gaze", False))
    return "error", f"unknown message type: {kind!r}"
