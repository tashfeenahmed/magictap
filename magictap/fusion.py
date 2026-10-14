"""Pure logic shared by `magictap serve` and `magictap mouse` (no camera, mic or sockets).

A tap at time t0 is placed at the median gaze point over [t0 - 0.30 s, t0 + 0.05 s]
(gaze settles on the target before the finger lands; earlier frames still hold
the previous target). Registered targets are hit-tested against that point.
"""

import json
from collections import deque

import numpy as np

TAP_WINDOW = (-0.30, 0.05)
MIN_SAMPLES = 2
OFFSCREEN_MARGIN = 100  # points


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
