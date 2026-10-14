"""Turns per-frame observations into hover / touch events in canvas points.

WHEN: the arm is in a "touching the screen" posture (classifier probability
high) and has stopped moving. Released when the posture goes away or the arm
moves again.
WHERE: where the eyes were looking just before the touch (people look at what
they touch), blended with the weaker body-posture estimate.
"""

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

P_ON, P_OFF, P_HOVER = 0.6, 0.35, 0.5
STILL_FRAMES = 4  # consecutive still frames (~130 ms) before a touch fires
RELEASE_FRAMES = 2
MIN_CLICK_GAP_S = 0.4
LOST_TIMEOUT_S = 0.2
GAZE_WINDOW_S = 0.6  # where you looked during this window before the touch


class OneEuroFilter:
    """Low lag when moving fast, strong smoothing when still (Casiez et al. 2012)."""

    def __init__(self, min_cutoff: float = 0.8, beta: float = 0.01, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x = self.dx = self.t = None

    def __call__(self, x, t: float) -> np.ndarray:
        x = np.asarray(x, float)
        if self.x is None:
            self.x, self.dx, self.t = x, np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        self.dx = self.dx + self._alpha(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self.dx))
        self.x = self.x + self._alpha(cutoff, dt) * (x - self.x)
        self.t = t
        return self.x


@dataclass
class TouchFrame:
    state: str  # "none" | "hover" (reaching toward the screen) | "touch"
    point: np.ndarray | None  # canvas points: touch position, or where you're looking
    events: list = field(default_factory=list)  # [("down" | "up", point)]
    p_touch: float = 0.0
    motion: float = 0.0
    gaze: np.ndarray | None = None  # smoothed gaze point


class TouchEngine:
    def __init__(self, model, canvas_size):
        self.model = model
        self.w, self.h = canvas_size
        self.gaze_filter = OneEuroFilter(min_cutoff=0.5, beta=0.005)
        self.gaze_hist = deque()  # (t, raw gaze point)
        self.body_recent = deque(maxlen=STILL_FRAMES)
        self.state = "none"
        self.touch_point = None
        self.still = self.off = 0
        self.last_seen = -1e9
        self.last_click = -1e9

    def _clip(self, xy):
        return np.clip(xy, 0, (self.w - 1, self.h - 1))

    def _release(self, events):
        if self.state == "touch":
            events.append(("up", self.touch_point))
        self.state, self.touch_point = "none", None
        self.still = self.off = 0
        self.body_recent.clear()

    def _gaze_before(self, t):
        pts = [p for tt, p in self.gaze_hist if t - GAZE_WINDOW_S <= tt <= t]
        return np.median(np.array(pts), axis=0) if pts else None

    def update(self, obs) -> TouchFrame:
        t, events = obs.t, []
        gaze = getattr(obs, "gaze", None)
        if gaze is not None and not gaze.blink:
            g = self._clip(self.model.predict_gaze(gaze.features))
            self.gaze_hist.append((t, g))
            self.gaze_filter(g, t)
        while self.gaze_hist and t - self.gaze_hist[0][0] > 2 * GAZE_WINDOW_S:
            self.gaze_hist.popleft()
        gaze_pt = self.gaze_filter.x

        if obs.features is None:
            if t - self.last_seen > LOST_TIMEOUT_S:
                self._release(events)
            return TouchFrame(self.state, self.touch_point if self.state == "touch" else gaze_pt, events, gaze=gaze_pt)
        self.last_seen = t

        body_xy, p = self.model.predict_body(obs.features)
        if body_xy is not None:
            self.body_recent.append(self._clip(body_xy))
        motion = obs.motion

        if self.state != "touch":
            still = p > P_ON and motion < self.model.still_thr
            self.still = self.still + 1 if still else 0
            if self.still >= STILL_FRAMES and t - self.last_click > MIN_CLICK_GAP_S:
                body_pt = np.median(np.array(self.body_recent), axis=0) if self.body_recent else None
                fused = self.model.fuse(self._gaze_before(t), body_pt)
                if fused is not None:
                    self.touch_point = self._clip(fused)
                    self.state, self.off, self.last_click = "touch", 0, t
                    events.append(("down", self.touch_point.copy()))
            if self.state != "touch":
                self.state = "hover" if p > P_HOVER else "none"
        else:
            leaving = p < P_OFF or motion > self.model.release_thr
            self.off = self.off + 1 if leaving else 0
            if self.off >= RELEASE_FRAMES:
                events.append(("up", self.touch_point.copy()))
                self.state, self.touch_point, self.still = "hover", None, 0
        shown = self.touch_point if self.state == "touch" else gaze_pt
        return TouchFrame(self.state, shown, events, p, motion, gaze_pt)


CELL_WINDOW_S = 0.4  # frames averaged when deciding which cell was touched


@dataclass
class CellFrame:
    state: str  # "none" | "hover" | "touch"
    cell: int | None  # cell being touched, or the cell holo thinks you're heading for
    confidence: float
    events: list = field(default_factory=list)  # [("down" | "up" | "reject", cell, confidence)]
    p_touch: float = 0.0
    motion: float = 0.0


class CellTouchEngine:
    """WHEN from body posture + stillness; WHICH CELL from one eyes+body classifier."""

    def __init__(self, model):
        self.model = model
        self.frames = deque()  # (t, joint features)
        self.state = "none"
        self.cell = None
        self.still = self.off = 0
        self.last_seen = -1e9
        self.last_click = -1e9
        self.live_cell, self.live_conf = None, 0.0

    def update(self, obs) -> CellFrame:
        from .model import joint_features

        t, events = obs.t, []
        gaze = getattr(obs, "gaze", None)
        g = gaze.features if gaze is not None and not gaze.blink else None
        if g is not None or obs.features is not None:
            self.frames.append((t, joint_features(g, obs.features)))
        while self.frames and t - self.frames[0][0] > CELL_WINDOW_S:
            self.frames.popleft()

        if obs.features is None:
            if t - self.last_seen > LOST_TIMEOUT_S and self.state == "touch":
                events.append(("up", self.cell, 0.0))
                self.state, self.cell, self.still = "none", None, 0
            return CellFrame(self.state, self.cell, 0.0, events)
        self.last_seen = t
        p, motion = self.model.p_touch(obs.features), obs.motion

        if self.state != "touch":
            if p > P_HOVER and self.frames:  # live preview of the likely cell
                self.live_cell, self.live_conf = self.model.predict_cell([f for _, f in self.frames])
            else:
                self.live_cell, self.live_conf = None, 0.0
            self.still = self.still + 1 if p > P_ON and motion < self.model.still_thr else 0
            if self.still >= STILL_FRAMES and t - self.last_click > MIN_CLICK_GAP_S and self.frames:
                cell, conf = self.model.predict_cell([f for _, f in self.frames])
                self.last_click = t
                if conf >= self.model.threshold:
                    self.state, self.cell, self.off = "touch", cell, 0
                    events.append(("down", cell, conf))
                else:
                    events.append(("reject", cell, conf))
                    self.still = -int(MIN_CLICK_GAP_S * 30)  # don't immediately retry the same pose
            if self.state != "touch":
                self.state = "hover" if p > P_HOVER else "none"
            return CellFrame(self.state, self.live_cell, self.live_conf, events, p, motion)

        leaving = p < P_OFF or motion > self.model.release_thr
        self.off = self.off + 1 if leaving else 0
        if self.off >= RELEASE_FRAMES:
            events.append(("up", self.cell, 0.0))
            self.state, self.cell, self.still = "hover", None, 0
        return CellFrame(self.state, self.cell, 1.0, events, p, motion)
