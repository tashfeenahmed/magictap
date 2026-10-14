"""Logic tests that don't need a camera: features, model training, touch state machine."""

import numpy as np

from holo.body import N_KP_FEATURES, N_MASK_FEATURES, BodyObs, features
from holo.calibration import grid_targets
from holo.config import MASK_H, MASK_W
from holo.model import TouchModel
from holo.touch import TouchEngine


def fake_pose(rng, reach=(0.5, 0.5)):
    pose = rng.uniform(0.2, 0.8, (33, 4)).astype(np.float32)
    pose[11, :2], pose[12, :2] = (0.6, 0.7), (0.4, 0.7)
    pose[15, :2] = reach  # wrist position encodes where the user reaches
    pose[13, :2] = (np.array(reach) + (0.5, 0.7)) / 2
    return pose


def test_feature_vector_layout():
    rng = np.random.default_rng(0)
    f = features(fake_pose(rng), [rng.uniform(0, 1, (21, 2))], rng.uniform(0, 1, (MASK_H, MASK_W)))
    assert f.shape == (N_KP_FEATURES + N_MASK_FEATURES,)
    assert np.all(np.isfinite(f))


def synthetic_dataset(seed=0, w=1728, h=1085):
    rng = np.random.default_rng(seed)
    targets = grid_targets(w, h, 5, 4)
    X, Y, G = [], [], []
    for g, tgt in enumerate(targets):
        reach = (0.2 + 0.6 * tgt[0] / w, 0.95 - 0.3 * tgt[1] / h)
        for _ in range(20):
            pose = fake_pose(rng, np.array(reach) + rng.normal(0, 0.004, 2))
            mask = np.zeros((MASK_H, MASK_W))
            X.append(features(pose, [], mask))
            Y.append(tgt)
            G.append(g)
    rest = [features(fake_pose(rng, (0.5 + rng.normal(0, 0.05), 1.1)), [], np.zeros((MASK_H, MASK_W)))
            for _ in range(80)]
    return np.array(X), np.array(Y), np.array(G), np.array(rest)


def synthetic_gaze(targets, n, rng, noise=0.01, group_offset=0, w=1728, h=1085):
    from holo.gaze import N_GEO_FEATURES, N_PATCH_FEATURES
    X, Y, G = [], [], []
    for g, t in enumerate(targets):
        for _ in range(n):
            f = rng.normal(0, 1, N_GEO_FEATURES + N_PATCH_FEATURES).astype(np.float32) * 0.1
            f[11], f[12] = t[0] / w * 0.3 - 0.15 + rng.normal(0, noise), t[1] / h * 0.2 + rng.normal(0, noise)
            X.append(f); Y.append(t); G.append(group_offset + g)
    return np.array(X), np.array(Y, float), np.array(G)


def test_model_uses_gaze_for_where_and_body_for_when():
    rng = np.random.default_rng(1)
    Xb, Yb, Gb, rest = synthetic_dataset()
    w, h = 1728, 1085
    gaze_train = synthetic_gaze(grid_targets(w, h, 5, 4), 20, rng)
    gaze_check = synthetic_gaze(grid_targets(w, h, 3, 3, 0.2), 20, rng, group_offset=1000)
    touch_targets = [Yb[Gb == g][0] for g in np.unique(Gb)]
    touch_gaze = synthetic_gaze(touch_targets, 20, rng, group_offset=2000)
    m = TouchModel().fit(gaze_train, gaze_check, touch_gaze, (Xb, Yb, Gb), rest,
                         hold_motion=[0.01] * 50, reach_motion=[0.3] * 50, canvas_size=(w, h))
    r = m.report
    assert r["gaze_validation"]["median"] < 60, r["gaze_validation"]
    assert r["final"]["median"] < r["final"]["baseline"] / 3
    assert r["touch_vs_rest_accuracy"] > 0.9
    assert m.w_gaze > 0 and m.release_thr > m.still_thr
    _, p_rest = m.predict_body(rest[0])
    assert p_rest < 0.5


class FakeModel:
    still_thr, release_thr = 0.05, 0.3

    def __init__(self):
        self.p = 0.9

    def predict_body(self, f):
        return np.array([400.0, 300.0]) + np.random.default_rng().normal(0, 2, 2), self.p

    def predict_gaze(self, f):
        return np.array([700.0, 500.0])

    def fuse(self, g, b):
        return b if g is None else 0.8 * np.asarray(g) + 0.2 * np.asarray(b)


def obs(t, motion, present=True):
    return BodyObs(t, features=np.zeros(3) if present else None, motion=motion)


def run(engine, motions, t0=0.0):
    events = []
    for i, m in enumerate(motions):
        events += [k for k, _ in engine.update(obs(t0 + i / 30, m)).events]
    return events


def test_reach_then_still_is_one_click_and_movement_releases():
    eng = TouchEngine(FakeModel(), (1000, 800))
    assert run(eng, [0.5] * 5 + [0.01] * 20 + [0.6] * 3) == ["down", "up"]


def test_moving_arm_never_clicks():
    eng = TouchEngine(FakeModel(), (1000, 800))
    assert run(eng, [0.5, 0.01, 0.5, 0.02, 0.4, 0.01, 0.01, 0.6] * 5) == []


def test_resting_posture_never_clicks():
    model = FakeModel()
    model.p = 0.1
    eng = TouchEngine(model, (1000, 800))
    assert run(eng, [0.0] * 60) == []


def test_touch_lands_where_you_were_looking():
    from types import SimpleNamespace
    eng = TouchEngine(FakeModel(), (1000, 800))
    gaze = SimpleNamespace(features=np.zeros(3), blink=False)
    down = None
    for i in range(10):
        o = obs(i / 30, 0.01)
        o.gaze = gaze
        for k, p in eng.update(o).events:
            if k == "down":
                down = p
    assert down is not None and np.allclose(down, 0.8 * np.array([700, 500]) + 0.2 * np.array([400, 300]), atol=5)


def test_body_lost_releases_touch():
    eng = TouchEngine(FakeModel(), (1000, 800))
    assert run(eng, [0.01] * 6) == ["down"]
    tf = eng.update(obs(2.0, 0.0, present=False))
    assert [k for k, _ in tf.events] == ["up"]


class FakeCellModel:
    still_thr, release_thr, threshold = 0.05, 0.3, 0.6

    def __init__(self, conf):
        self.conf = conf

    def p_touch(self, f):
        return 0.9

    def predict_cell(self, frames):
        return 4, self.conf


def test_cell_engine_clicks_confident_cell():
    from holo.touch import CellTouchEngine
    eng = CellTouchEngine(FakeCellModel(0.9))
    ev = []
    for i in range(10):
        ev += eng.update(obs(i / 30, 0.01)).events
    assert [e[:2] for e in ev] == [("down", 4)]


def test_cell_engine_refuses_when_unsure():
    from holo.touch import CellTouchEngine
    eng = CellTouchEngine(FakeCellModel(0.3))
    ev = []
    for i in range(10):
        ev += eng.update(obs(i / 30, 0.01)).events
    assert [e[0] for e in ev] == ["reject"]


def test_cell_model_learns_cells_from_eyes_and_arm():
    from holo.model import CellModel, fit_gaze_mapper, joint_features
    rng = np.random.default_rng(0)
    w, h = 1728, 1085
    mapper, name, check, _ = fit_gaze_mapper(synthetic_gaze(grid_targets(w, h, 5, 4), 20, rng),
                                             synthetic_gaze(grid_targets(w, h, 3, 3, 0.2), 20, rng, group_offset=1000),
                                             (w, h))
    assert check["median"] < 80
    m = CellModel(3, 2, (w, h))
    X, cells, caps, rounds = [], [], [], []
    cap = 0
    for rd in range(3):
        for cell in range(6):
            gx, _, _ = synthetic_gaze([m.centre(cell)], 15, rng)
            for g in gx:
                b = rng.normal(0, 1, N_KP_FEATURES + N_MASK_FEATURES).astype(np.float32)
                X.append(joint_features(g, b)); cells.append(cell); caps.append(cap); rounds.append(rd)
            cap += 1
    rest = rng.normal(0, 1, (60, N_KP_FEATURES + N_MASK_FEATURES)).astype(np.float32) + 3
    targets = np.array([m.centre(c) for c in cells])
    m.fit(mapper, np.array(X), targets, np.array(caps), np.array(rounds), rest, [0.01] * 30, [0.3] * 30)
    assert m.report["accuracy"] > 0.9, m.report["candidates"]
    assert m.predict_cell(X[:5])[0] == 0


def _clip(kind, rng):
    from holo.audio import HOP, SR, WIN_AFTER, WIN_BEFORE
    n = (WIN_BEFORE + WIN_AFTER) * HOP
    t = np.arange(n) / SR
    x = rng.normal(0, 1e-4, n)
    o = WIN_BEFORE * HOP
    tt = t[o:] - t[o]
    amp = rng.uniform(0.3, 1.0)
    if kind == "tap":  # sharp click + low lid thud, fast decay
        x[o:] += amp * (0.05 * rng.normal(0, 1, n - o) * np.exp(-tt / 0.003) + 0.05 * np.sin(2 * np.pi * 180 * tt) * np.exp(-tt / 0.02))
    elif kind == "typing":  # brighter click, no thud
        x[o:] += amp * 0.05 * rng.normal(0, 1, n - o) * np.exp(-tt / 0.008)
    else:  # voice-like: harmonic, slow
        x[o:] += amp * 0.03 * np.sin(2 * np.pi * rng.uniform(120, 250) * tt * np.arange(1, 2)[0]) * (1 - np.exp(-tt / 0.02))
    return x.astype(np.float32)


def test_tap_classifier_separates_taps_from_noise():
    from holo.audio import tap_features
    from holo.tapcal import train
    rng = np.random.default_rng(0)
    events = [{"label": "tap", "clip": _clip("tap", rng), "key_near": False, "gentle": i > 15} for i in range(22)]
    events += [{"label": "typing", "clip": _clip("typing", rng), "key_near": i % 3 != 0} for i in range(30)]
    events += [{"label": "talking", "clip": _clip("talk", rng), "key_near": False} for i in range(20)]
    model = train(events, progress=lambda s: None)
    report = model.report
    assert report["tap_catch_rate"] > 0.8, report
    assert all(n["false_alarms"] == 0 for n in report["noise"].values())
    # learning confirmed taps from the button screen keeps working and caps the memory
    model.learn_taps([tap_features(_clip("tap", rng)) for _ in range(50)])
    assert len(model.learned) == 40
    assert model.judge(tap_features(_clip("tap", rng)))[0]
    assert not model.judge(tap_features(_clip("talk", rng)))[0]


def test_onset_detector_aligns_taps_to_the_sample():
    from holo.audio import HOP, PRE, SR, OnsetDetector
    rng = np.random.default_rng(1)
    x = rng.normal(0, 3e-4, SR * 3).astype(np.float32)
    onsets = [int(SR * 1.6) + 37, int(SR * 2.2) + 311]  # anywhere inside a 10 ms block
    for o in onsets:
        k = np.arange(2000)
        x[o:o + 2000] += (0.02 * rng.normal(0, 1, 2000) * np.exp(-k / 150)).astype(np.float32)
    d, events = OnsetDetector(), []
    for i in range(0, len(x), HOP):
        events += d.feed(x[i:i + HOP], i / SR)
    assert len(events) == 2
    for (t, clip), o in zip(events, onsets):
        assert abs(t - o / SR) < 0.001
        assert abs(int(np.argmax(np.abs(clip) > 0.004)) - PRE) <= 10


def test_steady_noise_never_triggers_the_detector():
    from holo.audio import HOP, SR, OnsetDetector
    rng = np.random.default_rng(2)
    t = np.arange(SR * 4) / SR
    x = (0.01 * np.sin(2 * np.pi * 220 * t) + rng.normal(0, 2e-3, len(t))).astype(np.float32)  # hum + hiss
    d = OnsetDetector()
    assert sum(len(d.feed(x[i:i + HOP], i / SR)) for i in range(0, len(x), HOP)) == 0
