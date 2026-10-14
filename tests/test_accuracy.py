"""Accuracy pieces: re-centre fitting, fixation selection, confidence gating, pooled gaze mapper."""

import numpy as np

from magictap.fusion import MIN_CONFIDENCE, select_fixation, vote
from magictap.recenter import apply_correction, cell_centres, fit_correction


def test_affine_correction_recovers_a_known_drift():
    rng = np.random.default_rng(0)
    targets = np.array(cell_centres(1728, 1085))
    A = np.array([[0.9, 0.05], [-0.03, 1.1]])
    drifted = targets @ A.T + np.array([60.0, -40.0]) + rng.normal(0, 2, targets.shape)
    M, kind = fit_correction(drifted, targets)
    assert kind == "affine"
    assert np.abs(apply_correction(M, drifted) - targets).max() < 8
    assert np.allclose(apply_correction(M, drifted[0]), apply_correction(M, drifted[:1])[0])


def test_few_points_fall_back_to_a_median_shift():
    M, kind = fit_correction([(100, 100), (300, 120)], [(150, 80), (350, 100)])
    assert kind == "shift"
    assert np.allclose(apply_correction(M, (0, 0)), (50, -20))


def test_no_correction_is_identity():
    assert np.allclose(apply_correction(None, (12.0, 34.0)), (12.0, 34.0))


def test_fixation_skips_the_saccade_from_the_previous_button():
    t0 = 10.0
    items = [(t0 - 0.8 + k / 30, (200.0, 200.0)) for k in range(10)]               # previous button
    items += [(t0 - 0.45 + k / 30, (200.0 + 120 * k, 200.0 + 60 * k)) for k in range(5)]  # saccade
    items += [(t0 - 0.25 + k / 30, (900.0 + (k % 2) * 5, 500.0)) for k in range(9)]  # settled on the new one
    pts = select_fixation(items, t0)
    assert len(pts) >= 4 and all(abs(p[0] - 900) < 20 for p in pts)


def test_without_a_fixation_it_falls_back_to_the_fixed_window():
    t0 = 5.0
    items = [(t0 - 0.6 + k / 30, (100.0 * k, 0.0)) for k in range(20)]  # constantly moving
    pts = select_fixation(items, t0)
    assert all(t0 - 0.30 <= t <= t0 + 0.05 for t, xy in items if xy in pts)


def test_vote_and_confidence_gate():
    assert vote([]) == (None, 0.0)
    cell, conf = vote([2, 2, 2, 5])
    assert cell == 2 and conf == 0.75 and conf >= MIN_CONFIDENCE
    assert vote([1, 2, 3, 1, 2])[1] < MIN_CONFIDENCE


def _session(tmp_path, name, rng, drift):
    from magictap.gaze import N_GEO_FEATURES, N_PATCH_FEATURES
    from tests.test_core import synthetic_gaze
    from magictap.calibration import grid_targets

    tr = synthetic_gaze(grid_targets(1728, 1085, 5, 4), 12, rng)
    ck = synthetic_gaze(grid_targets(1728, 1085, 3, 3, 0.2), 12, rng, group_offset=1000)
    for X in (tr[0], ck[0]):
        X[:, 11] += drift  # this session's gaze reads a bit off
    d = tmp_path / name
    d.mkdir()
    np.savez(d / "dataset.npz", gaze_train_X=tr[0], gaze_train_Y=tr[1], gaze_train_groups=tr[2],
             gaze_check_X=ck[0], gaze_check_Y=ck[1], gaze_check_groups=ck[2])


def test_pooled_mapper_uses_every_session_and_scores_held_out_sessions(tmp_path):
    from magictap.model import fit_gaze_mapper_pooled, load_gaze_sessions

    rng = np.random.default_rng(3)
    for k, drift in enumerate((0.0, 0.02, -0.02)):
        _session(tmp_path, f"run{k}", rng, drift)
    sessions = load_gaze_sessions(tmp_path)
    assert len(sessions) == 3
    mapper, name, score, cands = fit_gaze_mapper_pooled(sessions, (1728, 1085))
    assert score["sessions"] == 3 and score["median"] < 120, score
    assert set(cands) >= {"gaze ray + eye images", "all gaze features"}
    assert fit_gaze_mapper_pooled(sessions[:1], (1728, 1085)) is None


def test_demo_evaluation_counts_right_wrong_and_unsure():
    from magictap.evaluate import evaluate
    from magictap.model import CellModel

    grid = CellModel(3, 2, (1728, 1085))
    cells = [0, 1, 2, 3, 4, 5, 0, 1, 2]
    pts = [[tuple(grid.centre(c) + 40)] * 6 for c in cells]  # consistently 40 pt off -> re-centre fixes it
    rows = evaluate(pts, pts, cells, grid, recenter_idx=list(range(6)))
    best = [r for r in rows if r[0] == "fixation, re-centre (affine)"][0]
    assert best[1:] == (3, 0, 0, 3)
