"""`magictap evaluate-demo`: offline accuracy on a recorded demo session.

A demo recording (`magictap record-demo`) has the camera video and, for each press,
the cell that was lit and when the press happened. This replays the eyes through the
current gaze mapping and reports, per press, which cell the gaze picks:

  - without a re-centre,
  - with a re-centre (shift or affine) fitted on the FIRST press of each cell (6 presses,
    standing in for `magictap recenter`) and scored on the remaining presses,
  - for a fixed window around the press and for the fixation-based selection,
  - and with the confidence gate (click only if >= 60% of the gaze samples agree).
"""

import json

import numpy as np

from .config import DATA_DIR
from .fusion import MIN_CONFIDENCE, TAP_WINDOW, select_fixation, vote
from .recenter import apply_correction, fit_correction

DEMO_DIR = DATA_DIR / "demo_runs"


def _gaze_features(run):
    path = run / "gaze_feats.npz"
    if path.exists():
        g = np.load(path)
        return g["t"], g["F"], g["ok"]
    import cv2

    from .gaze import GazeEstimator, N_GEO_FEATURES, N_PATCH_FEATURES

    print("Extracting gaze from the recording (one-off, ~30 s)...")
    d = json.loads((run / "demo.json").read_text())
    est, cap = GazeEstimator(), cv2.VideoCapture(str(run / "camera.mp4"))
    F, ok = [], []
    for t in d["frame_times"]:
        r, f = cap.read()
        if not r:
            break
        o = est.process(f, t)
        good = o is not None and not o.blink
        F.append(o.features if good else np.full(N_GEO_FEATURES + N_PATCH_FEATURES, np.nan, np.float32))
        ok.append(good)
    t = np.array(d["frame_times"][:len(F)])
    np.savez_compressed(path, t=t, F=np.array(F, np.float32), ok=np.array(ok))
    return t, np.array(F, np.float32), np.array(ok)


def press_times(demo):
    ev, ft = demo["events"], demo["frame_times"]
    out = []
    for k, e in enumerate(ev):
        if "t_press" in e:
            out.append(e["t_press"])
        elif k + 1 < len(ev):
            out.append(ev[k + 1]["t_prompt"] - 0.45)  # older recordings: Space, then a 0.45 s flash
        else:
            out.append(ft[-1] - 0.45)
    return out


def evaluate(points_fixed, points_fix, cells, model, recenter_idx):
    """Rows of (label, right, wrong, unsure, n) for every combination."""
    rows = []
    test = [k for k in range(len(cells)) if k not in recenter_idx]
    for wname, pts in (("fixed window", points_fixed), ("fixation", points_fix)):
        def run(idx, M, gate):
            r = w = u = 0
            for k in idx:
                P = apply_correction(M, np.array(pts[k])) if M is not None else np.array(pts[k])
                cell, conf = vote([model.cell_of(p) for p in P])
                if cell is None or conf < gate:
                    u += 1
                elif cell == cells[k]:
                    r += 1
                else:
                    w += 1
            return r, w, u, len(idx)

        rows.append((f"{wname}, no re-centre", *run(range(len(cells)), None, 0.0)))
        X = [np.median(np.array(pts[k]), axis=0) for k in recenter_idx]
        Y = [model.centre(cells[k]) for k in recenter_idx]
        for kind in ("shift", "affine"):
            if kind == "affine":
                M = fit_correction(X, Y)[0]
            else:
                shift = np.median(np.array(Y) - np.array(X), axis=0)
                M = np.array([[1, 0, shift[0]], [0, 1, shift[1]]], float)
            rows.append((f"{wname}, re-centre ({kind})", *run(test, M, 0.0)))
            rows.append((f"{wname}, re-centre ({kind}) + confidence >= {MIN_CONFIDENCE:.0%}", *run(test, M, MIN_CONFIDENCE)))
    return rows


def run_evaluate_demo(run=None):
    from pathlib import Path

    from .model import CellModel

    runs = sorted(p for p in DEMO_DIR.glob("*") if (p / "demo.json").exists()) if DEMO_DIR.exists() else []
    run = Path(run) if run else (runs[-1] if runs else None)
    if run is None or not (run / "demo.json").exists():
        raise SystemExit("No demo recording found. Record one with `magictap record-demo`.")
    demo = json.loads((run / "demo.json").read_text())
    t, F, ok = _gaze_features(run)
    model = CellModel.load()
    cols, rows_ = demo.get("grid", [model.cols, model.rows])
    canvas = tuple(demo.get("canvas", [model.w, model.h]))
    grid = CellModel(cols, rows_, canvas)  # the demo's own grid, whatever grid calibration picked
    mapped = np.full((len(t), 2), np.nan)
    mapped[ok] = model.gaze.predict(F[ok]) + model.offset
    items = [(tt, tuple(p) if o else None) for tt, p, o in zip(t, mapped, ok)]
    presses = press_times(demo)
    cells = [e["cell"] for e in demo["events"]]
    fixed = [[xy for tt, xy in items if xy is not None and tp + TAP_WINDOW[0] <= tt <= tp + TAP_WINDOW[1]] for tp in presses]
    fix = [select_fixation(items, tp) for tp in presses]
    recenter_idx = [cells.index(c) for c in sorted(set(cells))]  # first press of each cell
    rows = evaluate(fixed, fix, cells, grid, recenter_idx)
    from .calibration import Calibration

    eyes = Calibration.load().report.get("eyes", {}).get("model", "?")
    print(f"\nDemo run {run.name}: {len(cells)} presses on a {cols}x{rows_} grid; gaze mapping: {eyes}")
    print(f"Re-centre rows: fitted on the first press of each cell ({len(recenter_idx)}), scored on the other "
          f"{len(cells) - len(recenter_idx)}.\n")
    print(f"  {'setting':58s} {'right':>7s} {'wrong':>6s} {'not sure':>9s}   accuracy when it clicks")
    for label, r, w, u, n in rows:
        acc = f"{100 * r / (r + w):.0f}%" if r + w else "-"
        print(f"  {label:58s} {r:3d}/{n:<3d} {w:6d} {u:9d}   {acc}")
    return rows
