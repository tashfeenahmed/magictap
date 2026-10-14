"""Click-target experiment: tap targets across the screen and measure where clicks land.

Nothing outside this window is clicked. Results go to data/sessions/<timestamp>/.
"""

import csv
import json
import random
import time

import cv2
import numpy as np

from .app import App, pt, text, wait_until_ready
from .calibration import Calibration, grid_targets
from .config import SESSIONS_DIR
from .model import TouchModel
from .touch import TouchEngine


def run_experiment(camera=None, cols=4, rows=3, radius=None, seed=None, reps=2):
    calib = Calibration.load()
    if calib.report.get("mode") == "cells":
        return run_cell_experiment(calib, camera, reps, seed)
    model = TouchModel.load()
    if radius is None:  # size targets from the accuracy measured during calibration
        p90 = calib.report.get("final", {}).get("p90", 100)
        radius = int(np.clip(p90, 60, 200))
        print(f"Target radius {radius} pt (from calibration: 90% of touches within {p90:.0f} pt)")
    app = App(camera=calib.camera_index if camera is None else camera, window="holo experiment")
    c = app.canvas
    if (c.width, c.height) != tuple(calib.canvas_size):
        print("Warning: screen size changed since calibration; re-run `holo calibrate`.")
    engine = TouchEngine(model, calib.canvas_size)
    app.show_inset = False  # a camera view on screen pulls the eyes away from the target
    # Offset the grid from the calibration dots so we test positions the model never saw.
    targets = grid_targets(c.width, c.height, cols, rows, margin=0.12)
    random.Random(seed).shuffle(targets)

    out_dir = SESSIONS_DIR / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_f = open(out_dir / "trace.csv", "w", newline="")
    trace = csv.writer(trace_f)
    trace.writerow(["t", "body", "face", "hands", "motion", "p_touch", "state", "x", "y", "gaze_x", "gaze_y", "target"])

    trials, k, shown_at, flash = [], 0, None, None
    try:
        wait_until_ready(app, "Click experiment", [
            "Sit the same way as during calibration.",
            f"{len(targets)} yellow circles will appear one at a time. Look at each one, touch it, hold still briefly.",
            "The blue ring shows where holo thinks you're looking (G hides it).",
            "Can't reach one or it won't register? Press Space to skip it.",
        ])
        while k < len(targets):
            obs = app.step()
            tf = engine.update(obs)
            shown_at = shown_at or obs.t
            x, y = tf.point if tf.point is not None else (None, None)
            gx, gy = tf.gaze if tf.gaze is not None else (None, None)
            trace.writerow([f"{obs.t:.3f}", obs.pose is not None, obs.gaze is not None, len(obs.hands),
                            f"{tf.motion:.3f}", f"{tf.p_touch:.3f}", tf.state, x, y, gx, gy, k])

            target = targets[k]
            for kind, p in tf.events:
                if kind != "down":
                    continue
                err = float(np.linalg.norm(p - target))
                result = "hit" if err <= radius else "near" if err <= 2 * radius else "miss"
                trials.append({"target": target.tolist(), "click": p.tolist(), "error": err, "result": result,
                               "time": obs.t - shown_at, "p_touch": tf.p_touch})
                color = {"hit": (0, 220, 0), "near": (0, 200, 255), "miss": (0, 0, 255)}[result]
                flash = (p.copy(), color, obs.t + 0.6)
                k, shown_at = k + 1, None
                break

            key = _draw(app, tf, targets, k, trials, flash, obs, radius)
            if key == 27:
                break
            if key == ord(" ") and k < len(targets):
                trials.append({"target": targets[k].tolist(), "click": None, "error": None, "result": "skipped",
                               "time": None, "p_touch": None})
                k, shown_at = k + 1, None
            elif key in (ord("i"), ord("I")):
                app.show_inset = not app.show_inset
            elif key in (ord("g"), ord("G")):
                app.show_gaze = not app.show_gaze
    except KeyboardInterrupt:
        trace_f.close()
        app.close()
        return
    finally:
        trace_f.close()

    summary = _summarize(trials, len(targets), radius, calib)
    (out_dir / "summary.json").write_text(json.dumps({"summary": summary, "trials": trials}, indent=2))
    with open(out_dir / "trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target", "click", "error", "result", "time", "p_touch"])
        w.writeheader()
        w.writerows(trials)
    heat = _heatmap(c, trials, summary, radius)
    cv2.imwrite(str(out_dir / "heatmap.png"), heat)
    print(json.dumps(summary, indent=2))
    print(f"Results saved to {out_dir}")
    app.wait_key(heat)
    app.close()


def _draw(app, tf, targets, k, trials, flash, obs, radius):
    c = app.canvas
    img = c.blank(18)
    for t in trials:
        if t["click"] is not None:
            color = {"hit": (0, 140, 0), "near": (0, 140, 180), "miss": (0, 0, 160)}[t["result"]]
            cv2.circle(img, pt(t["target"]), 5, color, -1)
    if k < len(targets):
        cv2.circle(img, pt(targets[k]), radius, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.circle(img, pt(targets[k]), 6, (0, 220, 255), -1, cv2.LINE_AA)
    if tf.gaze is not None and app.show_gaze:  # where holo thinks you're looking
        cv2.circle(img, pt(tf.gaze), 22, (255, 160, 60), 2, cv2.LINE_AA)
    if tf.state == "touch" and tf.point is not None:
        cv2.drawMarker(img, pt(tf.point), (0, 0, 255), cv2.MARKER_CROSS, 34, 3, cv2.LINE_AA)
    if flash and obs.t < flash[2]:
        cv2.circle(img, pt(flash[0]), 16, flash[1], -1, cv2.LINE_AA)
    hits = sum(t["result"] == "hit" for t in trials)
    body = "body ok" if obs.pose is not None else "NO BODY"
    status = (f"target {min(k + 1, len(targets))}/{len(targets)}   hits {hits}   {body}   "
              f"touch prob {tf.p_touch:.2f}   motion {tf.motion:.2f}   state {tf.state}   fps {app.fps:.0f}")
    text(img, status, (c.width / 2, 24), 0.55, center=True)
    # p_touch meter
    cv2.rectangle(img, (c.width // 2 - 100, 40), (c.width // 2 + 100, 48), (60, 60, 60), -1)
    cv2.rectangle(img, (c.width // 2 - 100, 40), (int(c.width / 2 - 100 + 200 * tf.p_touch), 48),
                  (0, 0, 255) if tf.state == "touch" else (0, 200, 255), -1)
    text(img, "Look at the yellow circle, touch it, hold still briefly.  [Space] skip  [G] gaze cursor  [I] camera  [Esc] finish",
         (c.width / 2, c.height - 20), 0.5, (150, 150, 150), 1, center=True)
    app.inset(img, avoid=targets[k] if k < len(targets) else None)
    return app.show(img)


def _summarize(trials, n_targets, radius, calib):
    clicked = [t for t in trials if t["click"] is not None]
    errs = [t["error"] for t in clicked]
    s = {
        "targets": n_targets,
        "attempted": len(trials),
        "clicked": len(clicked),
        "skipped": sum(t["result"] == "skipped" for t in trials),
        "hits": sum(t["result"] == "hit" for t in clicked),
        "near": sum(t["result"] == "near" for t in clicked),
        "misses": sum(t["result"] == "miss" for t in clicked),
        "target_radius_pt": radius,
        "gaze_model": calib.report.get("gaze_model"),
        "body_model": calib.report.get("body_model"),
    }
    if errs:
        s.update(hit_rate=round(s["hits"] / len(clicked), 3),
                 median_error_pt=round(float(np.median(errs)), 1),
                 p90_error_pt=round(float(np.percentile(errs, 90)), 1),
                 median_time_s=round(float(np.median([t["time"] for t in clicked])), 2),
                 # a button this wide would catch 90% of clicks aimed at its centre
                 reliable_button_size_pt=round(2 * float(np.percentile(errs, 90))))
    return s


def _heatmap(c, trials, summary, radius):
    img = c.blank(18)
    for t in trials:
        tgt = t["target"]
        if t["click"] is None:
            cv2.circle(img, pt(tgt), radius, (110, 110, 110), 2)
            text(img, "skip", tgt, 0.45, (150, 150, 150), 1, center=True)
            continue
        e = t["error"]
        color = (0, 200, 0) if e <= radius else (0, 200, 255) if e <= 2 * radius else (0, 0, 255)
        cv2.circle(img, pt(tgt), radius, color, 2, cv2.LINE_AA)
        cv2.line(img, pt(tgt), pt(t["click"]), color, 1, cv2.LINE_AA)
        cv2.circle(img, pt(t["click"]), 5, color, -1)
        text(img, f"{e:.0f}", (tgt[0] + radius + 4, tgt[1]), 0.45, color, 1)
    lines = [f"hits {summary['hits']}/{summary['clicked']} clicked  "
             f"(near {summary['near']}, miss {summary['misses']}, skipped {summary['skipped']})"]
    if "median_error_pt" in summary:
        lines.append(f"median error {summary['median_error_pt']} pt, 90% within {summary['p90_error_pt']} pt  ->  "
                     f"buttons ~{summary['reliable_button_size_pt']:.0f} pt wide would work reliably")
    lines.append("green = hit, orange = near, red = miss, gray = skipped.  Press any key.")
    for n, line in enumerate(lines):
        text(img, line, (c.width / 2, c.height / 2 - 20 + 30 * n), 0.6, center=True)
    return img


# --------------------------------------------------------------------------
# Cell mode: touch highlighted cells; count right / wrong / skipped-by-holo.
# --------------------------------------------------------------------------

def run_cell_experiment(calib, camera=None, reps=2, seed=None):
    from .calibration import draw_grid
    from .model import CellModel
    from .touch import CellTouchEngine

    model = CellModel.load()
    app = App(camera=calib.camera_index if camera is None else camera, window="holo experiment")
    c = app.canvas
    engine = CellTouchEngine(model)
    app.show_inset = False  # a camera view on screen pulls the eyes away from the target
    n = model.cols * model.rows
    order = [k for _ in range(reps) for k in range(n)]
    random.Random(seed).shuffle(order)
    out_dir = SESSIONS_DIR / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    trials, k, shown_at, flash = [], 0, None, None
    rejects = 0
    try:
        wait_until_ready(app, "Cell experiment", [
            f"The screen is a {model.cols} x {model.rows} grid. One cell at a time lights up yellow.",
            "Look at it and touch it anywhere inside (centre is best), then lift your hand.",
            "Green flash = right cell, red = wrong cell, grey = holo wasn't sure and didn't click (try again).",
            "Can't get one? Press Space to skip it.",
        ])
        while k < len(order):
            obs = app.step()
            cf = engine.update(obs)
            shown_at = shown_at or obs.t
            target = order[k]
            for kind, cell, conf in cf.events:
                if kind == "reject":
                    rejects += 1
                    trials.append({"target": target, "cell": cell, "confidence": conf, "result": "not sure",
                                   "time": obs.t - shown_at})
                    flash = (cell, (90, 90, 90), obs.t + 0.6)
                elif kind == "down":
                    ok = cell == target
                    trials.append({"target": target, "cell": cell, "confidence": conf,
                                   "result": "right" if ok else "wrong", "time": obs.t - shown_at})
                    flash = (cell, (0, 200, 0) if ok else (0, 0, 220), obs.t + 0.6)
                    k, shown_at = k + 1, None
                    break

            img = c.blank(18)
            draw_grid(img, (model.cols, model.rows), highlight=order[k] if k < len(order) else None,
                      fill=(0, 150, 200))
            if cf.state == "hover" and cf.cell is not None:  # where holo thinks you're heading
                r, col = divmod(cf.cell, model.cols)
                cw, ch = c.width / model.cols, c.height / model.rows
                cv2.rectangle(img, (int(col * cw) + 6, int(r * ch) + 6), (int((col + 1) * cw) - 6, int((r + 1) * ch) - 6),
                              (255, 160, 60), 3)
            if flash and obs.t < flash[2] and flash[0] is not None:
                r, col = divmod(flash[0], model.cols)
                cw, ch = c.width / model.cols, c.height / model.rows
                overlay = img.copy()
                cv2.rectangle(overlay, (int(col * cw), int(r * ch)), (int((col + 1) * cw), int((r + 1) * ch)), flash[1], -1)
                img = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
            right = sum(t["result"] == "right" for t in trials)
            wrong = sum(t["result"] == "wrong" for t in trials)
            text(img, f"target {min(k + 1, len(order))}/{len(order)}   right {right}   wrong {wrong}   not sure {rejects}"
                      f"   touch prob {cf.p_touch:.2f}   fps {app.fps:.0f}", (c.width / 2, 24), 0.55, center=True)
            text(img, "Touch the yellow cell.  Blue outline = where holo thinks you're heading.  [Space] skip  "
                      "[I] camera  [Esc] finish", (c.width / 2, c.height - 20), 0.5, (170, 170, 170), 1, center=True)
            app.inset(img, avoid=model.centre(order[k]) if k < len(order) else None)
            key = app.show(img)
            if key == 27:
                break
            if key == ord(" ") and k < len(order):
                trials.append({"target": order[k], "cell": None, "confidence": None, "result": "skipped", "time": None})
                k, shown_at = k + 1, None
            elif key in (ord("i"), ord("I")):
                app.show_inset = not app.show_inset
    except KeyboardInterrupt:
        app.close()
        return

    clicks = [t for t in trials if t["result"] in ("right", "wrong")]
    per_cell = {}
    for t in clicks:
        per_cell.setdefault(t["target"], []).append(t["result"] == "right")
    summary = {
        "grid": [model.cols, model.rows], "targets": len(order),
        "right": sum(t["result"] == "right" for t in trials), "wrong": sum(t["result"] == "wrong" for t in trials),
        "not_sure_retries": rejects, "skipped": sum(t["result"] == "skipped" for t in trials),
        "accuracy_of_clicks": round(float(np.mean([t["result"] == "right" for t in clicks])), 3) if clicks else None,
        "per_cell_accuracy": {int(k): round(float(np.mean(v)), 2) for k, v in sorted(per_cell.items())},
        "median_time_s": round(float(np.median([t["time"] for t in clicks])), 2) if clicks else None,
    }
    (out_dir / "summary.json").write_text(json.dumps({"summary": summary, "trials": trials}, indent=2))
    img = c.blank(18)
    cw, ch = c.width / model.cols, c.height / model.rows
    for cell in range(n):
        r, col = divmod(cell, model.cols)
        acc = summary["per_cell_accuracy"].get(cell)
        colour = (60, 60, 60) if acc is None else (0, 120, 0) if acc >= 0.95 else (0, 120, 160) if acc >= 0.5 else (0, 0, 140)
        cv2.rectangle(img, (int(col * cw) + 4, int(r * ch) + 4), (int((col + 1) * cw) - 4, int((r + 1) * ch) - 4), colour, -1)
        text(img, "-" if acc is None else f"{100 * acc:.0f}%", ((col + 0.5) * cw, (r + 0.5) * ch), 0.9, center=True)
    acc = summary["accuracy_of_clicks"]
    text(img, f"Clicks landed in the right cell {'-' if acc is None else f'{100 * acc:.0f}%'} of the time "
              f"({summary['right']} right, {summary['wrong']} wrong, {rejects} 'not sure' retries, "
              f"{summary['skipped']} skipped). Press any key.", (c.width / 2, 30), 0.6, center=True)
    cv2.imwrite(str(out_dir / "cells.png"), img)
    print(json.dumps(summary, indent=2))
    print(f"Results saved to {out_dir}")
    app.wait_key(img)
    app.close()
