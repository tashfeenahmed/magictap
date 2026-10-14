"""Per-user models, trained during calibration.

WHERE  - gaze mapper: gaze features -> screen point (main signal; people look at
         what they touch). Trained on "look at the dot" data, validated on
         separate dots it never saw.
       - body regressor: body features -> screen point (weak, used as a tie-breaker).
       - fusion: inverse-variance weighting using each one's measured error.
WHEN   - classifier: body posture "touching the screen" vs. not, plus motion
         thresholds for "the arm has stopped".
"""

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .body import N_KP_FEATURES
from .config import CELL_MODEL_PATH, MODEL_PATH
from .gaze import N_GEO_FEATURES

KP = slice(0, N_KP_FEATURES)
MASK = slice(N_KP_FEATURES, None)
GEO = slice(0, N_GEO_FEATURES)
EYES = slice(N_GEO_FEATURES, None)
ALPHAS = np.logspace(-2, 4, 14)


def _pose_only():
    return ColumnTransformer([("kp", StandardScaler(), KP)])


def _pose_and_silhouette():
    return ColumnTransformer([("kp", StandardScaler(), KP),
                              ("mask", make_pipeline(PCA(20, random_state=0), StandardScaler()), MASK)])


BODY_CANDIDATES = {
    "pose ridge": lambda: make_pipeline(_pose_only(), RidgeCV(ALPHAS)),
    "pose+silhouette ridge": lambda: make_pipeline(_pose_and_silhouette(), RidgeCV(ALPHAS)),
    "pose+silhouette kNN": lambda: make_pipeline(_pose_and_silhouette(), KNeighborsRegressor(7, weights="distance")),
    "pose+silhouette trees": lambda: make_pipeline(
        _pose_and_silhouette(), ExtraTreesRegressor(150, min_samples_leaf=3, n_jobs=-1, random_state=0)),
}

# Gaze feature columns (see gaze.py): direction vector, pitch/yaw, face centre,
# head rotation, gaze-ray hit on the camera plane, iris ratios.
G_VEC, G_ANG, G_POS, G_ROT, G_HIT, G_IRIS = [0, 1, 2], [3, 4], [5, 6, 7], [8, 9, 10], [11, 12], [13, 14, 15, 16]


def _hit_quad(extra=(), eyes=False):
    """Quadratic in the gaze-ray hit point (head-position aware by construction) + linear extras."""
    parts = [("hit", make_pipeline(StandardScaler(), PolynomialFeatures(2), StandardScaler()), G_HIT)]
    if extra:
        parts.append(("x", StandardScaler(), list(extra)))
    if eyes:
        parts.append(("eyes", make_pipeline(PCA(20, random_state=0), StandardScaler()), EYES))
    return make_pipeline(ColumnTransformer(parts), RidgeCV(ALPHAS))


GAZE_CANDIDATES = {
    "all gaze features": lambda: make_pipeline(ColumnTransformer([("g", StandardScaler(), GEO)]), RidgeCV(ALPHAS)),
    "gaze ray": lambda: _hit_quad(G_VEC + G_ANG + G_IRIS),
    "gaze ray + eye images": lambda: _hit_quad(G_VEC + G_ANG, eyes=True),
    "gaze ray + iris + eye images": lambda: _hit_quad(G_VEC + G_ANG + G_IRIS, eyes=True),
}


def per_point(model, X, Y, groups):
    """Median prediction per target (robust to blinks and saccades)."""
    preds = model.predict(X)
    return {int(g): np.median(preds[groups == g], axis=0) for g in np.unique(groups)}, \
           {int(g): Y[groups == g][0] for g in np.unique(groups)}


def leave_one_target_out(make, X, Y, groups):
    """Prediction for each target from a model trained without that target."""
    preds = {}
    for g in np.unique(groups):
        test = groups == g
        model = make().fit(X[~test], Y[~test])
        preds[int(g)] = np.median(model.predict(X[test]), axis=0)
    return preds


def score(preds, truth, canvas_size):
    """Errors and coarse-region accuracy, plus the 'always guess the centre' baseline."""
    w, h = canvas_size
    errs = {g: float(np.linalg.norm(preds[g] - truth[g])) for g in preds}
    centre = np.array([w / 2, h / 2])
    baseline = float(np.median([np.linalg.norm(centre - t) for t in truth.values()]))
    third = lambda v, size: int(np.clip(v // (size / 3), 0, 2))
    col = [third(preds[g][0], w) == third(truth[g][0], w) for g in preds]
    row = [third(preds[g][1], h) == third(truth[g][1], h) for g in preds]
    return {"errors": errs, "median": float(np.median(list(errs.values()))), "baseline": baseline,
            "p90": float(np.percentile(list(errs.values()), 90)),
            "col": float(np.mean(col)), "row": float(np.mean(row)),
            "region": float(np.mean([a and b for a, b in zip(col, row)])),
            "pred": {g: np.asarray(p).tolist() for g, p in preds.items()},
            "truth": {g: np.asarray(t).tolist() for g, t in truth.items()}}


def verdict(median, baseline, p90):
    ratio = median / baseline
    if ratio > 0.8:
        return "no better than guessing - position isn't being picked up yet"
    if ratio > 0.5:
        return "rough sense of direction only (e.g. left vs right)"
    size = 2 * p90
    if size < 250:
        return f"good: buttons about {size:.0f} pt wide should work reliably"
    if size < 600:
        return f"coarse: buttons about {size:.0f} pt wide (a 3x3 grid or so)"
    return "clearly better than guessing, but only for very large areas"


def _fuse(pg, pb, wg, wb):
    if pb is None:
        return pg
    if pg is None:
        return pb
    return (wg * np.asarray(pg) + wb * np.asarray(pb)) / (wg + wb)


class TouchModel:
    def __init__(self):
        self.gaze = self.body = self.clf = None
        self.w_gaze, self.w_body = 1.0, 0.0
        self.report = {}
        self.still_thr = self.release_thr = 0.0

    def fit(self, gaze_train, gaze_val, touch_gaze, body_touch, rest_X, hold_motion, reach_motion,
            canvas_size, progress=lambda s: None):
        """gaze_*: (X, Y, groups). body_touch: (X, Y, groups) body features while touching."""
        Xg, Yg, gg = gaze_train
        Xv, Yv, gv = gaze_val
        Xt, Yt, gt = touch_gaze
        Xb, Yb, gb = body_touch
        r = {}

        # 1. Gaze. Two honest tests per candidate:
        #    - check dots: trained on the look dots (+ touch data), tested on dots it never saw
        #    - touch: each touch dot predicted by a model trained without that dot
        #    The candidate with the best average of the two wins, so it must work while touching too.
        w, h = canvas_size
        clip = lambda P: np.clip(P, 0, (w, h))
        gaze_scores, touch_scores, touch_preds = {}, {}, {}
        for name, make in GAZE_CANDIDATES.items():
            progress(f"Gaze: trying {name}")
            m = make().fit(np.vstack([Xg] + ([Xt] if len(Xt) else [])), np.vstack([Yg] + ([Yt] if len(Xt) else [])))
            preds, truth = per_point(m, Xv, Yv, gv)
            gaze_scores[name] = score({g: clip(p) for g, p in preds.items()}, truth, canvas_size)
            if len(np.unique(gt)) >= 3:
                tp = {}
                for g in np.unique(gt):
                    k = gt != g
                    mm = make().fit(np.vstack([Xg, Xv, Xt[k]]), np.vstack([Yg, Yv, Yt[k]]))
                    tp[int(g)] = clip(np.median(mm.predict(Xt[~k]), axis=0))
                touch_preds[name] = tp
                touch_scores[name] = score(tp, {int(g): Yt[gt == g][0] for g in np.unique(gt)}, canvas_size)
        rank = lambda n: gaze_scores[n]["median"] + (touch_scores[n]["median"] if n in touch_scores else 0)
        best_g = min(GAZE_CANDIDATES, key=rank)
        r["gaze_model"] = best_g
        r["gaze_validation"] = gaze_scores[best_g]
        r["gaze_candidates_median_pt"] = {
            n: {"check": round(gaze_scores[n]["median"], 1),
                "touch": round(touch_scores[n]["median"], 1) if n in touch_scores else None} for n in GAZE_CANDIDATES}
        gaze_at_touch = None
        if best_g in touch_scores:
            r["gaze_at_touch"] = touch_scores[best_g]
            gaze_at_touch = (touch_preds[best_g], {int(g): Yt[gt == g][0] for g in np.unique(gt)})

        # 2. Body: leave-one-dot-out on the touch dots.
        body_preds = None
        if len(np.unique(gb)) >= 6:
            body_scores, body_all = {}, {}
            for name, make in BODY_CANDIDATES.items():
                progress(f"Body: trying {name}")
                body_all[name] = leave_one_target_out(make, Xb, Yb, gb)
                truth = {int(g): Yb[gb == g][0] for g in np.unique(gb)}
                body_scores[name] = score(body_all[name], truth, canvas_size)
            best_b = min(body_scores, key=lambda n: body_scores[n]["median"])
            r["body_model"], r["body"] = best_b, body_scores[best_b]
            body_preds = body_all[best_b]
            self.body = BODY_CANDIDATES[best_b]().fit(Xb, Yb)

        # 3. Fusion weights from measured errors (inverse variance).
        eg = r.get("gaze_at_touch", r["gaze_validation"])["median"]
        self.w_gaze = 1 / max(eg, 1) ** 2
        self.w_body = 1 / max(r["body"]["median"], 1) ** 2 if body_preds else 0.0
        if gaze_at_touch and body_preds:
            preds, truth = gaze_at_touch
            # touch-gaze groups are 2000 + dot index, body groups are the dot index
            fused = {g: _fuse(preds[g], body_preds.get(g - 2000), self.w_gaze, self.w_body) for g in preds}
            r["fused_at_touch"] = score(fused, truth, canvas_size)
        final = r.get("fused_at_touch", r.get("gaze_at_touch", r["gaze_validation"]))
        r["final"] = final
        r["verdict"] = verdict(final["median"], final["baseline"], final["p90"])

        # Final gaze model uses every gaze sample we have.
        progress("Training final models")
        self.gaze = GAZE_CANDIDATES[best_g]().fit(np.vstack([Xg, Xv] + ([Xt] if len(Xt) else [])),
                                                 np.vstack([Yg, Yv] + ([Yt] if len(Xt) else [])))

        # 4. When: touching posture vs. not, and motion thresholds.
        if len(Xb) and len(rest_X) >= 5:
            X_cls = np.vstack([Xb, rest_X])
            y_cls = np.r_[np.ones(len(Xb)), np.zeros(len(rest_X))]
            make_clf = lambda: make_pipeline(_pose_and_silhouette(), LogisticRegression(C=0.3, max_iter=3000))
            r["touch_vs_rest_accuracy"] = float(np.mean(cross_val_score(make_clf(), X_cls, y_cls, cv=5)))
            self.clf = make_clf().fit(X_cls, y_cls)
        hold, reach = np.asarray(hold_motion), np.asarray(reach_motion)
        self.still_thr = float(np.percentile(hold, 85)) if len(hold) else 0.05
        self.release_thr = max(self.still_thr * 2.5, float(np.percentile(reach, 50)) if len(reach) else 0.0)
        r["still_threshold"], r["release_threshold"] = self.still_thr, self.release_thr
        r["fusion_weights"] = {"gaze": self.w_gaze, "body": self.w_body}
        self.report = r
        return self

    def predict_gaze(self, gaze_features):
        return self.gaze.predict(gaze_features[None])[0]

    def predict_body(self, body_features):
        """(body position estimate or None, probability the user is touching)."""
        x = body_features[None]
        xy = self.body.predict(x)[0] if self.body is not None else None
        p = float(self.clf.predict_proba(x)[0, 1]) if self.clf is not None else 0.0
        return xy, p

    def fuse(self, gaze_xy, body_xy):
        return _fuse(gaze_xy, body_xy, self.w_gaze, self.w_body)

    def save(self, path=MODEL_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path=MODEL_PATH) -> "TouchModel":
        if not path.exists():
            raise SystemExit("No trained model. Run `magictap calibrate` first.")
        from . import legacy_unpickle_aliases

        legacy_unpickle_aliases()
        return joblib.load(path)


# --------------------------------------------------------------------------
# Cell mode: one classifier on eyes + body at the moment of the touch.
# --------------------------------------------------------------------------

from sklearn.ensemble import ExtraTreesClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402

from .body import N_MASK_FEATURES  # noqa: E402
from .gaze import N_PATCH_FEATURES  # noqa: E402

N_GAZE = N_GEO_FEATURES + N_PATCH_FEATURES
J_GEO = list(range(N_GEO_FEATURES))
J_EYES = slice(N_GEO_FEATURES, N_GAZE)
J_KP = list(range(N_GAZE, N_GAZE + N_KP_FEATURES))
J_MASK = slice(N_GAZE + N_KP_FEATURES, N_GAZE + N_KP_FEATURES + N_MASK_FEATURES)


def joint_features(gaze_features, body_features):
    """Eyes + body in one vector; a missing part becomes NaN (imputed by the model)."""
    g = gaze_features if gaze_features is not None else np.full(N_GAZE, np.nan, np.float32)
    b = body_features if body_features is not None else np.full(N_KP_FEATURES + N_MASK_FEATURES, np.nan, np.float32)
    return np.r_[g, b].astype(np.float32)


def _joint_transform():
    pca = lambda: make_pipeline(PCA(15, random_state=0), StandardScaler())
    return ColumnTransformer([("geo", StandardScaler(), J_GEO), ("eyes", pca(), J_EYES),
                              ("kp", StandardScaler(), J_KP), ("mask", pca(), J_MASK)])


ARM_KP = list(range(N_GAZE + len(range(0, 60)), N_GAZE + N_KP_FEATURES))  # arm/hand positions relative to shoulders


def _stage2(gaze_xy_cols, kp_cols):
    """Small classifier input: where the eyes say (screen pt) + arm posture."""
    return ColumnTransformer([("gaze", StandardScaler(), gaze_xy_cols), ("arm", StandardScaler(), kp_cols)])


class CellModel:
    """Which grid cell is being touched, with a confidence; plus touch timing (WHEN).

    Stage 1: a gaze mapper (eye features -> screen point) trained on the dense
             look-only calibration, so it needs no touch data.
    Stage 2: a small classifier on [gaze point, arm posture] trained on the touch
             rounds - or just "which cell is the gaze point in", whichever is
             more reliable on rounds it never saw.
    """

    def __init__(self, cols, rows, canvas_size):
        self.cols, self.rows = cols, rows
        self.w, self.h = canvas_size
        self.gaze = None
        self.offset = np.zeros(2)
        self.method = None
        self.clf = self.when = None
        self.threshold = 0.0
        self.report = {}
        self.still_thr = self.release_thr = 0.0

    def cell_of(self, xy):
        c = int(np.clip(xy[0] // (self.w / self.cols), 0, self.cols - 1))
        r = int(np.clip(xy[1] // (self.h / self.rows), 0, self.rows - 1))
        return r * self.cols + c

    def centre(self, cell):
        r, c = divmod(int(cell), self.cols)
        return np.array([(c + 0.5) * self.w / self.cols, (r + 0.5) * self.h / self.rows])

    def _gaze_xy(self, X):
        """Gaze point per frame (NaN where the eyes weren't tracked)."""
        out = np.full((len(X), 2), np.nan)
        ok = ~np.isnan(X[:, 0])
        if ok.any():
            out[ok] = np.clip(self.gaze.predict(X[ok, :N_GAZE]), 0, (self.w, self.h))
        return out

    def _stage2_X(self, X):
        return np.hstack([self._gaze_xy(X), X[:, ARM_KP]])

    def inner(self, xy, margin=0.2):
        """True if xy is clearly inside its cell (not within `margin` of a cell edge)."""
        fx = (xy[0] / (self.w / self.cols)) % 1
        fy = (xy[1] / (self.h / self.rows)) % 1
        return margin < fx < 1 - margin and margin < fy < 1 - margin

    def _methods(self):
        n_arm = len(ARM_KP)
        cols_g, cols_k = [0, 1], list(range(2, 2 + n_arm))
        return {
            "eyes only": None,  # cell containing the median gaze point
            "eyes + touch shift": "offset",  # same, after removing the consistent gaze shift seen while touching
            "eyes + arm (logistic)": lambda: make_pipeline(SimpleImputer(), _stage2(cols_g, cols_k),
                                                           LogisticRegression(C=0.3, max_iter=5000)),
            "eyes + arm (trees)": lambda: make_pipeline(SimpleImputer(), _stage2(cols_g, cols_k),
                                                        ExtraTreesClassifier(300, min_samples_leaf=3, random_state=0,
                                                                             n_jobs=-1)),
        }

    def fit(self, gaze_mapper, X, targets, captures, rounds, body_rest, hold_motion, reach_motion,
            progress=lambda s: None):
        """X: joint features per touch frame; targets: the dot (screen pt) per frame; captures/rounds: ids."""
        self.gaze = gaze_mapper
        self.offset = np.zeros(2)
        cells = np.array([self.cell_of(t) for t in targets])
        S = self._stage2_X(X)
        valid = np.array([self.inner(t) for t in targets])  # judge only on dots clearly inside a cell
        r = {"grid": [self.cols, self.rows]}
        results = {}
        for name, make in self._methods().items():
            progress(f"Cells: testing {name} (hiding one round at a time)")
            results[name] = self._leave_one_round_out(make, S, cells, targets, captures, rounds, valid)
        best = max(results, key=lambda n: (results[n]["accuracy"], n.startswith("eyes ")))
        self.method = best
        res = results[best]
        r.update(model=best, candidates={n: v["accuracy"] for n, v in results.items()}, **res)
        # Confidence threshold: lowest that keeps accepted clicks >= 95% correct.
        conf, ok = np.array(res["confidences"]), np.array(res["correct"])
        self.threshold = None
        for t in np.unique(np.r_[0.0, conf]):
            acc = ok[conf >= t]
            if len(acc) and acc.mean() >= 0.95:
                self.threshold = float(t)
                break
        if self.threshold is None:
            self.threshold = float(np.percentile(conf, 50))
        kept = conf >= self.threshold
        r["threshold"] = self.threshold
        r["accepted_rate"] = float(kept.mean())
        r["accepted_accuracy"] = float(ok[kept].mean()) if kept.any() else None
        progress("Training final model")
        make = self._methods()[best]
        if make == "offset":
            self.offset = self._touch_offset(S, targets)
        self.clf = make().fit(S, cells) if callable(make) else None

        # WHEN: touching posture vs. not (body only) and motion thresholds.
        body_touch = X[:, N_GAZE:]
        ok_rows = ~np.isnan(body_touch[:, 0])
        Xw = np.vstack([body_touch[ok_rows], body_rest])
        yw = np.r_[np.ones(ok_rows.sum()), np.zeros(len(body_rest))]
        make_when = lambda: make_pipeline(_pose_and_silhouette(), LogisticRegression(C=0.3, max_iter=3000))
        r["touch_vs_rest_accuracy"] = float(np.mean(cross_val_score(make_when(), Xw, yw, cv=5)))
        self.when = make_when().fit(Xw, yw)
        hold, reach = np.asarray(hold_motion), np.asarray(reach_motion)
        self.still_thr = float(np.percentile(hold, 85)) if len(hold) else 0.05
        self.release_thr = max(self.still_thr * 2.5, float(np.percentile(reach, 50)) if len(reach) else 0.0)
        self.report = r
        return self

    @staticmethod
    def _touch_offset(S, targets):
        ok = ~np.isnan(S[:, 0])
        return np.median(targets[ok] - S[ok, :2], axis=0) if ok.any() else np.zeros(2)

    def _proba(self, model, S, offset=None):
        """Class probabilities averaged over frames."""
        n = self.cols * self.rows
        proba = np.zeros(n)
        if model is None:  # eyes only: vote by where the gaze points fall
            g = S[~np.isnan(S[:, 0]), :2] + (self.offset if offset is None else offset)
            if len(g):
                for xy in g:
                    proba[self.cell_of(xy)] += 1
                proba /= len(g)
            return proba
        proba[model.classes_] = model.predict_proba(S).mean(axis=0)
        return proba

    def _leave_one_round_out(self, make, S, cells, targets, captures, rounds, valid):
        n = self.cols * self.rows
        confusion = np.zeros((n, n), int)
        correct, confs = [], []
        for rd in np.unique(rounds):
            test = rounds == rd
            if callable(make) and len(np.unique(cells[~test])) < 2:
                continue
            m = make().fit(S[~test], cells[~test]) if callable(make) else None
            offset = self._touch_offset(S[~test], targets[~test]) if make == "offset" else np.zeros(2)
            for cap in np.unique(captures[test]):
                sel = test & (captures == cap)
                if not valid[sel][0]:
                    continue
                proba = self._proba(m, S[sel], offset)
                pred, truth = int(proba.argmax()), int(cells[sel][0])
                confusion[truth, pred] += 1
                correct.append(pred == truth)
                confs.append(float(proba.max()))
        per_cell = {int(c): float(confusion[c, c] / confusion[c].sum()) for c in range(n) if confusion[c].sum()}
        return {"accuracy": float(np.mean(correct)) if correct else 0.0, "per_cell": per_cell,
                "confusion": confusion.tolist(), "correct": [bool(c) for c in correct], "confidences": confs}

    def predict_cell(self, joint_frames):
        """Average over a few frames -> (cell, confidence)."""
        proba = self._proba(self.clf, self._stage2_X(np.array(joint_frames)))
        return int(proba.argmax()), float(proba.max())

    def p_touch(self, body_features):
        return float(self.when.predict_proba(body_features[None])[0, 1])

    def save(self, path=CELL_MODEL_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path=CELL_MODEL_PATH) -> "CellModel":
        if not path.exists():
            raise SystemExit("No cell model. Run `magictap calibrate` first.")
        from . import legacy_unpickle_aliases

        legacy_unpickle_aliases()
        return joblib.load(path)


def fit_gaze_mapper(gaze_train, gaze_check, canvas_size, progress=lambda s: None):
    """Pick the gaze mapper that does best on check dots it never trained on; refit on all."""
    Xg, Yg, _ = gaze_train
    Xv, Yv, gv = gaze_check
    w, h = canvas_size
    scores = {}
    for name, make in GAZE_CANDIDATES.items():
        progress(f"Eyes: trying {name}")
        m = make().fit(Xg, Yg)
        preds, truth = per_point(m, Xv, Yv, gv)
        scores[name] = score({g: np.clip(p, 0, (w, h)) for g, p in preds.items()}, truth, canvas_size)
    best = min(scores, key=lambda n: scores[n]["median"])
    mapper = GAZE_CANDIDATES[best]().fit(np.vstack([Xg, Xv]), np.vstack([Yg, Yv]))
    return mapper, best, scores[best], {n: round(s["median"], 1) for n, s in scores.items()}


# --------------------------------------------------------------------------
# Pooled gaze mapper: every stored eye calibration, judged on sessions it never saw.
# --------------------------------------------------------------------------

# Eye-image variants first: on equal scores they generalised best across sessions.
EYE_FIRST = ["gaze ray + eye images", "gaze ray + iris + eye images", "all gaze features", "gaze ray"]


def load_gaze_sessions(runs_dir):
    """One entry per stored calibration with an eye-calibration step.

    X/Y = look-only dots (training + check), plus the touch-round gaze of cells-mode runs
    (those were recorded with text kept off the centre of the screen).
    """
    sessions = []
    for p in sorted(runs_dir.glob("*/dataset.npz")):
        try:
            d = np.load(p)
        except Exception:
            continue
        if "gaze_train_X" not in d.files or "gaze_check_X" not in d.files:
            continue
        X = [d["gaze_train_X"], d["gaze_check_X"]]
        Y = [d["gaze_train_Y"], d["gaze_check_Y"]]
        if "X" in d.files and "targets" in d.files:
            g = d["X"][:, :N_GAZE]
            ok = ~np.isnan(g[:, 0])
            X.append(g[ok])
            Y.append(d["targets"][ok])
        sessions.append({"name": p.parent.name, "X": np.vstack(X), "Y": np.vstack(Y),
                         "train": (d["gaze_train_X"], d["gaze_train_Y"], d["gaze_train_groups"]),
                         "check": (d["gaze_check_X"], d["gaze_check_Y"], d["gaze_check_groups"])})
    return sessions


def _held_out_score(make, train_sessions, test, canvas_size):
    """Fit on other sessions; re-centre on the test session's training dots; score its check dots."""
    from .recenter import apply_correction, fit_correction

    m = make().fit(np.vstack([s["X"] for s in train_sessions]), np.vstack([s["Y"] for s in train_sessions]))
    tX, tY, tg = test["train"]
    P = m.predict(tX)
    groups = np.unique(tg)
    M, _ = fit_correction([np.median(P[tg == g], axis=0) for g in groups], [tY[tg == g][0] for g in groups])
    vX, vY, vg = test["check"]
    Q = apply_correction(M, m.predict(vX))
    w, h = canvas_size
    preds = {int(g): np.clip(np.median(Q[vg == g], axis=0), 0, (w, h)) for g in np.unique(vg)}
    truth = {int(g): vY[vg == g][0] for g in np.unique(vg)}
    return score(preds, truth, canvas_size)


def fit_gaze_mapper_pooled(sessions, canvas_size, progress=lambda s: None):
    """Leave-one-session-out choice among the gaze candidates, then fit on all sessions pooled.

    Returns (mapper, name, held-out score dict, {candidate: mean held-out median}) or None.
    """
    if len(sessions) < 2:
        return None
    results = {}
    for name in EYE_FIRST:
        progress(f"Eyes: {name}, leave-one-session-out over {len(sessions)} sessions")
        runs = [_held_out_score(GAZE_CANDIDATES[name], [o for o in sessions if o is not s], s, canvas_size)
                for s in sessions]
        results[name] = {"median": float(np.mean([r["median"] for r in runs])),
                         "p90": float(np.mean([r["p90"] for r in runs])),
                         "region": float(np.mean([r["region"] for r in runs])),
                         "col": float(np.mean([r["col"] for r in runs])),
                         "row": float(np.mean([r["row"] for r in runs]))}
    # Prefer eye-image variants when they're within 10% of the best held-out error: they carry more
    # of the eyes' appearance and generalised better to a new, independent session.
    floor = min(r["median"] for r in results.values())
    best = next(n for n in EYE_FIRST if results[n]["median"] <= 1.10 * floor)
    mapper = GAZE_CANDIDATES[best]().fit(np.vstack([s["X"] for s in sessions]), np.vstack([s["Y"] for s in sessions]))
    eye_score = dict(results[best], sessions=len(sessions), method="held-out session, after re-centre")
    return mapper, best, eye_score, {n: round(r["median"], 1) for n, r in results.items()}
