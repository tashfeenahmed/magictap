# Magic Tap: working configuration

This is the exact setup that produced the working **look-and-tap buttons**: your
**eyes** pick the button, and the **sound of your tap** on the screen says when.
It uses only the MacBook's built-in camera and microphone, with no mirror and no
extra hardware.

Snapshot: 2026-10-14, Magic Tap 0.1.0.

---

## 1. Result

| What | Score | How it was measured |
|---|---|---|
| Look-and-tap buttons, 3×2 grid (live) | **26 / 32 right (81%)** | Button screen, guided mode, before the two fixes in §6 |
| Eye tracking on check dots | **median 83 pt** off; 3×3 cell **100%** | 9 dots the eye model never trained on |
| Which button, from eyes at touch time (3×2) | **100%** | Each touch round predicted by a model that never saw it (offline) |
| Screen taps heard | **86%** (gentle 75%) at first; **91–95%** (gentle 75–88%) with the unusual-sound check (§3) | Taps held out from training |
| False taps from typing, talking, knocks, claps, music, quiet room | **0** | 73 noise sounds |

## 2. Hardware and environment

| Item | Setting |
|---|---|
| Machine | MacBook Pro, Apple M4 Max, macOS 26.5 |
| Display | Built-in 3456×2234 Retina, **1728×1117 pt**. Fullscreen canvas = **1728×1085 pt** (32 pt notch inset at the top) |
| Camera | Built-in "MacBook Pro Camera", **1760×1328 (4:3)** at 30 fps. Center Stage **off** (Magic Tap turns it off). Max available is 1920 px wide |
| Microphone | Built-in "MacBook Pro Microphone", **48 kHz mono**, ~61 ms input latency |
| Position | Sit as you will use it. Lid tilted so **shoulders and elbows are in view**. Keep the lid angle and seat the same after calibrating |
| Permissions (terminal app) | Camera, Microphone. Accessibility only for `magictap live` |
| Python | 3.12 via `uv` |
| Key packages | mediapipe 1.1.0, opencv-python 5.0.0.93, torch 2.14.1 (MPS), timm 1.0.30, scikit-learn 1.9.1, numpy 2.5.4, sounddevice 0.5.6, pyobjc |

## 3. How it works

```
camera ─┬─ MediaPipe Face Landmarker (478 pts + irises)
        │     └─ head pose (solvePnP on 3D face template)
        │         └─ face normalised to a virtual camera 60 cm away (224×224)
        │             └─ ETH-XGaze ResNet-18 ──> gaze pitch/yaw ──> gaze ray
        │                 └─ per-user gaze mapper ──> point on screen (pt)
        │                     └─ + touch shift (learned) ──> which button
        └─ MediaPipe Pose (+ segmentation) and Hands ──> body features
              (recorded and used for the touch-shift / arm features)

microphone ─ onset detector (adaptive noise floor)
            └─ tap classifier (your taps vs. your noises)
                └─ gates: no key/trackpad press within 0.15 s, eyes on the screen
                    └─ TAP at time t ──> button = eyes in [t − 0.3 s, t + 0.05 s]
```

### WHERE: eyes
- **Gaze model:** ETH-XGaze ResNet-18 (`hysts/ptgaze-eth-xgaze-resnet18`, MIT),
  run on Apple MPS. About 12 ms per frame including face mesh.
- **Normalisation** (Zhang et al. 2018, as in ptgaze): virtual camera focal 960,
  distance 0.6 m, 224×224 crop. Real-camera intrinsics are approximated from a
  110° diagonal FOV; calibration absorbs the error.
- **Gaze features (17 numbers + eye images):** gaze vector (3), pitch/yaw (2),
  face centre (3), head rotation (3), **gaze-ray hit point on the camera
  plane (2)**, iris ratios (4); plus two 24×12 grey, contrast-normalised eye crops.
- **Per-user mapper chosen in calibration: "gaze ray + eye images":**
  quadratic in the ray hit point, plus linear gaze vector and angles, plus
  20 PCA components of the eye images, fitted with RidgeCV
  (alphas 10⁻²…10⁴). Picked because it scored best on check dots:

  | Candidate | Check-dot median |
  |---|---|
  | **gaze ray + eye images** | **83 pt** |
  | gaze ray + iris + eye images | 112 pt |
  | all gaze features (linear) | 149 pt |
  | gaze ray (no eye images) | 151 pt |

- **Touch shift:** while touching, gaze reads about **+187 pt too high**
  (dx ≈ +1, dy ≈ +187). It's learned from the touch rounds and **subtracted**.
  Without it, 4×3 accuracy was 42%; with it, 81%.
- **Button choice:** cells of a **3×2 grid** over the full canvas
  (576 × 542.5 pt each). Each frame's gaze point votes for a cell; confidence
  is the winning share. **Minimum confidence 0.6**; below that the screen
  says "not sure" and doesn't click.

### WHEN: tap sound
- **Onset detector** (`audio.OnsetDetector`). It scans **3 ms windows** (1.5 ms hop),
  because a fingertip impact lasts only a few ms and 10 ms windows dilute gentle taps
  (idea from Holo PR #17). A window counts as an onset if its energy is **> 6×** the
  running noise floor (median of the last ~1.5 s) **and** > 1e-6 **and** it is spiky
  (peak/RMS > 1.5). A **0.25 s** refractory period gives one onset per sound.
- **Exact onset alignment** (idea from Holo PR #16): the clip is cut so the impact
  starts at **exactly** sample 960 (20 ms pre-roll, 100 ms after). Previously it
  started anywhere inside a 10 ms block, so features partly described buffer timing.
  Tested: onset found to within ~0.1 ms.
- **Features (80):** 32 log-mel bands of the attack (first ~21 ms) and of the
  following ~21 ms (relative to the attack level), the 12-frame energy
  envelope shape, spectral centroid, energy share below 300 Hz (the lid thud),
  peak-to-RMS ratio, and overall level.
- **Classifier:** StandardScaler + LogisticRegression (C = 0.5). It beat
  ExtraTrees with zero false alarms. The threshold is set just above the
  highest-scoring held-out noise sound that got past the other checks (minimum 0.35).
- **"Unusual sound" check** (novelty; idea from Holo PR #17): a sound is also rejected
  if it's far from every known tap. Measure: nearest-neighbour distance in standardised
  feature space, with a limit of 1.5× the 95th percentile of tap-to-tap distances. It
  blocks about half of all noise before the classifier (19/23 talking, 18/36 everyday
  noises). That lets the classifier be less strict: taps caught went **86% → 95%**,
  gentle taps **75% → 88%**, with 0 false taps.
- **Gates:**
  1. **Keyboard/trackpad:** the sound is ignored if a key or click happened within
     **0.15 s** (macOS `CGEventSourceSecondsSinceLastEventType`; no permission needed).
  2. **Eyes:** at least 2 tracked gaze frames in the tap window, and the gaze must be
     on the screen (within 100 pt of its edges).

### Learning while you use it (guided mode)
Idea from the devarshh08 fork. In guided mode Magic Tap knows which button you were asked
for, so every tap is a labelled example:
- **Tap model:** each heard guided tap is added (most recent **40** kept) and the
  model is refitted every 5 taps. New taps come from the aligned detector, so the
  model gradually moves onto natively aligned data.
- **Touch shift:** updated from where your eyes were vs. the asked button. The new value
  blends the calibration shift (weighted as 10 taps) with up to 40 recent guided taps.
- Both are saved when you quit (`Learned from N guided taps. Touch shift A -> B`).
- The saved session also records per-stage counts: sounds heard, and ignored by reason.

## 4. Calibration (what you did)

Both steps open a screen that waits for **Space**. **P** pauses and **Esc** quits.

### 4a. Eyes + touch rounds: `magictap calibrate` (~6 min, cells mode)
1. **Position check:** live camera view; checklist must show body, shoulders,
   elbows and eyes OK.
2. **Eyes:** look at **20 dots** (5×4 grid, 1.6 s each, first 0.6 s ignored).
3. **Eyes + head movement:** **6 dots** (3×2) for **4 s** each. Keep your eyes on the
   dot while leaning left/right and forward/back. This stops the model from
   treating head position as a shortcut for gaze.
4. **Check dots:** **9 dots** (3×3, offset from the training grid). Used only to
   score eye accuracy and pick the gaze mapper.
5. **Not touching:** 8 s, moving naturally.
6. **Touch rounds:** 4×3 grid, every cell **3 times** in separate rounds, at a
   random spot inside the cell. Look at the dot, touch it, press **Space** with
   the other hand; 1 s is recorded.
7. **Training:** the gaze mapper, then the cell model for 4×3, 3×3, 3×2 and 2×2, each
   scored by **hiding one whole round at a time**. Dots near a cell edge are
   excluded from that grid's score. The finest grid that is **≥ 95% right on ≥ 80% of
   touches** is chosen: **3×2**.

| Grid | Method chosen | Right on fresh rounds |
|---|---|---|
| 4×3 | eyes + touch shift | 72% |
| 3×3 | eyes + touch shift | 88% |
| **3×2** | **eyes + touch shift** | **100%** |
| 2×2 | eyes only | 94% |

### 4b. Taps: `magictap taps` (~2 min)
1. Tap **22 dots** on the screen: 14 firm, then 8 gentle.
2. **Typing** 15 s, **talking** 10 s, **other sounds** 20 s (desk knocks, claps,
   palm-rest taps, trackpad clicks, setting things down), **media** 15 s (optional),
   **quiet** 4 s. Every sound here is a "must not trigger" example.
3. Training uses 4-fold cross-validation, split by time within each segment.

| Sounds | Heard | Ignored by keyboard check | False taps |
|---|---|---|---|
| Typing | 2 | 2 | **0** |
| Talking | 23 | 4 | **0** |
| Other noises | 36 | 1 | **0** |
| Media | 10 | 5 | **0** |
| Quiet | 2 | 0 | **0** |

## 5. Using it

```sh
magictap buttons    # 3x2 look-and-tap buttons, with right/wrong feedback
```
- **Guided mode:** a button lights up blue. Look at it and tap the screen on it.
  It flashes green "RIGHT" or red "WRONG". Grey "not sure" means no click.
- **Free mode (F):** tap any button, then press **Y** if it was right or **N** if not.
- Ignored sounds are shown at the bottom with the reason
  (*not a tap*, *keyboard/trackpad*, *you weren't looking at the screen*).
- **R** resets the score. Results go to `data/sessions/buttons-<time>.json`.

Other commands:

| Command | Purpose |
|---|---|
| `magictap preview` | Check framing; shows skeleton, silhouette and eyes |
| `magictap calibrate` | Eyes + touch-round calibration (cells mode). `--grid 4x3 --rounds 3` |
| `magictap taps` | Tap-sound training. `--retrain` refits from the last saved recording |
| `magictap retrain [run]` | Retrain from a saved calibration recording, no new session |
| `magictap experiment` | Grid experiment using the body-posture touch trigger (superseded by `buttons`) |
| `magictap live` | Drive the real mouse (needs Accessibility; uses the body trigger, not taps yet) |

## 6. Parameters

| Parameter | Value | Where |
|---|---|---|
| Camera mode | 1760×1328 @ 30 fps | `config.CAMERA_WIDTH/HEIGHT` |
| Gaze window around a tap | **−0.30 s … +0.05 s** | `buttons.TAP_WINDOW` (was −0.5…+0.1) |
| Min button confidence | **0.6** | `buttons.MIN_TAP_CONFIDENCE` (new) |
| Key/trackpad gate | 0.15 s | `tapcal.KEY_GATE_S` |
| Onset ratio / min level | 6× noise floor / 1e-6 | `audio.ONSET_RATIO`, `MIN_LEVEL` |
| Onset window / hop / spikiness | 3 ms / 1.5 ms / peak÷RMS > 1.5 | `audio.SHORT_WIN`, `SHORT_HOP`, `MIN_CREST` |
| Clip alignment | impact at sample 960 exactly | `audio.PRE`, `POST` |
| Unusual-sound limit | 1.5 × p95 tap-to-tap distance | `tapcal.TapModel` |
| Learned guided taps kept | 40 | `tapcal.MAX_LEARNED_TAPS` |
| Touch-shift prior weight | 10 taps | `buttons.Learner.PRIOR_WEIGHT` |
| Refractory | 0.25 s | `audio.REFRACTORY_S` |
| Tap threshold | learned (min 0.35) | `data/tap_model.joblib` |
| Eye dots / time / settle | 5×4, 1.6 s, 0.6 s | `calibration.GAZE_GRID`, `GAZE_DOT_TIME`, `GAZE_SETTLE` |
| Head-movement dots | 3×2, 4 s | `HEAD_DOT_TIME` |
| Touch capture | 1.0 s | `config.HOLD_TIME` |
| Not-touching phase | 8 s | `config.REST_TIME` |
| Chosen grid | 3×2 | `~/Library/Application Support/MagicTap/data/calibration.json` → `chosen_grid` |
| Touch shift | (+0.7, +187.1) pt, subtracted | `data/cell_model.joblib` → `offset` |

The last two changes (shorter gaze window and the 0.6 floor) came after the 26/32
session. Most of its 6 errors picked the *previously* pressed button, because
the old 0.5 s look-back caught the eyes before they'd moved. The other two were
low-confidence guesses.

## 7. Saved state

| File | Contents |
|---|---|
| `~/Library/Application Support/MagicTap/data/calibration.json` | Mode `cells`, chosen grid, all scores |
| `data/cell_model.joblib` | Gaze mapper, touch shift, 3×2 cell model |
| `data/tap_model.joblib` | Tap classifier and threshold |
| `data/calibration_runs/20261013-235058/` | The calibration used: full-res video, every frame's features, dataset, report |
| `data/tap_runs/20261014-000607/` | Tap training sounds and report |
| `data/sessions/buttons-20261014-001006.json` | The 26/32 button session |
| `models/` | Downloaded model files (re-downloaded automatically if missing) |

## 8. Lessons that shaped this setup

- **The camera can't see the moment of touch.** While touching, your hands are below the
  frame (wrist visibility ~0.04). A posture classifier only learned *which part of
  the session it was*; the lid nudge from a press is < 1–4 px and caught 8% of
  touches. So **when** comes from the microphone.
- **Body pose alone can't tell where** (~450 pt, close to guessing the centre). **Eyes can.**
- **Nothing readable mid-screen during calibration or use.** Instruction text in
  the centre and a camera preview pulled gaze away and once made touch-time gaze
  ~500 pt off. Text now sits in one line at the screen edge away from the target,
  and the preview is hidden.
- **Train gaze with head movement.** Without it, the model used head position as a
  shortcut and broke when you leaned to touch (1430 pt).
- **Honest scoring:** score only on rounds or dots the model never saw, and
  never on back-to-back captures (they flattered 3×3 to 94%; fresh reaches gave 36–72%).

## 9. Reproduce from scratch

```sh
uv sync
magictap preview      # tilt the lid until shoulders, elbows and eyes are tracked
magictap calibrate    # eyes + 3 touch rounds on a 4x3 grid -> picks the grid
magictap taps         # tap sounds vs. your everyday noises
magictap buttons      # look at a button, tap it
```
