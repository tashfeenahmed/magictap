# Magic Tap

**Touch input for a MacBook screen that has no touchscreen.** Magic Tap uses only
the built-in camera and microphone: no mirror and no extra hardware.

![Gaze and tap localization](docs/gaze-and-tap-localization.png)

- **Where:** your **eyes**. People look at what they touch. The camera tracks
  your gaze and maps it to a point on the screen.
- **When:** the **sound of your tap**. The microphone hears the fingertip hit the
  glass. Typing, talking, music and room noise are filtered out.

Put together, you look at a button and tap the screen on it, and Magic Tap
presses that button.

## How it works

**Vision path:**
1. **Face:** a MediaPipe face mesh with irises.
2. **Head pose:** the face is normalised for head pose.
3. **Gaze:** an ETH-XGaze ResNet-18 estimates where you're looking.
4. **Screen point:** a per-user mapping, learned during calibration, turns that into a
   point on the screen.

While you touch the screen your gaze reads slightly high, so Magic Tap learns that
shift and subtracts it.

**Audio path:**
1. **Onset detection:** 3 ms windows against an adaptive noise floor, with the clip aligned
   to the exact sample of impact.
2. **Classifier:** a tap classifier trained on your taps vs. your everyday noises.
3. **Unusual-sound check:** rejects sounds that don't resemble any of your taps.
4. **Gates:** no key or trackpad press in the last 150 ms, and your eyes must be on the screen.

**Fusion:** when a tap is heard, the button is decided by where your eyes were
over the last ~0.3 s. If the vote isn't confident enough, Magic Tap doesn't click.

## Results so far (one user, not a controlled study)

| | |
|---|---|
| Gaze error on held-out dots | 83 pt median |
| Taps detected | 91–95 % |
| False taps (typing, talking, knocks, music) | 0 of 73 |
| Live accuracy, 3×2 buttons | 81 % (26/32) before the latest fixes |

Accuracy is good enough for a few large buttons, not for precise pointing.
[CONFIGURATION.md](CONFIGURATION.md) has the exact setup, parameters and findings.

## Requirements

- MacBook with Apple Silicon and the built-in camera and microphone (developed on an M4 Max)
- macOS 14 or later, Python 3.12, [uv](https://docs.astral.sh/uv/)
- Camera and Microphone permission for your terminal app (and Accessibility for `live`)

## Usage

```sh
uv sync
uv run holo preview      # check framing: face, shoulders and eyes should be tracked
uv run holo calibrate    # ~6 min: look at dots, then touch each cell 3 times
uv run holo taps         # ~2 min: teach it your taps vs. typing, talking and noise
uv run holo buttons      # look at a button and tap it; guided mode learns as you go
```

Every mode opens a live view first and waits for **Space**.

| Command | What it does |
|---|---|
| `holo preview` | Camera view with the body and eyes overlay |
| `holo calibrate` | Eye calibration, plus touch rounds on a grid (picks the finest reliable grid) |
| `holo taps` | Tap-sound training (`--retrain` refits from the last recording) |
| `holo buttons` | Big on-screen buttons with right/wrong feedback |
| `holo retrain` | Retrain the cell model from a saved calibration recording |
| `holo live` | Drive the real mouse pointer (needs Accessibility) |

Recordings (video and per-frame features) are saved under `data/`, which is
git-ignored, so methods can be compared offline on the same session.

## Tests

```sh
uv run pytest
```

## Credits

- **Gaze model weights:** ETH-XGaze ResNet-18 by hysts
  ([ptgaze](https://github.com/hysts/pytorch_mpiigaze_demo), MIT). The 3D face template
  in `holo/assets` comes from the same project; its licence is included.
- **Face, pose and hand landmarks:** [MediaPipe](https://github.com/google-ai-edge/mediapipe).
- **Ideas for the tap detector:** short-window onsets, exact alignment, novelty
  rejection and learning from evaluation taps were inspired by
  [JustinGamer191/Holo](https://github.com/JustinGamer191/Holo) and its forks.
