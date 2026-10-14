"""Record a demo: Magic Tap guides you to touch cells while the camera records.

Rest your finger on the highlighted cell and press Space (other hand) to move on.
The output is raw material for editing: the camera video of you, and the exact
time of every Space press plus which cell it was. The polished buttons are
rendered separately and synced to these timestamps.
"""

import json
import random
import time

import cv2

from .app import text
from .camera import Camera, find_camera
from .config import CAMERA_HEIGHT, CAMERA_NAME, CAMERA_WIDTH, DATA_DIR
from .screen import main_canvas, open_fullscreen

WINDOW = "Magic Tap - record demo"
COLS, ROWS = 3, 2
LABELS = "ABCDEF"


def _cell_rect(canvas, i, pad=14):
    r, c = divmod(i, COLS)
    cw, ch = canvas.width / COLS, canvas.height / ROWS
    return int(c * cw + pad), int(r * ch + pad), int((c + 1) * cw - pad), int((r + 1) * ch - pad)


def _draw(canvas, target, flash=None, status=""):
    img = canvas.blank(10)
    for i in range(COLS * ROWS):
        x0, y0, x1, y1 = _cell_rect(canvas, i)
        on = i == target
        cv2.rectangle(img, (x0, y0), (x1, y1), (60, 60, 60), 2, cv2.LINE_AA)
        if on:
            cv2.rectangle(img, (x0, y0), (x1, y1), (70, 50, 20), -1)
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 190, 90), 4, cv2.LINE_AA)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            cv2.circle(img, (cx, cy), 46, (255, 210, 120), 3, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), 10, (255, 230, 170), -1, cv2.LINE_AA)
        if flash is not None and flash[0] == i:
            cv2.rectangle(img, (x0, y0), (x1, y1), flash[1], -1)
        text(img, LABELS[i], (x0 + 30, y0 + 50), 1.1, (150, 150, 150), 2)
    if status:
        text(img, status, (canvas.width / 2, canvas.height - 22), 0.55, (150, 150, 150), 1, center=True)
    return img


def run_record_demo(camera=None, taps=24, seed=None):
    canvas = main_canvas()
    cam = Camera(find_camera(CAMERA_NAME) if camera is None else camera, CAMERA_WIDTH, CAMERA_HEIGHT)
    out = DATA_DIR / "demo_runs" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    open_fullscreen(WINDOW)

    rng = random.Random(seed)
    order = []
    while len(order) < taps:  # every cell equally often, no immediate repeats
        block = list(range(COLS * ROWS))
        rng.shuffle(block)
        if order and block[0] == order[-1]:
            block.reverse()
        order += block
    order = order[:taps]

    writer, frame_times, events = None, [], []
    seq = 0

    def grab():
        nonlocal writer, seq
        prev = seq
        frame, t, seq = cam.read(seq, timeout=0.05)
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out / "camera.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
        if recording and seq != prev:  # only new frames, so timing stays exact
            writer.write(frame)
            frame_times.append(t)
        return frame

    recording = False
    try:
        # ready: wait for Space, with a small self-view so you can frame yourself
        while True:
            frame = grab()
            img = canvas.blank(10)
            vw = 640
            view = cv2.resize(frame, (vw, int(frame.shape[0] * vw / frame.shape[1])))
            y0 = (canvas.height - view.shape[0]) // 2 - 40
            img[y0:y0 + view.shape[0], (canvas.width - vw) // 2:(canvas.width + vw) // 2] = view
            text(img, "Record demo", (canvas.width / 2, 60), 1.0, center=True)
            text(img, f"{taps} cells, about a minute. Rest your finger on the highlighted cell, then press Space.",
                 (canvas.width / 2, y0 + view.shape[0] + 50), 0.6, (200, 200, 200), 1, center=True)
            text(img, "Space = start  (during recording: Space = next cell, Esc stops)",
                 (canvas.width / 2, canvas.height - 40), 0.7, (0, 220, 255), center=True)
            cv2.imshow(WINDOW, img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                raise KeyboardInterrupt
            if k == ord(" "):
                break
        recording = True
        for n in (3, 2, 1):
            t0 = time.monotonic()
            while time.monotonic() - t0 < 1:
                grab()
                img = _draw(canvas, None)
                text(img, str(n), (canvas.width / 2, canvas.height / 2), 4, (255, 255, 255), 6, center=True)
                cv2.imshow(WINDOW, img)
                cv2.waitKey(1)

        for k_, target in enumerate(order):
            t_prompt = time.monotonic()
            flash, done = None, False
            while not done:
                grab()
                img = _draw(canvas, target, status=f"{k_ + 1} / {len(order)}")
                cv2.imshow(WINDOW, img)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    raise KeyboardInterrupt
                if key == ord(" "):
                    events.append({"n": k_, "cell": target, "label": LABELS[target], "t_prompt": t_prompt,
                                   "t_press": time.monotonic()})
                    done = True
            t_done = time.monotonic()
            while time.monotonic() - t_done < 0.45:  # confirmation flash
                grab()
                cv2.imshow(WINDOW, _draw(canvas, None, (target, (60, 170, 60))))
                cv2.waitKey(1)
    except KeyboardInterrupt:
        pass
    finally:
        recording = False
        if writer is not None:
            writer.release()
        cam.close()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    meta = {
        "grid": [COLS, ROWS], "canvas": [canvas.width, canvas.height],
        "video": "camera.mp4", "video_fps_nominal": 30,
        "frame_times": frame_times,               # monotonic seconds of each written frame
        "events": events,                          # one per cell: which cell + when Space was pressed
    }
    (out / "demo.json").write_text(json.dumps(meta, indent=1))
    dur = (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0
    print(f"Recorded {dur:.0f} s of video and {len(events)} presses. Saved to {out}")
