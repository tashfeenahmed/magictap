"""Tap detection from the built-in microphone, with layered noise rejection.

1. Onset detector: scans 3 ms windows (a fingertip impact lasts only a few ms,
   so longer windows dilute gentle taps). A sound must jump well above the
   *current* background level (adaptive noise floor) and be spiky, so steady
   noise - fans, music, a noisy room - never triggers it on its own. Each clip
   is aligned so the impact starts at exactly the same sample.
2. Tap classifier: each onset's short "sound shape" (spectrum of the attack,
   how fast it decays, low thud vs. high click) is scored by a classifier
   trained on the user's own screen taps vs. typing, talking, knocks, claps...
3. Gates applied by the caller: no keyboard/trackpad press in the last moment
   (macOS reports these), and the eyes must be on the screen.
"""

import collections
import threading
import time

import numpy as np

SR = 48000
HOP = 480  # 10 ms analysis frames
WIN_BEFORE, WIN_AFTER = 2, 10  # frames kept around an onset (20 ms before, 100 ms after)
N_MELS = 32
REFRACTORY_S = 0.25  # one sound -> one onset
ONSET_RATIO = 6.0  # energy must jump this many times above the noise floor
MIN_LEVEL = 1e-6  # ignore near-silence


def _mel_filters(n_fft=1024, n_mels=N_MELS, sr=SR, fmin=40, fmax=16000):
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    inv = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = inv(np.linspace(mel(fmin), mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        a, b, c = bins[i], bins[i + 1], bins[i + 2]
        fb[i, a:b] = (np.arange(a, b) - a) / max(b - a, 1)
        fb[i, b:c] = (c - np.arange(b, c)) / max(c - b, 1)
    return fb


_MEL = _mel_filters()
_WINDOW = np.hanning(1024)


def tap_features(clip: np.ndarray) -> np.ndarray:
    """Features of a ~120 ms clip starting 20 ms before the onset."""
    clip = clip.astype(np.float64)
    o = WIN_BEFORE * HOP
    attack = clip[o:o + 1024]
    body = clip[o + 1024:o + 1024 + 2048]
    spec = lambda x: np.abs(np.fft.rfft(np.pad(x[:1024], (0, max(0, 1024 - len(x[:1024])))) * _WINDOW)) ** 2
    mel_attack = np.log(_MEL @ spec(attack) + 1e-10)
    mel_body = np.log(_MEL @ spec(body[:1024]) + 1e-10)
    frames = clip[: (len(clip) // HOP) * HOP].reshape(-1, HOP)
    env = np.log(np.mean(frames ** 2, axis=1) + 1e-12)
    env = env - env.max()  # loudness-independent envelope shape
    freqs = np.fft.rfftfreq(1024, 1 / SR)
    sa = spec(attack)
    centroid = float((freqs * sa).sum() / (sa.sum() + 1e-12))
    low = float(sa[freqs < 300].sum() / (sa.sum() + 1e-12))
    peak = float(np.max(np.abs(clip[o:o + 960])) / (np.sqrt(np.mean(clip[o:o + 960] ** 2)) + 1e-12))
    # spectral shape relative to its own level (so soft and hard taps look alike)
    return np.r_[mel_attack - mel_attack.mean(), mel_body - mel_attack.mean(), env,
                 np.log(centroid + 1), low, peak, mel_attack.mean()].astype(np.float32)


def ensure_mic_permission(timeout: float = 60) -> None:
    """Ask macOS for microphone access (once) and wait for the answer."""
    import AVFoundation as AV

    status = AV.AVCaptureDevice.authorizationStatusForMediaType_(AV.AVMediaTypeAudio)
    if status == 0:
        done = threading.Event()
        AV.AVCaptureDevice.requestAccessForMediaType_completionHandler_(AV.AVMediaTypeAudio, lambda ok: done.set())
        print("Waiting for microphone permission (check the macOS dialog)...")
        done.wait(timeout)
        status = AV.AVCaptureDevice.authorizationStatusForMediaType_(AV.AVMediaTypeAudio)
    if status != 3:
        raise RuntimeError("Microphone access is not granted. Enable your terminal app in System Settings > "
                           "Privacy & Security > Microphone, then restart the terminal.")


SHORT_WIN = 144  # 3 ms: a fingertip impact lasts only a few ms; longer windows dilute gentle taps
SHORT_HOP = 72  # 1.5 ms between short windows
PRE = WIN_BEFORE * HOP  # samples kept before the exact onset
POST = WIN_AFTER * HOP  # samples kept after it
MIN_CREST = 1.5  # peak / RMS inside the short window: impacts are spiky, hum and voices aren't


def find_onset(x, floor_energy=0.0):
    """Index of the first sample of the impact in x (the 3 ms window with the biggest energy jump)."""
    n = len(x)
    if n < SHORT_WIN:
        return 0
    starts = np.arange(0, n - SHORT_WIN + 1, SHORT_HOP)
    energy = np.array([np.mean(x[s:s + SHORT_WIN] ** 2) for s in starts])
    w = int(np.argmax(energy))
    seg = np.abs(x[starts[w]:starts[w] + SHORT_WIN])
    level = max(0.5 * seg.max(), 3 * np.sqrt(floor_energy))
    above = np.nonzero(seg >= level)[0]
    return int(starts[w] + (above[0] if len(above) else 0))


def realign_clip(clip):
    """Shift a recorded clip so its impact starts exactly at PRE (for clips saved before alignment)."""
    clip = np.asarray(clip, np.float32)
    on = find_onset(clip[: PRE + 2 * HOP], float(np.median(clip[:PRE] ** 2)))
    shift = on - PRE
    out = np.zeros_like(clip)
    if shift >= 0:
        out[: len(clip) - shift] = clip[shift:]
    else:
        out[-shift:] = clip[: len(clip) + shift]
    return out


class OnsetDetector:
    """Streaming onset detector working on 3 ms windows, with sample-exact onset alignment.

    feed() takes successive audio blocks and returns finished events as (time, clip), where the
    impact always starts exactly PRE samples into the clip. Counters record every stage.
    """

    def __init__(self, ratio=ONSET_RATIO, min_level=MIN_LEVEL):
        self.ratio, self.min_level = ratio, min_level
        self.size = 2 * SR
        self.ring = np.zeros(self.size, np.float32)
        self.total = 0  # samples seen so far
        self.floor_hist = collections.deque(maxlen=150)  # ~1.5 s of per-block window medians
        self.pending = []  # (onset sample index, onset time)
        self.last_onset = -10 ** 12
        self.floor = 0.0
        self.counts = collections.Counter()

    def _get(self, start, length):
        idx = (np.arange(start, start + length)) % self.size
        return self.ring[idx]

    def feed(self, x, t_block):
        x = np.asarray(x, np.float32)
        n = len(x)
        block_start = self.total
        idx = (np.arange(block_start, block_start + n)) % self.size
        self.ring[idx] = x
        self.total += n

        # 3 ms windows over this block plus the tail of the previous one
        lead = min(SHORT_WIN - SHORT_HOP, block_start)
        seg = self._get(block_start - lead, n + lead)
        starts = np.arange(0, len(seg) - SHORT_WIN + 1, SHORT_HOP)
        energy = np.array([np.mean(seg[s:s + SHORT_WIN] ** 2) for s in starts])
        peak = np.array([np.max(np.abs(seg[s:s + SHORT_WIN])) for s in starts])
        floor = float(np.median(self.floor_hist)) if len(self.floor_hist) > 20 else float(np.median(energy))
        self.floor = floor
        crest = peak / np.sqrt(energy + 1e-20)
        hits = np.nonzero((energy > self.ratio * floor) & (energy > self.min_level) & (crest > MIN_CREST))[0]
        if len(hits):
            w = hits[0]
            win = np.abs(seg[starts[w]:starts[w] + SHORT_WIN])
            level = max(0.5 * win.max(), 3 * np.sqrt(floor))
            first = np.nonzero(win >= level)[0]
            onset = block_start - lead + starts[w] + (int(first[0]) if len(first) else 0)
            if onset - self.last_onset > REFRACTORY_S * SR:
                self.last_onset = onset
                self.counts["sounds"] += 1
                self.pending.append((onset, t_block + (onset - block_start) / SR))
        self.floor_hist.append(float(np.median(energy)))

        out, keep = [], []
        for onset, t in self.pending:
            if onset + POST <= self.total:
                if onset - PRE >= max(0, self.total - self.size):
                    out.append((t, self._get(onset - PRE, PRE + POST).copy()))
            else:
                keep.append((onset, t))
        self.pending = keep
        return out


class Mic:
    """Microphone stream; emits onsets (time, clip) via `onsets()`."""

    def __init__(self, device=None):
        import sounddevice as sd

        ensure_mic_permission()
        self.detector = OnsetDetector()
        self._ready = []
        self._lock = threading.Lock()
        self.level = 0.0
        self.floor = 0.0
        self.recording = None  # optional list to collect raw audio
        self.stream = sd.InputStream(samplerate=SR, channels=1, blocksize=HOP, dtype="float32",
                                     device=device, callback=self._callback, latency="low")
        self.latency = float(self.stream.latency)
        self.stream.start()

    def _callback(self, data, frames, time_info, status):
        t = time.monotonic() - self.latency - frames / SR  # approximate capture time of this block's start
        x = data[:, 0].copy()
        if self.recording is not None:
            self.recording.append((t, x))
        events = self.detector.feed(x, t)
        with self._lock:
            self.level, self.floor = float(np.mean(x ** 2)), self.detector.floor
            self._ready += events

    @property
    def counts(self):
        return self.detector.counts

    def onsets(self):
        with self._lock:
            out, self._ready = self._ready, []
        return out

    def close(self):
        self.stream.stop()
        self.stream.close()


def seconds_since_input():
    """Seconds since the last key press or trackpad/mouse click (no permission needed)."""
    import Quartz

    src = Quartz.kCGEventSourceStateHIDSystemState
    return min(Quartz.CGEventSourceSecondsSinceLastEventType(src, k) for k in
               (Quartz.kCGEventKeyDown, Quartz.kCGEventLeftMouseDown, Quartz.kCGEventRightMouseDown))
