"""Camera capture on a background thread, always serving the newest frame."""

import threading
import time

import cv2


def list_cameras() -> list[str]:
    """Camera names in the order OpenCV's AVFoundation backend indexes them."""
    import AVFoundation as AV

    devices = AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo)
    return [str(d.localizedName()) for d in devices]


def find_camera(name: str) -> int:
    for i, cam in enumerate(list_cameras()):
        if name.lower() in cam.lower():
            return i
    return 0


def disable_center_stage() -> None:
    """Center Stage crops and pans the image, which would break calibration."""
    try:
        import AVFoundation as AV

        AV.AVCaptureDevice.setCenterStageControlMode_(1)  # app-controlled
        AV.AVCaptureDevice.setCenterStageEnabled_(False)
    except Exception:
        pass


def ensure_camera_permission(timeout: float = 60) -> None:
    """Ask macOS for camera access (once) and wait for the answer."""
    import AVFoundation as AV

    status = AV.AVCaptureDevice.authorizationStatusForMediaType_(AV.AVMediaTypeVideo)
    if status == 0:  # not determined yet
        done = threading.Event()
        AV.AVCaptureDevice.requestAccessForMediaType_completionHandler_(AV.AVMediaTypeVideo, lambda ok: done.set())
        print("Waiting for camera permission (check the macOS dialog)...")
        done.wait(timeout)
        status = AV.AVCaptureDevice.authorizationStatusForMediaType_(AV.AVMediaTypeVideo)
    if status != 3:  # 3 = authorized
        raise RuntimeError(
            "Camera access is not granted. Enable your terminal app in System Settings > "
            "Privacy & Security > Camera, then restart the terminal."
        )


class Camera:
    def __init__(self, index: int, width: int, height: int):
        ensure_camera_permission()
        disable_center_stage()
        self.cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {index}. Check System Settings > Privacy & "
                "Security > Camera and allow your terminal app."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self._frame = None
        self._t = 0.0
        self._seq = 0
        self._cond = threading.Condition()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._cond:
                self._frame, self._t = frame, time.monotonic()
                self._seq += 1
                self._cond.notify_all()

    def read(self, last_seq: int, timeout: float = 1.0):
        """Block until a frame newer than `last_seq` arrives. Returns (frame, t, seq)."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq > last_seq, timeout)
            if self._frame is None:
                raise RuntimeError("No frames from camera (is camera permission granted?)")
            return self._frame, self._t, self._seq

    def close(self):
        self._running = False
        self._thread.join(timeout=1)
        self.cap.release()


class VideoSource:
    """Replays a recorded video as if it were a live camera (for offline tuning)."""

    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self._seq = 0

    def read(self, last_seq: int, timeout: float = 1.0):
        ok, frame = self.cap.read()
        if not ok:
            raise EOFError
        self._seq += 1
        return frame, self._seq / self.fps, self._seq

    def close(self):
        self.cap.release()
