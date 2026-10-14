"""`magictap doctor` and `magictap setup`."""

from .config import CALIBRATION_PATH, CELL_MODEL_PATH, HOME

OK, BAD, WARN = "✓", "✗", "!"


def _auth(media):
    import AVFoundation as AV

    status = AV.AVCaptureDevice.authorizationStatusForMediaType_(media)
    return {0: "not asked yet", 1: "restricted", 2: "denied", 3: "allowed"}.get(status, str(status))


def run_doctor(fix_models=True):
    """Print a checklist; returns True if Magic Tap can run (calibration may still be missing)."""
    import AVFoundation as AV

    rows, ready = [], True

    def row(ok, name, detail, fix=None):
        rows.append((ok, name, detail, fix))

    cam = _auth(AV.AVMediaTypeVideo)
    if cam == "not asked yet":
        from .camera import ensure_camera_permission

        try:
            ensure_camera_permission()
            cam = "allowed"
        except RuntimeError:
            cam = _auth(AV.AVMediaTypeVideo)
    row(OK if cam == "allowed" else BAD, "Camera", cam,
        None if cam == "allowed" else "System Settings > Privacy & Security > Camera: enable your terminal app")
    ready &= cam == "allowed"

    mic = _auth(AV.AVMediaTypeAudio)
    if mic == "not asked yet":
        from .audio import ensure_mic_permission

        try:
            ensure_mic_permission()
            mic = "allowed"
        except RuntimeError:
            mic = _auth(AV.AVMediaTypeAudio)
    row(OK if mic == "allowed" else BAD, "Microphone", mic,
        None if mic == "allowed" else "System Settings > Privacy & Security > Microphone: enable your terminal app")
    ready &= mic == "allowed"

    from .mouse import accessibility_ok

    acc = accessibility_ok()
    row(OK if acc else WARN, "Accessibility", "allowed" if acc else "not allowed (only needed for `magictap mouse`)",
        None if acc else "System Settings > Privacy & Security > Accessibility: enable your terminal app")

    from .camera import list_cameras

    cams = list_cameras()
    builtin = [c for c in cams if "MacBook" in c]
    row(OK if builtin else WARN, "Built-in camera", builtin[0] if builtin else f"not found ({', '.join(cams) or 'none'})",
        None if builtin else "Magic Tap is tuned for a laptop's built-in camera; use --camera N to pick another")

    if fix_models:
        try:
            from .body import _ensure
            from .config import FACE_MODEL, HAND_MODEL, POSE_MODEL
            from .gaze import _load_gaze_net

            for m in (POSE_MODEL, HAND_MODEL, FACE_MODEL):
                _ensure(m)
            _load_gaze_net()
            row(OK, "Models", f"downloaded to {HOME / 'models'}")
        except Exception as e:  # network or disk problem
            row(BAD, "Models", f"download failed: {e}", "Check your internet connection and run `magictap doctor` again")
            ready = False

    from .tapcal import TAP_MODEL_PATH

    cal = CALIBRATION_PATH.exists() and CELL_MODEL_PATH.exists()
    row(OK if cal else WARN, "Eye + touch calibration", "done" if cal else "not yet", None if cal else "Run `magictap calibrate`")
    taps = TAP_MODEL_PATH.exists()
    row(OK if taps else WARN, "Tap training", "done" if taps else "not yet", None if taps else "Run `magictap taps`")
    from .recenter import MAX_AGE_HOURS, load_correction

    M, age = load_correction()
    fresh = M is not None and age <= MAX_AGE_HOURS
    row(OK if fresh else WARN, "Re-centre (this session)",
        f"{age:.1f} h ago" if M is not None else "not yet",
        None if fresh else "Run `magictap recenter` (15 s) at the start of each session")

    print("\nMagic Tap doctor\n")
    for ok, name, detail, fix in rows:
        print(f"  {ok} {name:<26} {detail}")
        if fix:
            print(f"      → {fix}")
    print(f"\n  Data folder: {HOME}\n")
    return ready


def run_setup(camera=None):
    print("Magic Tap setup: 1) check your laptop  2) eye + touch calibration (~6 min)  3) tap training (~2 min)"
          "  4) re-centre (15 s)\n")
    if not run_doctor():
        raise SystemExit("Fix the items marked ✗ above, then run `magictap setup` again.")
    input("Press Enter to start calibration (a fullscreen window opens and waits for Space)...")
    from .calibration import run_cell_calibration

    run_cell_calibration(camera)
    input("\nPress Enter to start tap training (fullscreen, waits for Space)...")
    from .tapcal import run_tap_calibration

    run_tap_calibration()
    input("\nPress Enter to re-centre (15 s: look at 6 dots, touch each, press Space)...")
    from .recenter import run_recenter

    run_recenter(camera)
    print("\nAll set. Try it:  magictap buttons\nUse it in other apps:  magictap mouse   or   magictap serve\n"
          "Each new session (new seat, light or lid angle): magictap recenter  (15 s)\n")
