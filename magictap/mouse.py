"""`magictap mouse`: zero-integration mode. Look at something, tap the screen, and it's clicked."""

import time

from .fusion import MIN_CONFIDENCE, resolve_tap


def accessibility_ok() -> bool:
    from ApplicationServices import AXIsProcessTrusted

    return bool(AXIsProcessTrusted())


def click(x, y):
    import Quartz

    for kind in (Quartz.kCGEventLeftMouseDown, Quartz.kCGEventLeftMouseUp):
        ev = Quartz.CGEventCreateMouseEvent(None, kind, (x, y), Quartz.kCGMouseButtonLeft)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
        time.sleep(0.02)


def run_mouse(camera=None, min_confidence=MIN_CONFIDENCE):
    if not accessibility_ok():
        raise SystemExit("Mouse mode needs Accessibility permission to click.\n"
                         "System Settings > Privacy & Security > Accessibility: enable your terminal app, then retry.")
    from .engine import Engine

    engine = Engine(camera)

    def on_tap(t, samples):
        msg = resolve_tap(t, samples, engine.screen, [])
        if msg["type"] == "tap" and msg["confidence"] >= min_confidence:
            click(msg["x"], msg["y"])
            print(f"click at ({msg['x']:.0f}, {msg['y']:.0f})  confidence {msg['confidence']:.2f}")
        elif msg["type"] == "tap":
            print(f"not sure (confidence {msg['confidence']:.2f}) - no click")
        else:
            print(f"ignored: {msg['reason']}")

    engine.on_tap = on_tap
    engine.on_ignored = lambda t, reason: print(f"ignored sound: {reason}")
    engine.start()
    print("Mouse mode: once ready, look at something and tap the screen on it to click it. Ctrl+C to quit.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
