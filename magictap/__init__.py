"""Magic Tap: touch input for a laptop screen using gaze and the sound of a tap."""

__version__ = "0.2.0"


def legacy_unpickle_aliases():
    """Models saved before the rename reference the `holo` package; map it to `magictap`."""
    import importlib
    import sys

    for name in ("", ".model", ".tapcal", ".body", ".gaze", ".touch", ".calibration", ".audio"):
        try:
            sys.modules.setdefault("holo" + name, importlib.import_module("magictap" + name))
        except Exception:
            pass
