"""Minimal Magic Tap integration in Python.

Run `magictap serve` in another terminal, then:  uv run python examples/python_client.py
Splits the screen into a left and right half and reports which one you tap.
"""

from magictap.client import MagicTapClient

client = MagicTapClient()


def on_tap(tap):
    print(f"tapped {tap['target'] or 'nothing'} at ({tap['x']:.0f}, {tap['y']:.0f}), confidence {tap['confidence']:.2f}")


def on_ignored(msg):
    print(f"ignored: {msg['reason']}")


client.on_tap, client.on_ignored = on_tap, on_ignored
# Screen size is announced by the service on connect; assume a 1728x1117 pt MacBook display here.
client.set_targets([
    {"id": "left", "x": 0, "y": 0, "w": 864, "h": 1117},
    {"id": "right", "x": 864, "y": 0, "w": 864, "h": 1117},
])
client.run_forever()
