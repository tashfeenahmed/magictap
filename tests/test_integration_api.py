"""Tests for the integration layer: fusion logic and the WebSocket service (no camera or mic)."""

import asyncio
import json

from magictap.fusion import (GazeBuffer, confidence, hit_test, on_screen, parse_client, resolve_tap,
                             to_global)
from magictap.serve import Hub, _send_all, make_handler

SCREEN = (1728, 1117)
TARGETS = [{"id": "a", "x": 0, "y": 0, "w": 864, "h": 558}, {"id": "b", "x": 864, "y": 0, "w": 864, "h": 558},
           {"id": "small", "x": 100, "y": 100, "w": 50, "h": 50}]


def test_gaze_window_uses_frames_around_the_tap():
    g = GazeBuffer()
    g.add(0.0, (10, 10))      # long before: previous target
    g.add(9.75, (500, 300))
    g.add(9.9, None)          # blink
    g.add(10.02, (510, 310))
    g.add(10.2, (900, 900))   # after the window
    assert g.window(10.0) == [(500.0, 300.0), (510.0, 310.0)]


def test_hit_test_prefers_the_smallest_containing_target():
    assert hit_test((120, 120), TARGETS)["id"] == "small"
    assert hit_test((1000, 100), TARGETS)["id"] == "b"
    assert hit_test((1000, 900), TARGETS) is None


def test_confidence_is_share_of_samples_inside():
    assert confidence([(10, 10), (20, 20), (1000, 10), (30, 30)], TARGETS[0]) == 0.75
    assert confidence([], TARGETS[0]) == 0.0


def test_resolve_tap_messages():
    m = resolve_tap(5.0, [(1000, 100), (1010, 110), (990, 90)], SCREEN, TARGETS)
    assert m["type"] == "tap" and m["target"] == "b" and m["confidence"] == 1.0
    m = resolve_tap(5.0, [(1000, 100), (1010, 110)], SCREEN, [])
    assert m["type"] == "tap" and m["target"] is None
    assert resolve_tap(5.0, [(1000, 100)], SCREEN, TARGETS) == {"type": "ignored", "reason": "eyes not on screen", "t": 5.0}
    assert resolve_tap(5.0, [(5000, 100), (5100, 90)], SCREEN, TARGETS)["reason"] == "eyes not on screen"


def test_helpers():
    assert to_global((10, 20), (0, 32)) == (10, 52)
    assert on_screen((-50, 10), SCREEN) and not on_screen((-500, 10), SCREEN)


def test_parse_client_messages():
    assert parse_client(json.dumps({"type": "targets", "targets": [{"id": 1, "x": 0, "y": 0, "w": 5, "h": 5}]}))[0] == "targets"
    assert parse_client(json.dumps({"type": "subscribe", "gaze": True})) == ("subscribe", True)
    assert parse_client("nope")[0] == "error"
    assert parse_client(json.dumps({"type": "targets", "targets": [{"id": 1}]}))[0] == "error"


def test_websocket_service_end_to_end():
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve

    async def scenario():
        hub = Hub(SCREEN)
        async with serve(make_handler(hub), "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}") as ws:
                hello = json.loads(await ws.recv())
                assert hello["type"] == "hello" and hello["screen"] == [1728, 1117]
                await ws.send(json.dumps({"type": "targets", "targets": TARGETS}))
                assert json.loads(await ws.recv()) == {"type": "ok", "targets": 3}
                await ws.send(json.dumps({"type": "subscribe", "gaze": True}))
                assert json.loads(await ws.recv())["gaze"] is True
                # simulate the engine: a heard tap with gaze on target "b", then a gaze sample
                await _send_all(hub.tap_messages(7.0, [(1000, 100), (1005, 105)]))
                tap = json.loads(await ws.recv())
                assert tap["type"] == "tap" and tap["target"] == "b"
                await _send_all(hub.gaze_messages(8.0, (12, 34)))
                assert json.loads(await ws.recv()) == {"type": "gaze", "x": 12, "y": 34}
                await _send_all(hub.ignored_messages(9.0, "keyboard/trackpad"))
                assert json.loads(await ws.recv())["reason"] == "keyboard/trackpad"

    asyncio.run(scenario())
