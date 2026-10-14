"""`magictap serve`: a local WebSocket service apps use to receive taps.

Protocol (JSON text frames) - see README "Integrate it":
  client -> server  {"type":"targets","targets":[{"id","x","y","w","h"}]}   (global screen points, replaces the set)
                    {"type":"subscribe","gaze":true}
  server -> client  {"type":"hello","version","screen":[w,h]}
                    {"type":"tap","x","y","target","confidence","t"}
                    {"type":"ignored","reason","t"}
                    {"type":"gaze","x","y"}                                    (~15 Hz, only if subscribed)
"""

import asyncio
import json

from . import __version__
from .fusion import error_msg, gaze_msg, hello_msg, ignored_msg, parse_client, resolve_tap

HOST, PORT = "127.0.0.1", 8765
GAZE_HZ = 15


class Hub:
    """Connected clients and their targets; turns engine events into per-client messages."""

    def __init__(self, screen):
        self.screen = screen
        self.clients = {}  # websocket -> {"targets": [...], "gaze": bool}
        self._last_gaze = 0.0

    def hello(self):
        return hello_msg(__version__, self.screen)

    def handle(self, ws, raw):
        """Apply a client message; returns a reply message or None."""
        kind, value = parse_client(raw)
        state = self.clients.setdefault(ws, {"targets": [], "gaze": False})
        if kind == "targets":
            state["targets"] = value
            return {"type": "ok", "targets": len(value)}
        if kind == "subscribe":
            state["gaze"] = value
            return {"type": "ok", "gaze": value}
        return error_msg(value)

    def tap_messages(self, t, samples):
        return [(ws, resolve_tap(t, samples, self.screen, st["targets"])) for ws, st in self.clients.items()]

    def ignored_messages(self, t, reason):
        return [(ws, ignored_msg(reason, t)) for ws in self.clients]

    def gaze_messages(self, t, xy):
        if t - self._last_gaze < 1 / GAZE_HZ:
            return []
        self._last_gaze = t
        return [(ws, gaze_msg(xy)) for ws, st in self.clients.items() if st["gaze"]]


def make_handler(hub):
    async def handler(ws):
        hub.clients[ws] = {"targets": [], "gaze": False}
        await ws.send(json.dumps(hub.hello()))
        try:
            async for raw in ws:
                reply = hub.handle(ws, raw)
                if reply:
                    await ws.send(json.dumps(reply))
        finally:
            hub.clients.pop(ws, None)
    return handler


async def _send_all(pairs):
    for ws, msg in pairs:
        try:
            await ws.send(json.dumps(msg))
        except Exception:
            pass


def run_serve(camera=None, host=HOST, port=PORT, no_camera=False):
    from websockets.asyncio.server import serve

    engine = None
    if no_camera:
        from .calibration import Calibration
        from .config import CALIBRATION_PATH
        from .screen import main_canvas

        if CALIBRATION_PATH.exists():
            c = Calibration.load()
            screen = (c.canvas_size[0] + c.canvas_offset[0], c.canvas_size[1] + c.canvas_offset[1])
        else:
            cv = main_canvas()
            screen = (cv.width + cv.offset_x, cv.height + cv.offset_y)
    else:
        from .engine import Engine

        engine = Engine(camera)
        screen = engine.screen
    hub = Hub(screen)
    handler = make_handler(hub)

    async def main():
        loop = asyncio.get_running_loop()
        if engine:
            post = lambda pairs: asyncio.run_coroutine_threadsafe(_send_all(pairs), loop)
            engine.on_tap = lambda t, s: post(hub.tap_messages(t, s))
            engine.on_ignored = lambda t, r: post(hub.ignored_messages(t, r))
            engine.on_gaze = lambda t, xy: post(hub.gaze_messages(t, xy))
            engine.start()
        async with serve(handler, host, port):
            print(f"Magic Tap service on ws://{host}:{port}  (screen {int(screen[0])}x{int(screen[1])} pt). Ctrl+C to stop.")
            await asyncio.Future()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        if engine:
            engine.stop()
