"""Python client for `magictap serve`.

    from magictap.client import MagicTapClient

    client = MagicTapClient()
    client.set_targets([{"id": "next", "x": 1200, "y": 300, "w": 400, "h": 300}])
    client.on_tap = lambda tap: print(tap["target"], tap["confidence"])
    client.run_forever()
"""

import asyncio
import json


class MagicTapClient:
    def __init__(self, url="ws://127.0.0.1:8765", subscribe_gaze=False, reconnect_s=2.0):
        self.url, self.subscribe_gaze, self.reconnect_s = url, subscribe_gaze, reconnect_s
        self.targets = []
        self.screen = None
        self.on_tap = None       # fn(dict): {"type":"tap","x","y","target","confidence","t"}
        self.on_ignored = None   # fn(dict): {"type":"ignored","reason","t"}
        self.on_gaze = None      # fn(dict): {"type":"gaze","x","y"}  (only if subscribe_gaze)
        self._ws = None

    def set_targets(self, targets):
        """Targets in global screen points: [{"id", "x", "y", "w", "h"}]. Replaces the previous set."""
        self.targets = list(targets)
        if self._ws is not None:
            asyncio.run_coroutine_threadsafe(self._send_targets(), self._loop)

    async def _send_targets(self):
        await self._ws.send(json.dumps({"type": "targets", "targets": self.targets}))

    async def run(self):
        from websockets.asyncio.client import connect

        self._loop = asyncio.get_running_loop()
        while True:
            try:
                async with connect(self.url) as ws:
                    self._ws = ws
                    await self._send_targets()
                    if self.subscribe_gaze:
                        await ws.send(json.dumps({"type": "subscribe", "gaze": True}))
                    async for raw in ws:
                        msg = json.loads(raw)
                        kind = msg.get("type")
                        if kind == "hello":
                            self.screen = msg.get("screen")
                        elif kind == "tap" and self.on_tap:
                            self.on_tap(msg)
                        elif kind == "ignored" and self.on_ignored:
                            self.on_ignored(msg)
                        elif kind == "gaze" and self.on_gaze:
                            self.on_gaze(msg)
            except (OSError, ConnectionError) as e:
                print(f"Magic Tap: can't reach {self.url} ({e}); is `magictap serve` running? Retrying...")
            except Exception as e:  # connection dropped
                print(f"Magic Tap: connection lost ({type(e).__name__}); reconnecting...")
            finally:
                self._ws = None
            await asyncio.sleep(self.reconnect_s)

    def run_forever(self):
        try:
            asyncio.run(self.run())
        except KeyboardInterrupt:
            pass
