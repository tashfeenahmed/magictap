// Magic Tap client for `magictap serve` (ES module, no dependencies).
//
//   import { connect } from "./magictap.js";
//   const mt = connect();                                   // ws://127.0.0.1:8765
//   mt.setTargets([{ id: "next", x: 1200, y: 300, w: 400, h: 300 }]);   // global screen points
//   mt.on("tap", (tap) => console.log(tap.target, tap.confidence));
//
// Events: "hello" {version, screen:[w,h]}, "tap" {x, y, target, confidence, t},
//         "ignored" {reason, t}, "gaze" {x, y} (after subscribeGaze()), "open", "close".

export function connect(url = "ws://127.0.0.1:8765", { reconnectMs = 2000 } = {}) {
  const handlers = {};
  let ws = null;
  let targets = [];
  let gaze = false;
  let closed = false;

  const emit = (name, data) => (handlers[name] || []).forEach((fn) => fn(data));
  const send = (msg) => { if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg)); };

  function open() {
    ws = new WebSocket(url);
    ws.onopen = () => {
      send({ type: "targets", targets });
      if (gaze) send({ type: "subscribe", gaze: true });
      emit("open");
    };
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      emit(msg.type, msg);
    };
    ws.onclose = () => {
      emit("close");
      if (!closed) setTimeout(open, reconnectMs);
    };
  }
  open();

  return {
    on(name, fn) { (handlers[name] ||= []).push(fn); return this; },
    setTargets(list) { targets = list; send({ type: "targets", targets }); },
    subscribeGaze(on = true) { gaze = on; send({ type: "subscribe", gaze: on }); },
    close() { closed = true; ws && ws.close(); },
  };
}

// Helper for web pages: an element's rectangle in global screen points.
// Exact when the page is fullscreen; otherwise it accounts for the window position and toolbar.
export function screenRect(el) {
  const r = el.getBoundingClientRect();
  const chromeTop = window.outerHeight - window.innerHeight;
  const fullscreen = !!document.fullscreenElement;
  const left = fullscreen ? 0 : window.screenX;
  const top = fullscreen ? 0 : window.screenY + chromeTop;
  return { x: left + r.left, y: top + r.top, w: r.width, h: r.height };
}
