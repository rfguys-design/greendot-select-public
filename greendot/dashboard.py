"""Local results dashboard. OWNER: Markus

A web page on this PC (http://localhost:8765) that shows what GreenDot is doing
and every result: the hint text, the copied text, the chart. main.py calls
set_status() and add_result(); the page polls /state twice a second.

Standard library only -- no web framework to install.
"""

import json
import os
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

from greendot.config import CONFIG

_lock = threading.Lock()
_state = {"version": 0, "status": "starting", "detail": "", "results": []}
MAX_RESULTS = 12
_on_capture = None   # set by start(); called when the page asks for a capture


def set_status(status, detail=""):
    """status is one of: ready, captured, listening, thinking, speaking, error."""
    with _lock:
        _state["status"] = status
        _state["detail"] = detail
        _state["version"] += 1


def add_result(**fields):
    """Add one capture result. Image paths are converted to dashboard URLs."""
    for key in ("crop", "plot", "image"):
        path = fields.pop(key, None)
        if path:
            rel = os.path.relpath(path, CONFIG["crop_dir"]).replace(os.sep, "/")
            fields[key + "_url"] = "/files/" + rel
    fields["time"] = time.strftime("%H:%M:%S")
    with _lock:
        _state["results"].insert(0, fields)
        del _state["results"][MAX_RESULTS:]
        _state["version"] += 1


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # keep the terminal for capture messages

    def _send(self, body, content_type, code=200):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path.startswith("/state"):
            with _lock:
                body = json.dumps(_state, ensure_ascii=False)
            self._send(body.encode("utf-8"), "application/json; charset=utf-8")
        elif self.path.startswith("/files/"):
            # Only files inside crop_dir can be served.
            root = os.path.realpath(CONFIG["crop_dir"])
            path = os.path.realpath(os.path.join(root, unquote(self.path[len("/files/"):])))
            if not path.startswith(root + os.sep) or not os.path.isfile(path):
                self._send(b"not found", "text/plain", 404)
                return
            kind = "image/png" if path.endswith(".png") else "image/jpeg"
            with open(path, "rb") as f:
                self._send(f.read(), kind)
        else:
            self._send(b"not found", "text/plain", 404)

    def do_POST(self):
        # The Capture button / double-click on the page. POST only, so merely
        # opening a URL can never take a capture.
        if self.path == "/capture" and _on_capture:
            print("[trigger] dashboard")
            threading.Thread(target=_on_capture, daemon=True).start()
            self._send(b"ok", "text/plain")
        else:
            self._send(b"not found", "text/plain", 404)


def start(on_capture=None, open_browser=True):
    """Serve the dashboard on localhost in a background thread.

    on_capture: called (on its own thread) when the page's Capture button is
    clicked or the page is double-clicked -- a backup for the headphone tap.
    """
    global _on_capture
    _on_capture = on_capture
    port = CONFIG["dashboard_port"]
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://localhost:{port}"
    print(f"[dashboard] {url}")
    if open_browser:
        webbrowser.open(url)
    return server


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GreenDot Select</title>
<style>
  :root {
    --bg: #f6f7f5; --card: #ffffff; --text: #1d211f; --muted: #66706b;
    --line: #e2e6e3; --accent: #1f9d55; --accent-soft: #e3f4ea;
    --warn: #c2410c; --warn-soft: #fdeee4; --blue: #2563eb; --blue-soft: #e6eefe;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #121513; --card: #1b1f1d; --text: #e8ece9; --muted: #98a29d;
      --line: #2b312e; --accent: #4ade80; --accent-soft: #173323;
      --warn: #fb923c; --warn-soft: #3a2316; --blue: #7aa2ff; --blue-soft: #1c2847;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 16px/1.5 system-ui, -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; }
  header { display: flex; align-items: center; gap: 14px; padding: 18px 28px;
           border-bottom: 1px solid var(--line); background: var(--card); }
  header h1 { font-size: 20px; margin: 0; font-weight: 650; }
  .dot { width: 14px; height: 14px; border-radius: 50%; background: var(--accent); }
  #capture { margin-left: auto; padding: 7px 18px; border-radius: 999px; border: 0;
             background: var(--accent); color: #fff; font: inherit; font-weight: 650; cursor: pointer; }
  #capture:hover { filter: brightness(1.08); }
  #capture:disabled { opacity: .45; cursor: default; }
  .hint { color: var(--muted); font-size: 13px; }
  #status { padding: 6px 14px; border-radius: 999px; font-weight: 600;
            background: var(--accent-soft); color: var(--accent); }
  #status.busy { background: var(--blue-soft); color: var(--blue); }
  #status.busy::before { content: ""; display: inline-block; width: 8px; height: 8px;
            margin-right: 8px; border-radius: 50%; background: currentColor;
            animation: pulse 1s infinite; vertical-align: middle; }
  #status.error { background: var(--warn-soft); color: var(--warn); }
  @keyframes pulse { 50% { opacity: .25; } }
  main { max-width: 1100px; margin: 0 auto; padding: 24px 28px 60px; }
  .empty { color: var(--muted); text-align: center; padding: 80px 0; font-size: 18px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 14px;
          padding: 22px 24px; margin-bottom: 18px; }
  .meta { display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
          color: var(--muted); font-size: 14px; }
  .chip { padding: 2px 10px; border-radius: 999px; background: var(--accent-soft);
          color: var(--accent); font-weight: 600; font-size: 13px; }
  .said { margin: 10px 0 0; color: var(--muted); font-style: italic; }
  .text { white-space: pre-wrap; margin: 14px 0 0; font-size: 20px; }
  .latest .text { font-size: 26px; line-height: 1.45; }
  .note { margin-top: 12px; font-size: 14px; color: var(--accent); font-weight: 600; }
  .note.warn { color: var(--warn); }
  .images { display: flex; flex-wrap: wrap; gap: 16px; margin-top: 16px; align-items: flex-start; }
  .images figure { margin: 0; }
  .images figcaption { font-size: 13px; color: var(--muted); margin-top: 4px; }
  .crop { max-width: 340px; max-height: 260px; border-radius: 8px; border: 1px solid var(--line); }
  .plot { max-width: 100%; width: 640px; border-radius: 8px; background: #fff; }
  .picture { max-width: 100%; max-height: 70vh; border-radius: 8px; }
  .older .picture { max-height: 220px; }
  .older .crop { max-width: 200px; max-height: 140px; }
  .older .plot { width: 360px; }
  .older .text { font-size: 16px; max-height: 9em; overflow: hidden; }
  h2 { font-size: 14px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted);
       margin: 30px 0 12px; }
</style>
</head>
<body>
<header><span class="dot"></span><h1>GreenDot Select</h1>
  <button id="capture" title="Same as a headphone tap">Capture</button>
  <span class="hint">or double-click the page</span>
  <span id="status">starting</span></header>
<main id="main"><div class="empty">Place two green dots, then tap your headphones.</div></main>
<script>
const LABELS = { ready: "Ready — tap to capture", captured: "Captured", listening: "Listening…",
  thinking: "Thinking…", speaking: "Speaking…", error: "Problem", starting: "Starting…" };
const TASKS = { hint: "Hint", solve_math: "Solution", read_aloud: "Read aloud",
  translate: "Translation", explain: "Explanation", copy_text: "Copied text", copy_image: "Copied image", edit_image: "New picture",
  plot_table: "Chart", unclear: "Unclear", no_dots: "No dots" };
let version = -1;
let busy = false;

function capture() { if (!busy) fetch("/capture", { method: "POST" }); }
document.getElementById("capture").addEventListener("click", capture);
document.addEventListener("dblclick", e => {
  if (e.target.closest("#capture")) return;      // the button already fired
  window.getSelection().removeAllRanges();        // a double-click selects a word; undo that
  capture();
});

function esc(s) { const d = document.createElement("div"); d.textContent = s ?? ""; return d.innerHTML; }

function card(r, latest) {
  const imgs = [];
  if (r.image_url) imgs.push(`<figure><img class="picture" src="${r.image_url}"><figcaption>New picture</figcaption></figure>`);
  if (r.plot_url) imgs.push(`<figure><img class="plot" src="${r.plot_url}"><figcaption>Chart</figcaption></figure>`);
  if (r.crop_url) imgs.push(`<figure><img class="crop" src="${r.crop_url}"><figcaption>What the camera captured</figcaption></figure>`);
  return `<section class="card ${latest ? "latest" : "older"}">
    <div class="meta"><span class="chip">${esc(TASKS[r.task] || r.task)}</span>
      <span>${esc(r.time)}</span>${r.latency ? `<span>${r.latency}s</span>` : ""}
      ${r.path ? `<span>${esc(r.path)}</span>` : ""}</div>
    ${r.transcript ? `<p class="said">You said: “${esc(r.transcript)}”</p>` : ""}
    ${r.text ? `<div class="text">${esc(r.text)}</div>` : ""}
    ${r.note ? `<div class="note ${r.warn ? "warn" : ""}">${esc(r.note)}</div>` : ""}
    ${imgs.length ? `<div class="images">${imgs.join("")}</div>` : ""}
  </section>`;
}

async function poll() {
  try {
    const s = await (await fetch("/state")).json();
    if (s.version !== version) {
      version = s.version;
      const el = document.getElementById("status");
      el.textContent = LABELS[s.status] + (s.detail ? " · " + s.detail : "");
      el.className = s.status === "error" ? "error" : (s.status === "ready" ? "" : "busy");
      busy = !(s.status === "ready" || s.status === "error");
      document.getElementById("capture").disabled = busy;
      const [first, ...rest] = s.results;
      document.getElementById("main").innerHTML = first
        ? card(first, true) + (rest.length ? "<h2>Earlier</h2>" + rest.map(r => card(r, false)).join("") : "")
        : '<div class="empty">Place two green dots, then tap your headphones.</div>';
    }
  } catch (e) {
    document.getElementById("status").textContent = "GreenDot is not running";
    document.getElementById("status").className = "error";
  }
  setTimeout(poll, 500);
}
poll();
</script>
</body>
</html>
"""
