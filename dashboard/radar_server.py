# radar_server.py -- relay latest MATLAB CFAR results to HarmonyOS app
# Python 3 standard library only. Run on a trusted local network.
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Lock

latest_frame = {"frame_id": 0, "weather": "waiting", "targets": []}
lock = Lock()


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/api/frame":
            self.reply(404, {"error": "not found"})
            return
        with lock:
            frame = latest_frame.copy()
        self.reply(200, frame)

    def do_POST(self):
        global latest_frame
        if self.path != "/api/frame":
            self.reply(404, {"error": "not found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 200000:
                raise ValueError("invalid body size")
            frame = json.loads(self.rfile.read(size).decode("utf-8"))
            if not isinstance(frame, dict) or not isinstance(frame.get("targets"), list):
                raise ValueError("expected a JSON object with a targets array")
            if not isinstance(frame.get("frame_id"), (int, float)):
                raise ValueError("frame_id must be numeric")
            for t in frame["targets"]:
                if not isinstance(t, dict) or not all(k in t for k in
                    ("range_m", "velocity_mps", "peak_db")):
                    raise ValueError("each target requires range_m, velocity_mps, peak_db")
            with lock:
                latest_frame = frame
            print("Frame", frame["frame_id"], "weather", frame.get("weather"),
                  "targets", len(frame["targets"]), flush=True)
            self.reply(200, {"ok": True})
        except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as err:
            self.reply(400, {"error": str(err)})


if __name__ == "__main__":
    print("Radar API: http://127.0.0.1:8765/api/frame", flush=True)
    print("Local-network demo only; do not expose this unauthenticated server to the Internet.")
    ThreadingHTTPServer(("0.0.0.0", 8765), Handler).serve_forever()
