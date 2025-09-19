"""
Flask backend (API + MJPEG) for AI-Drone, no UI.

Endpoints
- GET  /api/state       : consolidated JSON (video, YOLO, telemetry, class map, settings)
- GET  /api/settings    : current settings
- POST /api/settings    : update settings (e.g., {"onlyWhenYolo":true,"jpegQuality":80})
- GET  /video           : MJPEG stream (served when frames exist)
- GET  /snapshot.jpg    : last JPEG frame
- GET  /healthz         : minimal health for probes
- GET  /telemetry/live.json : legacy telemetry snapshot (compat)
"""

import os, json, time, math, threading
from typing import Dict, Any

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from flask import Flask, Response, jsonify, request

# config defaults
DEFAULT_STREAM_TOPIC = "/camera/overlay/compressed"
DEFAULT_PORT         = 5000
READY_TIMEOUT_S      = 3.0

CLASS_MAP: Dict[str, str] = {
    "0": "person",
    "7": "truck",
}

app = Flask(__name__)

@app.after_request
def add_cors_headers(resp):
    resp.headers['Access-Control-Allow-Origin']  = request.headers.get('Origin', '*')
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    resp.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return resp

latest = {'jpg': None, 't': 0.0}

state: Dict[str, Any] = {
    "video": {"ready": False, "age_ms": -1, "fps": 0.0, "last_t": 0.0},
    "yolo":  {"ready": False, "age_ms": -1, "fps": 0.0, "last_t": 0.0,
              "class_counts": {}, "backend": None, "ms": None},
    "telemetry": {"ready": False, "ts": 0.0, "lat": None, "lon": None, "alt": None,
                  "yaw_deg": None, "yaw_deg_norm": None, "yaw_deg_cont": None,
                  "pitch_deg": None, "roll_deg": None},
    "class_map": CLASS_MAP,
    "settings": {"onlyWhenYolo": True, "jpegQuality": 80}
}

def _now(): return time.monotonic()

@app.route('/api/state')
def api_state():
    now = _now()
    v = state["video"]
    y = state["yolo"]
    v["age_ms"] = int((now - latest['t']) * 1000) if latest['t'] else -1
    y["age_ms"] = int((now - y["last_t"]) * 1000) if y["last_t"] else -1
    v["ready"]  = (latest['jpg'] is not None)
    return jsonify(state)

@app.route('/api/settings', methods=['GET','POST','OPTIONS'])
def api_settings():
    if request.method == 'POST':
        incoming = request.get_json(silent=True) or {}
        for k in ("onlyWhenYolo", "jpegQuality"):
            if k in incoming:
                state["settings"][k] = incoming[k]
    return jsonify(state["settings"])

def _mjpeg_gen():
    boundary = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
    while True:
        buf = latest['jpg']
        if buf is not None:
            yield boundary + buf + b'\r\n'
        time.sleep(0.01)

@app.route('/video')
def video():
    t0 = _now()
    while latest['jpg'] is None and (_now() - t0) < READY_TIMEOUT_S:
        time.sleep(0.05)
    if latest['jpg'] is None:
        return jsonify(error="no frames yet", ready=False), 503
    return Response(_mjpeg_gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/snapshot.jpg')
def snapshot():
    if latest['jpg'] is None:
        return jsonify(error="no snapshot yet"), 503
    return Response(latest['jpg'], mimetype='image/jpeg')

@app.route('/healthz')
def healthz():
    y = state["yolo"]
    return jsonify(
        ready=(latest['jpg'] is not None),
        yolo_ready=y["ready"],
        yolo_backend=y["backend"],
        yolo_fps=y["fps"],
        infer_ms=y["ms"],
    )

@app.route('/telemetry/live.json')
def tele_compat():
    t = state["telemetry"]
    return jsonify({"ready": t["ready"], **t} if t["ready"] else {"ready": False})

class FlaskStream(Node):
    def __init__(self):
        super().__init__('flask')
        topic_img = os.getenv('STREAM_TOPIC', DEFAULT_STREAM_TOPIC)
        port      = int(os.getenv('FLASK_PORT', DEFAULT_PORT))

        self.sub_img = self.create_subscription(CompressedImage, topic_img, self._on_img, 10)
        self.sub_det = self.create_subscription(String, '/detections_raw', self._on_det, 10)
        self.sub_tel = self.create_subscription(String, '/telemetry/raw', self._on_tel, 10)

        self._yaw_last = None
        self._yaw_cont = 0.0

        threading.Thread(
            target=lambda: app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False, threaded=True),
            daemon=True
        ).start()
        self.get_logger().info(f'Flask API on http://0.0.0.0:{port}  (stream={topic_img})')

    # callbacks
    def _on_img(self, msg: CompressedImage):
        latest['jpg'] = bytes(msg.data)
        latest['t'] = _now()
        v = state["video"]
        t = _now()
        if v["last_t"] > 0:
            inst = 1.0 / max(1e-6, (t - v["last_t"]))
            v["fps"] = 0.9 * v["fps"] + 0.1 * inst
        v["last_t"] = t

    def _on_det(self, msg: String):
        y = state["yolo"]
        t = _now()
        y["last_t"] = t
        y["ready"] = True
        try:
            j = json.loads(msg.data)
            # If payload contains meta info from yolo_trt_node
            meta = j.get("meta") or {}
            if "backend" in meta: y["backend"] = meta.get("backend")
            if "ms" in meta:      y["ms"]      = float(meta.get("ms", 0))
            if "fps" in meta:     y["fps"]     = float(meta.get("fps", y["fps"]))
            # Class counts either in meta or compute from detections
            if "class_counts" in meta:
                y["class_counts"] = {str(k): int(v) for k, v in meta["class_counts"].items()}
            else:
                dets = j.get("detections", [])
                counts = {}
                for d in dets:
                    cls = str(d.get("cls"))
                    if cls is None: continue
                    counts[cls] = counts.get(cls, 0) + 1
                y["class_counts"] = counts
        except Exception:
            pass

    def _on_tel(self, msg: String):
        tstate = state["telemetry"]
        try:
            j = json.loads(msg.data)
            tstate["ready"] = True
            tstate["ts"]    = float(j.get("ts", time.time()))
            tstate["lat"]   = _to_float(j.get("lat"))
            tstate["lon"]   = _to_float(j.get("lon"))
            tstate["alt"]   = _to_float(j.get("alt"))

            yaw = _to_float(j.get("yaw_deg"))
            if yaw is not None:
                # normalize -180..180
                yaw_norm = ((yaw + 180.0) % 360.0) - 180.0
                tstate["yaw_deg"]      = yaw
                tstate["yaw_deg_norm"] = yaw_norm
                # unwrap for continuity
                if self._yaw_last is None:
                    self._yaw_cont = yaw
                else:
                    d = yaw - self._yaw_last
                    d = (d + 180.0) % 360.0 - 180.0  # shortest arc
                    # self._yaw_cont += d
                self._yaw_last = yaw
                tstate["yaw_deg_cont"] = self._yaw_cont

            tstate["pitch_deg"] = _to_float(j.get("pitch_deg"))
            tstate["roll_deg"]  = _to_float(j.get("roll_deg"))
        except Exception:
            tstate["ready"] = False

def _to_float(v):
    try:
        if v is None: return None
        return float(v)
    except Exception:
        return None

def main():
    rclpy.init()
    n = FlaskStream()
    rclpy.spin(n)
    n.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

