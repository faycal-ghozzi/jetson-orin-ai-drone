import threading, time
from flask import Flask, Response, jsonify, redirect, url_for
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

STREAM_TOPIC = "/camera/overlay/compressed"
FLASK_PORT = 5000
READY_TIMEOUT_S = 3.0

app = Flask(__name__)

_latest_jpg = b""
_latest_jpg_t = 0.0

_tel = {
    "ts": 0.0,
    "lat": None, "lon": None, "alt": None,
    "yaw_deg": None, "pitch_deg": None, "roll_deg": None,
    "enu": {"x": None, "y": None, "z": None}
}

HTML_INDEX = """<!doctype html>
<title>AI-Drone</title>
<style>body{background:#111;color:#ccc;font:16px/1.4 system-ui,Segoe UI,Roboto,sans-serif} .wrap{max-width:1200px;margin:24px auto;text-align:center} img{max-width:100%;height:auto;background:#000} a{color:#7fd}</style>
<div class="wrap">
  <h1>AI-Drone Stream</h1>
  <p><a href="/snapshot.jpg" target="_blank">Snapshot</a> ·
     <a href="/healthz" target="_blank">Health</a> ·
     <a href="/telemetry/live.json" target="_blank">Telemetry JSON</a></p>
  <img src="/video" alt="stream"/>
</div>
"""

def _now(): return time.monotonic()

@app.route("/")
def index(): return HTML_INDEX

def _mjpeg_gen():
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    while True:
        if _latest_jpg:
            yield boundary + _latest_jpg + b"\r\n"
        time.sleep(0.01)

@app.route("/video")
def video():
    t0 = _now()
    while not _latest_jpg and (_now() - t0) < READY_TIMEOUT_S:
        time.sleep(0.05)
    if not _latest_jpg:
        return jsonify(error="no frames yet", ready=False), 503
    return Response(_mjpeg_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/snapshot.jpg")
def snapshot():
    if not _latest_jpg:
        return jsonify(error="no snapshot yet"), 503
    return Response(_latest_jpg, mimetype="image/jpeg")

@app.route("/healthz")
def healthz():
    age_ms = int((_now() - _latest_jpg_t) * 1000) if _latest_jpg_t else -1
    return jsonify(ready=bool(_latest_jpg), age_ms=age_ms)

@app.route("/telemetry/live.json")
def telemetry_live():
    age = time.time() - (_tel["ts"] or 0.0)
    ready = age < 2.0
    return jsonify({"ready":ready, "age_ms": int(age*1000) if _tel["ts"] else -1, **_tel})

@app.errorhandler(404)
def not_found(_): return redirect(url_for("index"))

class FlaskStream(Node):
    def __init__(self):
        super().__init__("flask")
        self.create_subscription(CompressedImage, STREAM_TOPIC, self._on_img, 10)
        self.create_subscription(PoseStamped, "/telemetry/pose", self._on_pose, 10)
        self.create_subscription(String, "/telemetry/raw", self._on_raw, 10)

        threading.Thread(
            target=lambda: app.run(host="0.0.0.0", port=FLASK_PORT, debug=False, use_reloader=False),
            daemon=True
        ).start()
        self.get_logger().info(f"Flask: http://0.0.0.0:{FLASK_PORT}/  (topic={STREAM_TOPIC})")

    def _on_img(self, msg: CompressedImage):
        global _latest_jpg, _latest_jpg_t
        _latest_jpg = bytes(msg.data)
        _latest_jpg_t = _now()

    def _on_pose(self, msg: PoseStamped):
        # also mirror ENU for convenience in JSON
        _tel["enu"]["x"] = float(msg.pose.position.x)
        _tel["enu"]["y"] = float(msg.pose.position.y)
        _tel["enu"]["z"] = float(msg.pose.position.z)
        _tel["ts"] = time.time()

    def _on_raw(self, msg: String):
        try:
            j = json.loads(msg.data)
            _tel.update({
                "ts": float(j.get("ts", time.time())),
                "lat": j.get("lat"), "lon": j.get("lon"), "alt": j.get("alt"),
                "yaw_deg": j.get("yaw_deg"), "pitch_deg": j.get("pitch_deg"), "roll_deg": j.get("roll_deg"),
            })
            if "enu" in j:
                _tel["enu"] = {
                    "x": j["enu"].get("x"), "y": j["enu"].get("y"), "z": j["enu"].get("z")
                }
        except Exception:
            pass

def main():
    rclpy.init()
    n = FlaskStream()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

