
import json, math, threading, time
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Quaternion

from flask import Flask, request

FLASK_HOST = "0.0.0.0"  # Listen on all interfaces
FLASK_PORT = 8000       # Port to receive telemetry POSTs

def quat_from_euler(roll: float, pitch: float, yaw: float) -> Quaternion:
    """roll/pitch/yaw in radians → geometry_msgs/Quaternion"""
    sr, cr = math.sin(roll*0.5),  math.cos(roll*0.5)
    sp, cp = math.sin(pitch*0.5), math.cos(pitch*0.5)
    sy, cy = math.sin(yaw*0.5),   math.cos(yaw*0.5)
    q = Quaternion()
    q.w = cr*cp*cy + sr*sp*sy
    q.x = sr*cp*cy - cr*sp*sy
    q.y = cr*sp*cy + sr*cp*sy
    q.z = cr*cp*sy - sr*sp*cy
    return q

def meters_per_deg(lat_deg: float) -> Tuple[float, float]:
    """Very good local approximation of meters per degree at latitude."""
    lat = math.radians(lat_deg)
    m_per_deg_lat = 111132.92 - 559.82*math.cos(2*lat) + 1.175*math.cos(4*lat) - 0.0023*math.cos(6*lat)
    m_per_deg_lon = 111412.84*math.cos(lat) - 93.5*math.cos(3*lat) + 0.118*math.cos(5*lat)
    return m_per_deg_lat, m_per_deg_lon


class TelemetryAcquire(Node):
    """
    Receives telemetry via HTTP POST (Flask) from phone app.
    Expected fields (any extra are ignored):
      { "ts": 169..., "lat": xx.x, "lon": yy.y, "alt": 12.3,
        "yaw_deg": 123.4, "pitch_deg": 0.0 (opt), "roll_deg": 0.0 (opt) }

    Publishes:
      - /telemetry/raw  (std_msgs/String)      # normalized JSON with lat/lon/alt/yaw/… (+ ENU)
      - /telemetry/pose (geometry_msgs/PoseStamped)  # ENU pose for perception
      - /telemetry/overlay_lines (std_msgs/String)   # 1-2 lines for on-video HUD
    """

    def __init__(self):
        super().__init__("telemetry")
        self.pub_raw   = self.create_publisher(String, "/telemetry/raw", 10)
        self.pub_pose  = self.create_publisher(PoseStamped, "/telemetry/pose", 10)
        self.pub_lines = self.create_publisher(String, "/telemetry/overlay_lines", 10)

        self._home: Optional[Tuple[float,float]] = None  # (lat0, lon0)
        self._mdeg: Tuple[float,float] = (111320.0, 111320.0)  # lat/lon meters per degree

        # Start Flask server in background thread
        threading.Thread(target=self._run_flask, daemon=True).start()
        self.get_logger().info(f"TelemetryAcquire HTTP server on {FLASK_HOST}:{FLASK_PORT}")

    def _run_flask(self):
        flask_app = Flask(__name__)

        @flask_app.route('/data', methods=['POST'])
        def receive_data():
            data = request.get_data(as_text=True)
            self.get_logger().info(f"Received data: {data}")
            pkt = None
            try:
                js = json.loads(data)
                # If top-level is a list, find latest location and orientation
                if isinstance(js, list):
                    # Find latest location and orientation entries by 'time' field
                    loc = None
                    ori = None
                    for entry in reversed(js):
                        if isinstance(entry, dict):
                            if loc is None and entry.get('name') == 'location':
                                loc = entry
                            if ori is None and entry.get('name') == 'orientation':
                                ori = entry
                        if loc and ori:
                            break
                    if loc and 'values' in loc and isinstance(loc['values'], dict):
                        vloc = loc['values']
                        lat = float(vloc.get("latitude", 0.0))
                        lon = float(vloc.get("longitude", 0.0))
                        alt = float(vloc.get("altitude", 0.0))
                        yaw = float(vloc.get("course", 0.0))
                        ts = float(loc.get("time", time.time()))
                        # Default pitch/roll to 0
                        pitch = 0.0
                        roll = 0.0
                        if ori and 'values' in ori and isinstance(ori['values'], dict):
                            vori = ori['values']
                            # SensorLog/Logger may use radians for yaw/pitch/roll
                            # Convert to degrees if so
                            yaw = float(vori.get("yaw", yaw))
                            pitch = float(vori.get("pitch", 0.0))
                            roll = float(vori.get("roll", 0.0))
                            # If values are in radians (abs(yaw) < 2pi), convert to deg
                            if abs(yaw) < 7 and abs(pitch) < 7 and abs(roll) < 7:
                                yaw = math.degrees(yaw)
                                pitch = math.degrees(pitch)
                                roll = math.degrees(roll)
                        pkt = {
                            "lat": lat,
                            "lon": lon,
                            "alt": alt,
                            "yaw_deg": yaw,
                            "pitch_deg": pitch,
                            "roll_deg": roll,
                            "ts": ts
                        }
                # Fallback: try dict with 'payload' or flat dict
                elif isinstance(js, dict):
                    if 'payload' in js and isinstance(js['payload'], list):
                        loc = None
                        for entry in js['payload']:
                            if isinstance(entry, dict) and entry.get('name') == 'location':
                                loc = entry
                                break
                        if loc and 'values' in loc and isinstance(loc['values'], dict):
                            v = loc['values']
                            pkt = {
                                "lat": float(v.get("latitude", 0.0)),
                                "lon": float(v.get("longitude", 0.0)),
                                "alt": float(v.get("altitude", 0.0)),
                                "yaw_deg": float(v.get("course", 0.0)),
                                "pitch_deg": float(v.get("pitch", 0.0)),
                                "roll_deg": float(v.get("roll", 0.0)),
                                "ts": float(loc.get("ts", js.get("ts", time.time())))
                            }
                    elif all(k in js for k in ("lat", "lon")):
                        pkt = js
            except Exception as e:
                self.get_logger().warn(f"JSON parse error: {e}")
            if pkt:
                self._handle_packet(pkt)
                return '', 200
            else:
                self.get_logger().warn("Bad telemetry packet received")
                return 'Bad data', 400

        flask_app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, use_reloader=False, threaded=True)

    def _handle_packet(self, pkt: dict):
        try:
            ts   = float(pkt.get("ts", time.time()))
            lat  = float(pkt["lat"])
            lon  = float(pkt["lon"])
            alt  = float(pkt.get("alt", 0.0))
            yawd = float(pkt.get("yaw_deg", 0.0))
            pitchd = float(pkt.get("pitch_deg", 0.0))
            rolld  = float(pkt.get("roll_deg", 0.0))

            if self._home is None:
                self._home = (lat, lon)
                self._mdeg = meters_per_deg(lat)
            lat0, lon0 = self._home
            mlat, mlon = self._mdeg

            # local ENU (meters) using equirectangular approximation
            x = (lon - lon0) * mlon  # East
            y = (lat - lat0) * mlat  # North
            z = alt                  # Up (treat as height; refine later with AGL if available)

            # PoseStamped
            ps = PoseStamped()
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.header.frame_id = "map"
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = float(z)

            # orientation from roll/pitch/yaw if provided (degrees → rad)
            q = quat_from_euler(
                math.radians(rolld),
                math.radians(pitchd),
                math.radians(yawd)
            )
            ps.pose.orientation = q
            self.pub_pose.publish(ps)

            # Publish normalized raw JSON (+ ENU)
            out = {
                "ts": ts,
                "lat": lat, "lon": lon, "alt": alt,
                "yaw_deg": yawd, "pitch_deg": pitchd, "roll_deg": rolld,
                "enu": {"x": x, "y": y, "z": z}
            }
            m = String(); m.data = json.dumps(out, separators=(',', ':'))
            self.pub_raw.publish(m)

            # HUD lines for overlay
            line1 = f"LAT {lat:.6f}  LON {lon:.6f}"
            line2 = f"ALT {alt:.1f} m   YAW {yawd:.1f}°"
            hud = String(); hud.data = json.dumps({"lines":[line1, line2]}, separators=(',', ':'))
            self.pub_lines.publish(hud)
        except Exception as e:
            self.get_logger().warn(f"bad telemetry packet: {e}")


def main():
    rclpy.init()
    n = TelemetryAcquire()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

