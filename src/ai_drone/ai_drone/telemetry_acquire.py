import json, math, socket, threading, time
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Quaternion

# ----------------------- EDIT HERE (source of telemetry) ----------------------
TELEM_HOST = "192.168.55.100"   # your laptop that runs the small simulator
TELEM_PORT = 9001               # simulator TCP port
RECONNECT_SEC = 2.0
# -----------------------------------------------------------------------------

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
    Reads newline-delimited JSON over TCP from TELEM_HOST:TELEM_PORT.
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
        self._stop = False

        threading.Thread(target=self._loop, daemon=True).start()
        self.get_logger().info(f"TelemetryAcquire connecting to {TELEM_HOST}:{TELEM_PORT}")

    # ----------------------------- TCP loop -----------------------------------
    def _loop(self):
        while not self._stop:
            try:
                with socket.create_connection((TELEM_HOST, TELEM_PORT), timeout=5.0) as s:
                    s.settimeout(5.0)
                    self.get_logger().info("Telemetry TCP connected ✅")

                    buf = b""
                    while not self._stop:
                        chunk = s.recv(4096)
                        if not chunk:
                            raise ConnectionError("socket closed")
                        buf += chunk
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            self._handle_packet(line)
            except Exception as e:
                self.get_logger().warn(f"Telemetry connect failed: {e}")
                time.sleep(RECONNECT_SEC)

    # ----------------------------- Packet path --------------------------------
    def _handle_packet(self, raw: bytes):
        try:
            pkt = json.loads(raw.decode("utf-8", errors="ignore"))
        except Exception as e:
            self.get_logger().warn(f"bad JSON: {e}")
            return

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

def main():
    rclpy.init()
    n = TelemetryAcquire()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n._stop = True
    n.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

