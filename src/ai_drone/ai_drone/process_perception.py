import json, math, time
from typing import Dict, Any, List, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

IMG_W = 640
IMG_H = 480
HFOV_DEG = 70.0
VFOV_DEG = 55.0

MIN_TAN = 1e-3
MAX_RANGE_M = 300.0
STATUS_HZ = 5.0

def euler_from_quat(x: float, y: float, z: float, w: float) -> Tuple[float,float,float]:
    """Return roll, pitch, yaw (rad) from quaternion (ENU)."""
    # roll (x-axis)
    sinr_cosp = 2.0 * (w*x + y*z)
    cosr_cosp = 1.0 - 2.0 * (x*x + y*y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis)
    sinp = 2.0 * (w*y - z*x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi/2, sinp)
    else:
        pitch = math.asin(sinp)
    # yaw (z-axis)
    siny_cosp = 2.0 * (w*z + x*y)
    cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

def fx_fy_from_fov(img_w: int, img_h: int, hfov_deg: float, vfov_deg: float):
    fx = (img_w / 2.0) / math.tan(math.radians(hfov_deg) * 0.5)
    fy = (img_h / 2.0) / math.tan(math.radians(vfov_deg) * 0.5)
    return fx, fy

class PerceptionProcessor(Node):
    """
    /detections_raw (String JSON) + /telemetry/pose (PoseStamped) → ground ENU points
    Publishes:
      - /detections/ground_enu (String JSON)
      - /reproj/range_hints (String JSON)  # for overlay to show "<range> m"
      - /overlay/lines (String JSON)       # status line on the video
    """
    def __init__(self):
        super().__init__("process_perception")

        self.pub_ground  = self.create_publisher(String, "/detections/ground_enu", 10)
        self.pub_range   = self.create_publisher(String, "/reproj/range_hints", 10)
        self.pub_overlay = self.create_publisher(String, "/overlay/lines", 10)

        self.create_subscription(String, "/detections_raw", self.on_dets, 20)
        self.create_subscription(PoseStamped, "/telemetry/pose", self.on_pose, 20)

        self.fx, self.fy = fx_fy_from_fov(IMG_W, IMG_H, HFOV_DEG, VFOV_DEG)
        self.cx, self.cy = IMG_W * 0.5, IMG_H * 0.5

        self.pose_xy = (0.0, 0.0)
        self.alt_m   = 1.2          # fallback camera height if telemetry missing
        self.yaw_rad = 0.0
        self.pitch_rad = math.radians(10.0)  # fallback: pitched slightly down

        self._last_status = 0.0
        self.get_logger().info("PerceptionProcessor ready ✅ (ground reprojection)")

    # ---------------------------- callbacks -----------------------------------
    def on_pose(self, msg: PoseStamped):
        self.pose_xy = (float(msg.pose.position.x), float(msg.pose.position.y))
        self.alt_m   = float(msg.pose.position.z)

        x,y,z,w = msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w
        roll, pitch, yaw = euler_from_quat(x,y,z,w)
        self.yaw_rad = yaw
        # We want "pitch_down" positive when camera looks down; pitch is +up in ENU, so:
        self.pitch_rad = -pitch

    def on_dets(self, msg: String):
        t0 = time.time()
        try:
            dets = json.loads(msg.data).get("detections", [])
        except Exception:
            return

        points: List[Dict[str, Any]] = []
        hints: List[Dict[str, Any]] = []

        for d in dets:
            try:
                x1,y1,x2,y2 = float(d["x1"]), float(d["y1"]), float(d["x2"]), float(d["y2"])
                score = float(d.get("score", 0.0))
                clsid = int(d.get("cls", -1))
            except Exception:
                continue

            u = 0.5*(x1+x2)   # bottom center for ground contact
            v = y2

            du = (u - self.cx) / self.fx
            dv = (v - self.cy) / self.fy

            az = self.yaw_rad + math.atan2(du, 1.0)

            elev_center = -self.pitch_rad
            elev = elev_center + math.atan2(dv, 1.0)

            tan_neg_elev = math.tan(-elev)
            if tan_neg_elev < MIN_TAN:
                continue

            R = self.alt_m / max(MIN_TAN, tan_neg_elev)
            if not (0.0 < R < MAX_RANGE_M):
                continue

            xg = self.pose_xy[0] + R * math.cos(az)
            yg = self.pose_xy[1] + R * math.sin(az)

            points.append({"x": float(xg), "y": float(yg),
                           "cls": clsid, "score": score, "range_m": float(R)})
            hints.append({"cx": float(u), "cy": float(v), "range_m": float(R)})

        # publish results
        self.pub_ground.publish(String(data=json.dumps({"points": points, "ts": time.time()}, separators=(',',':'))))
        self.pub_range.publish(String(data=json.dumps({"hints": hints, "ts": time.time()}, separators=(',',':'))))

        # status line on the video
        now = time.time()
        if now - self._last_status > 1.0 / STATUS_HZ:
            dt_ms = (now - t0)*1000.0
            status = f"reproj: {len(points)}/{len(dets)}  alt:{self.alt_m:.1f}m  pitch↓:{math.degrees(self.pitch_rad):.1f}°  {dt_ms:.1f}ms"
            self.pub_overlay.publish(String(data=json.dumps({"lines":[status]}, separators=(',',':'))))
            self._last_status = now

def main():
    rclpy.init()
    n = PerceptionProcessor()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

