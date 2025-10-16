import os, time, cv2
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

from ai_drone.config import VIDEO_FILE, LOOP_VIDEO, RESIZE_W, RESIZE_H, ROTATE_DEG, MIRROR, SPEED_FACTOR, JPEG_QUALITY, PUB_RAW_TOPIC, PUB_COMP_TOPIC


def _rotate(frame, deg):
    if deg == 0:   return frame
    if deg == 90:  return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180: return cv2.rotate(frame, cv2.ROTATE_180)
    if deg == 270: return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


class VideoFileNode(Node):
    """
        - Play demo video (ros2 launch bringup_file.launch.py)
    """

    def __init__(self):
        super().__init__("video_file")

        self.declare_parameter("file", VIDEO_FILE)
        self.declare_parameter("loop", LOOP_VIDEO)
        self.declare_parameter("resize_w", RESIZE_W)
        self.declare_parameter("resize_h", RESIZE_H)
        self.declare_parameter("rotate_deg", ROTATE_DEG)
        self.declare_parameter("mirror", MIRROR)
        self.declare_parameter("speed", SPEED_FACTOR)
        self.declare_parameter("jpeg_quality", JPEG_QUALITY)
        self.declare_parameter("pub_raw", PUB_RAW_TOPIC)
        self.declare_parameter("pub_comp", PUB_COMP_TOPIC)

        self.path         = self.get_parameter("file").get_parameter_value().string_value
        self.loop         = self.get_parameter("loop").get_parameter_value().bool_value
        self.resize_w     = int(self.get_parameter("resize_w").value)
        self.resize_h     = int(self.get_parameter("resize_h").value)
        self.rotate_deg   = int(self.get_parameter("rotate_deg").value)
        self.mirror       = bool(self.get_parameter("mirror").value)
        self.speed_factor = float(self.get_parameter("speed").value)
        self.jpeg_q       = int(self.get_parameter("jpeg_quality").value)
        pub_raw           = self.get_parameter("pub_raw").get_parameter_value().string_value
        pub_comp          = self.get_parameter("pub_comp").get_parameter_value().string_value

        self.bridge = CvBridge()
        self.pub_raw  = self.create_publisher(Image, pub_raw, 10)
        self.pub_comp = self.create_publisher(CompressedImage, pub_comp, 10)

        self.cap = cv2.VideoCapture(self.path, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            self.get_logger().error(f"Cannot open video file: {self.path}")
            raise SystemExit(1)

        src_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.src_fps = src_fps if src_fps and src_fps > 0 else 25.0
        self.dt = 0.0 if self.speed_factor <= 0 else (1.0 / self.src_fps) / self.speed_factor

        self.get_logger().info(
            f"VideoFileNode ready ✅ (file='{self.path}', fps≈{self.src_fps:.2f}, "
            f"resize={self.resize_w}x{self.resize_h}, rotate={self.rotate_deg}°, "
            f"mirror={self.mirror}, speed={self.speed_factor}x)"
        )

        period = 0.0 if self.dt == 0.0 else self.dt
        self.timer = self.create_timer(max(0.0, period), self._tick)

        self.last_pub_time = time.monotonic()

    def _read_frame(self):
        ok, frame = self.cap.read()
        if ok:
            return frame

        if self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok2, frame2 = self.cap.read()
            if ok2:
                return frame2
        return None

    def _tick(self):
        if self.dt == 0.0:
            now = time.monotonic()
            if now - self.last_pub_time < 0.0005:
                return
            self.last_pub_time = now

        frame = self._read_frame()
        if frame is None:
            time.sleep(0.01)
            return

        if self.rotate_deg:
            frame = _rotate(frame, self.rotate_deg)
        if self.mirror:
            frame = cv2.flip(frame, 1)
        if self.resize_w > 0 and self.resize_h > 0:
            frame = cv2.resize(frame, (self.resize_w, self.resize_h), interpolation=cv2.INTER_LINEAR)

        stamp = Clock().now().to_msg()
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = stamp
        msg.header.frame_id = "camera"

        self.pub_raw.publish(msg)

        ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_q])
        if ok:
            c = CompressedImage()
            c.header.stamp = stamp
            c.header.frame_id = "camera"
            c.format = "jpeg"
            c.data = enc.tobytes()
            self.pub_comp.publish(c)

    def destroy_node(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    n = VideoFileNode()
    try:
        rclpy.spin(n)
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
