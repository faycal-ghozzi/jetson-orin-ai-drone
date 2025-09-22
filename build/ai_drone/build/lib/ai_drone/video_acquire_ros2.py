import time, cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge


# Video params
RTSP_URL     = "rtsp://admin:1234@10.191.67.102:8554/live"
VIDEO_WIDTH  = 640
VIDEO_HEIGHT = 480
VIDEO_FPS    = 24
VIDEO_LATENCY = 0
VIDEO_USE_HW = True
VIDEO_ROTATE = 270
JPEG_QUALITY = 80

def gst_pipeline(rtsp: str, use_hw: bool, width: int, height: int, latency: int) -> str:
    """GStreamer pipeline for Jetson: NVDEC → BGRx → videoconvert → BGR → appsink."""
    scale_caps = "" if width <= 0 or height <= 0 else f",width={width},height={height}"
    if use_hw:
        return (
            f'rtspsrc location="{rtsp}" latency={latency} protocols=tcp drop-on-latency=true ! '
            f'rtph264depay ! h264parse ! nvv4l2decoder ! '
            f'nvvidconv ! video/x-raw,format=BGRx{scale_caps} ! '
            f'videoconvert ! video/x-raw,format=BGR ! '
            f'appsink drop=true max-buffers=1 sync=false'
        )
    else:
        return (
            f'rtspsrc location="{rtsp}" latency={latency} protocols=tcp drop-on-latency=true ! '
            f'rtph264depay ! h264parse ! avdec_h264 ! '
            f'videoconvert ! video/x-raw,format=BGR{scale_caps} ! '
            f'appsink drop=true max-buffers=1 sync=false'
        )


class RtspNode(Node):
    """ROS2 node: acquire RTSP frames and publish raw + compressed images."""

    def __init__(self):
        super().__init__('video')

        self.rtsp   = RTSP_URL
        self.w      = VIDEO_WIDTH
        self.h      = VIDEO_HEIGHT
        self.fps    = VIDEO_FPS
        self.lat    = VIDEO_LATENCY
        self.use_hw = VIDEO_USE_HW
        self.rotate = VIDEO_ROTATE
        self.period = 1.0 / max(1, self.fps)

        self.bridge = CvBridge()
        self.pub_raw = self.create_publisher(Image, '/camera/image', 10)
        self.pub_jpg = self.create_publisher(CompressedImage, '/camera/image/compressed', 10)

        self.cap = None
        self.timer = self.create_timer(self.period, self.loop)

        self.get_logger().info(
            f"Video node ready (RTSP={self.rtsp}, HW={self.use_hw}, "
            f"rotate={self.rotate}°, {self.w}x{self.h}@{self.fps}fps)"
        )

    def _open(self):
        """Open RTSP pipeline via GStreamer."""
        pipe = gst_pipeline(self.rtsp, self.use_hw, self.w, self.h, self.lat)
        self.cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        if self.cap.isOpened():
            backend = "HW (nvv4l2decoder)" if self.use_hw else "SW (avdec_h264)"
            self.get_logger().info(f"RTSP opened successfully via GStreamer {backend}")
        else:
            self.get_logger().error("Failed to open RTSP pipeline")

    def loop(self):
        """Grab a frame, rotate if needed, publish raw + compressed."""
        if self.cap is None or not self.cap.isOpened():
            self._open()
            time.sleep(0.1)
            return

        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn("⚠️ Lost stream, reconnecting…")
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
            return

        if self.rotate == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotate == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotate == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.pub_raw.publish(msg)

        ok, enc = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if ok:
            c = CompressedImage()
            c.header = msg.header
            c.format = 'jpeg'
            c.data = enc.tobytes()
            self.pub_jpg.publish(c)


def main():
    rclpy.init()
    node = RtspNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

