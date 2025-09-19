import json
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32MultiArray

# Tracker configuration
MAX_AGE = 10           # number of frames before a track is removed
DIST_TH = 100.0        # maximum distance to associate (pixels)
DIST_TH2 = DIST_TH ** 2

class Tracker(Node):
    """
    Tracker node:
    - subscribes to /detections_raw (YOLO detections JSON)
    - assigns persistent IDs to detections
    - publishes [x, y, id, ...] on /tracks_xy_id
    """

    def __init__(self):
        super().__init__('tracker')
        self.tracks = {}
        self.next_id = 1

        self.create_subscription(String, '/detections_raw', self._on_det, 10)
        self.pub = self.create_publisher(Int32MultiArray, '/tracks_xy_id', 10)

        self.get_logger().info("Tracker ready ✅ (listening to /detections_raw)")

    def _on_det(self, msg: String):
        """Receive JSON detections, update tracks, publish [x, y, id,...]."""
        try:
            dets = json.loads(msg.data).get("detections", [])
        except Exception as e:
            self.get_logger().warn(f"Failed to parse detections: {e}")
            dets = []

        centers = (
            np.array(
                [((d["x1"] + d["x2"]) / 2.0, (d["y1"] + d["y2"]) / 2.0) for d in dets],
                dtype=np.float32,
            )
            if dets else np.zeros((0, 2), dtype=np.float32)
        )

        ids = []
        
        for x, y in centers:
            best, bid = 1e12, None
            for tid, (tx, ty, age) in self.tracks.items():
                d2 = (tx - x) ** 2 + (ty - y) ** 2
                if d2 < best:
                    best, bid = d2, tid
            if best <= DIST_TH2 and bid is not None:
                self.tracks[bid] = [x, y, 0]
                ids.append(bid)
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = [x, y, 0]
                ids.append(tid)

        dead = []
        for tid, v in self.tracks.items():
            v[2] += 1
            if v[2] > MAX_AGE:
                dead.append(tid)
        for tid in dead:
            self.tracks.pop(tid, None)

        out = Int32MultiArray()
        out.data = []
        for (x, y), tid in zip(centers, ids):
            out.data.extend([int(x), int(y), int(tid)])
        self.pub.publish(out)


def main():
    rclpy.init()
    node = Tracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

