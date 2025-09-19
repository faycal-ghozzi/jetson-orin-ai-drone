import json
import time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

# Hard-coded YOLO config
# ==============================================================
PT_PATH       = "~/ai-drone-ws/src/ai_drone/ai_drone/models/yolov8n.pt"
CONF_TH       = 0.25
KEEP_CLASSES  = (0, 7)   # 0=person, 7=truck
IMG_SZ_GPU    = 416      # good balance on Jetson GPU
IMG_SZ_CPU    = 320      # lighter when CPU fallback
NMS_IOU       = 0.45
# ==============================================================


class YoloFromCompressed(Node):
    """
    Subscribe to /camera/image/compressed (JPEG), decode with cv2.imdecode,
    run YOLOv8 (Ultralytics), publish JSON on /detections_raw.
    No cv_bridge needed.
    """
    def __init__(self):
        super().__init__('yolo')

        self.backend_label = "none"
        self.model = None
        self.device = 'cpu'
        self.half = False
        self.imgsz = IMG_SZ_CPU

        try:
            import torch
            from ultralytics import YOLO
            self.torch = torch
            self.model = YOLO(PT_PATH if PT_PATH else 'yolov8n.pt')
            if torch.cuda.is_available():
                self.device = 0
                self.half = True
                self.imgsz = IMG_SZ_GPU
                self.backend_label = "ultra:gpu-fp16"
            else:
                self.device = 'cpu'
                self.half = False
                self.imgsz = IMG_SZ_CPU
                self.backend_label = "ultra:cpu"
            self.get_logger().info(f"YOLO backend: {self.backend_label} (imgsz={self.imgsz}) ✅")
        except Exception as e:
            self.get_logger().error(f"Ultralytics init failed: {e}")
            self.model = None

        self.sub = self.create_subscription(
            CompressedImage, '/camera/image/compressed', self._on_jpeg, 10
        )
        self.pub = self.create_publisher(String, '/detections_raw', 10)

    def _on_jpeg(self, msg: CompressedImage):
        """Decode JPEG → BGR np.array; run YOLO; publish detections as JSON."""
        if self.model is None:
            out = String()
            out.data = json.dumps({"header":{}, "backend":"none", "detections":[]})
            self.pub.publish(out)
            return

        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("cv2.imdecode returned None")
        except Exception as e:
            self.get_logger().warn(f"JPEG decode error: {e}")
            out = String()
            out.data = json.dumps({"header":{}, "backend":self.backend_label, "detections":[]})
            self.pub.publish(out)
            return

        t0 = time.time()
        dets = []
        try:
            res = self.model(
                source=img[..., ::-1],
                imgsz=self.imgsz,
                device=self.device,
                half=self.half,
                conf=CONF_TH,
                iou=NMS_IOU,
                verbose=False
            )[0]

            if getattr(res, 'boxes', None) is not None:
                xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
                conf = res.boxes.conf.cpu().numpy()
                cls  = res.boxes.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), sc, cl in zip(xyxy, conf, cls):
                    if sc < CONF_TH: 
                        continue
                    if KEEP_CLASSES and (cl not in KEEP_CLASSES): 
                        continue
                    dets.append({
                        "x1": int(x1), "y1": int(y1),
                        "x2": int(x2), "y2": int(y2),
                        "score": float(sc), "cls": int(cl)
                    })
        except Exception as e:
            self.get_logger().error(f"YOLO inference error: {e}")
            dets = []

        payload = {
            "header": {
                "stamp": f"{msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}",
                "frame_id": msg.header.frame_id
            },
            "backend": self.backend_label,
            "infer_ms": round((time.time() - t0) * 1000.0, 1),
            "detections": dets
        }
        out = String()
        out.data = json.dumps(payload, separators=(',',':'))
        self.pub.publish(out)


def main():
    rclpy.init()
    node = YoloFromCompressed()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

