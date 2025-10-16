import json
import time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import Conv
try:
    from ultralytics.nn.modules.block import C3k2
except Exception:
    C3k2 = None
from torch.nn import (Conv2d, Linear, BatchNorm2d, ReLU, SiLU, Sequential)

from ai_drone.config import PT_PATH, CONF_TH, KEEP_CLASSES, IMG_SZ_GPU, NMS_IOU

class YoloFromCompressed(Node):
    """
        - Subscribe to /camera/image/compressed.
        - run YOLOv11 (custom drone model). 
        - publish JSON on /detections_raw.
    """
    def __init__(self):
        super().__init__('yolo')

        self.backend_label = "none"
        self.model = None
        self.device = 'cuda'
        self.half = False
        self.imgsz = IMG_SZ_GPU

        self.pub = self.create_publisher(String, '/detections_raw', 10)
        self.sub = self.create_subscription(
            CompressedImage, '/camera/image/compressed', self._on_jpeg, 10
        )

        try:
            allowlist = [DetectionModel, Conv, Conv2d, Linear, BatchNorm2d, ReLU, SiLU, Sequential]
            if C3k2 is not None:
                allowlist.append(C3k2)
            torch.serialization.add_safe_globals(allowlist)

            self.torch = torch
            self.model = YOLO(PT_PATH)

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device required but not available")

            self.model.to('cuda')

            try:
                self.model.model.half()
                self.half = True
                self.backend_label = "ultra:gpu-fp16"
            except Exception:
                self.model.model.float()
                self.half = False
                self.backend_label = "ultra:gpu-fp32"

            self.get_logger().info(f"YOLO backend: {self.backend_label} (imgsz={self.imgsz}) ✅")

        except Exception as e:
            self.get_logger().error(f"Ultralytics init failed: {e}")
            self.model = None

    def _on_jpeg(self, msg: CompressedImage):
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

        H, W = img.shape[:2]

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

        header = {"stamp": "0.0", "frame_id": ""}
        try:
            header = {
                "stamp": f"{msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}",
                "frame_id": msg.header.frame_id
            }
        except Exception:
            pass

        payload = {
            "header": header,
            "backend": self.backend_label,
            "infer_ms": round((time.time() - t0) * 1000.0, 1),
            "src_w": int(W),
            "src_h": int(H),
            "detections": dets
        }
        out = String()
        out.data = json.dumps(payload, separators=(',',':'))
        self.pub.publish(out)

        self.get_logger().debug(f"Published {len(dets)} detections (infer_ms={payload['infer_ms']})")

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
