import json, numpy as np, cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String, Int32MultiArray
from cv_bridge import CvBridge

COCO = {0:"person",1:"bicycle",2:"car",3:"motorbike",5:"bus",7:"truck"}

class OverlayNode(Node):
    RANGE_PX_TH2 = 60.0**2

    def __init__(self):
        super().__init__("overlay")
        self.bridge = CvBridge()

        self.create_subscription(Image, "/camera/image", self._on_img, 10)
        self.create_subscription(String, "/detections_raw", self._on_det, 10)
        self.create_subscription(Int32MultiArray, "/tracks_xy_id", self._on_tracks, 10)
        self.create_subscription(String, "/reproj/range_hints", self._on_range, 10)
        self.create_subscription(String, "/overlay/lines", self._on_lines_system, 10)
        self.create_subscription(String, "/telemetry/overlay_lines", self._on_lines_telemetry, 10)

        self.pub_img = self.create_publisher(Image, "/camera/overlay", 10)
        self.pub_jpg = self.create_publisher(CompressedImage, "/camera/overlay/compressed", 10)

        self.last_dets = []
        self.last_ids  = []
        self.last_ranges = []
        self.lines_system = []
        self.lines_telem  = []

        self.get_logger().info("OverlayNode ready")

    def _on_det(self, msg: String):
        try: self.last_dets = json.loads(msg.data).get("detections", [])
        except Exception: self.last_dets = []

    def _on_tracks(self, msg: Int32MultiArray):
        v = msg.data; self.last_ids = [(v[i],v[i+1],v[i+2]) for i in range(0,len(v),3)]

    def _on_range(self, msg: String):
        try:
            hints = json.loads(msg.data).get("hints", [])
            self.last_ranges = [(float(h["cx"]), float(h["cy"]), float(h["range_m"])) for h in hints]
        except Exception:
            self.last_ranges = []

    def _on_lines_system(self, msg: String):
        try: self.lines_system = list(json.loads(msg.data).get("lines", []))[:6]
        except Exception: self.lines_system = []

    def _on_lines_telemetry(self, msg: String):
        try: self.lines_telem = list(json.loads(msg.data).get("lines", []))[:6]
        except Exception: self.lines_telem = []

    def _on_img(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        H, W = img.shape[:2]
        centers = []
        for d in self.last_dets:
            try:
                x1,y1,x2,y2 = map(int,(d["x1"],d["y1"],d["x2"],d["y2"]))
                sc = float(d.get("score",0)); c = int(d.get("cls",-1))
            except Exception:
                continue
            x1=max(0,min(W-1,x1)); x2=max(0,min(W-1,x2))
            y1=max(0,min(H-1,y1)); y2=max(0,min(H-1,y2))
            centers.append(((x1+x2)/2.0,(y1+y2)/2.0))

        # match ranges to boxes
        ranges = [None]*len(centers)
        if self.last_ranges and centers:
            c_np = np.array(centers, np.float32)
            for (cx,cy,rm) in self.last_ranges:
                d2 = ((c_np - np.array([cx,cy], np.float32))**2).sum(axis=1)
                j = int(np.argmin(d2))
                if d2[j] <= self.RANGE_PX_TH2:
                    if ranges[j] is None or d2[j] < ranges[j][1]:
                        ranges[j] = (rm, d2[j])

        font = cv2.FONT_HERSHEY_SIMPLEX

        # draw boxes
        for i, d in enumerate(self.last_dets):
            try:
                x1,y1,x2,y2 = map(int,(d["x1"],d["y1"],d["x2"],d["y2"]))
                sc = float(d.get("score",0)); c = int(d.get("cls",-1))
            except Exception:
                continue
            x1=max(0,min(W-1,x1)); x2=max(0,min(W-1,x2))
            y1=max(0,min(H-1,y1)); y2=max(0,min(H-1,y2))
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

            label = f"{COCO.get(c, f'cls:{c}')} {sc:.2f}"
            if ranges[i] is not None:
                label += f" {ranges[i][0]:.1f}m"
            (tw,th),_ = cv2.getTextSize(label, font, 0.6, 2)
            cv2.rectangle(img,(x1,max(0,y1-th-8)),(x1+tw+6,y1),(0,255,0),-1)
            cv2.putText(img,label,(x1+3,y1-6),font,0.6,(0,0,0),2,cv2.LINE_AA)

        # track IDs (optional)
        if self.last_ids and centers:
            det_np = np.array(centers, np.float32)
            for tx,ty,tid in self.last_ids:
                d2 = ((det_np - np.array([tx,ty],np.float32))**2).sum(axis=1)
                j = int(np.argmin(d2))
                cx,cy = int(centers[j][0]), int(centers[j][1])
                cv2.circle(img,(cx,cy),5,(255,200,0),-1)
                cv2.putText(img,f"ID {tid}",(cx+6,cy-6),font,0.6,(255,200,0),2,cv2.LINE_AA)

        # HUD lines (system + telemetry)
        y = 24
        for line in (self.lines_system + self.lines_telem)[:8]:
            cv2.putText(img, line, (10, y), font, 0.6, (255,255,255), 2, cv2.LINE_AA)
            y += 22

        # publish
        m = self.bridge.cv2_to_imgmsg(img, "bgr8"); m.header = msg.header
        self.pub_img.publish(m)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY),80])
        if ok:
            c = CompressedImage(); c.header = msg.header; c.format = "jpeg"; c.data = buf.tobytes()
            self.pub_jpg.publish(c)

def main():
    rclpy.init()
    n = OverlayNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

