#!/usr/bin/env python3
import time
import math
import json
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge

from ultralytics import YOLO


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def gamma_correction(bgr: np.ndarray, gamma: float) -> np.ndarray:
    inv = 1.0 / max(gamma, 1e-6)
    table = (np.arange(256) / 255.0) ** inv
    table = np.clip(table * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(bgr, table)


def poly_area(pts: np.ndarray) -> float:
    return float(cv2.contourArea(pts.reshape(-1, 1, 2).astype(np.int32)))


def order_quad(quad: np.ndarray) -> np.ndarray:
    """Order quad points as [tl, tr, br, bl]."""
    pts = quad.astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def is_parallelogram(pts: np.ndarray, parallel_cos_tol: float = 0.985) -> bool:
    """
    Check opposite sides roughly parallel:
      v01 || v23 and v12 || v30
    """
    pts = pts.astype(np.float32)

    def unit(v):
        n = np.linalg.norm(v)
        return v / (n + 1e-6)

    v01 = unit(pts[1] - pts[0])
    v12 = unit(pts[2] - pts[1])
    v23 = unit(pts[3] - pts[2])
    v30 = unit(pts[0] - pts[3])

    cos1 = abs(float(np.dot(v01, v23)))
    cos2 = abs(float(np.dot(v12, v30)))
    return (cos1 >= parallel_cos_tol) and (cos2 >= parallel_cos_tol)


class YoloParallelogramMulti(Node):
    def __init__(self):
        super().__init__("yolo_parallelogram_multi")

        # ===================== params =====================
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("model_path", "/home/asus/Downloads/best(1).pt")
        self.declare_parameter("target_class", "wood")

        # ✅ mặc định conf = 0.5 theo yêu cầu
        self.declare_parameter("conf", 0.5)
        self.declare_parameter("iou", 0.4)
        self.declare_parameter("max_det", 30)
        self.declare_parameter("imgsz", 640)

        self.declare_parameter("gamma", 1.5)
        self.declare_parameter("blur_ksize", 5)

        self.declare_parameter("canny1", 50)
        self.declare_parameter("canny2", 150)

        self.declare_parameter("hough_thresh", 60)
        self.declare_parameter("hough_min_len", 30)
        self.declare_parameter("hough_max_gap", 10)

        self.declare_parameter("min_quad_area", 700)
        self.declare_parameter("parallel_cos_tol", 0.985)

        # draw
        self.declare_parameter("draw_yolo_bbox", True)
        self.declare_parameter("draw_hough_lines", False)  # thường để False cho đỡ rối
        self.declare_parameter("limit_hough_draw", 40)

        # publish
        self.declare_parameter("publish_json", True)  # publish list targets dạng json string

        self.image_topic = self.get_parameter("image_topic").value
        self.model_path = self.get_parameter("model_path").value
        self.target_class = str(self.get_parameter("target_class").value)

        self.conf = float(self.get_parameter("conf").value)
        self.iou = float(self.get_parameter("iou").value)
        self.max_det = int(self.get_parameter("max_det").value)
        self.imgsz = int(self.get_parameter("imgsz").value)

        self.gamma = float(self.get_parameter("gamma").value)
        self.blur_ksize = int(self.get_parameter("blur_ksize").value)

        self.canny1 = int(self.get_parameter("canny1").value)
        self.canny2 = int(self.get_parameter("canny2").value)

        self.hough_thresh = int(self.get_parameter("hough_thresh").value)
        self.hough_min_len = int(self.get_parameter("hough_min_len").value)
        self.hough_max_gap = int(self.get_parameter("hough_max_gap").value)

        self.min_quad_area = int(self.get_parameter("min_quad_area").value)
        self.parallel_cos_tol = float(self.get_parameter("parallel_cos_tol").value)

        self.draw_yolo_bbox = bool(self.get_parameter("draw_yolo_bbox").value)
        self.draw_hough_lines = bool(self.get_parameter("draw_hough_lines").value)
        self.limit_hough_draw = int(self.get_parameter("limit_hough_draw").value)

        self.publish_json = bool(self.get_parameter("publish_json").value)

        # ===================== YOLO =====================
        self.get_logger().info(f"Load YOLO: {self.model_path}")
        self.model = YOLO(self.model_path)

        # ===================== ROS =====================
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, self.image_topic, self.cb, qos)

        self.pub_targets = self.create_publisher(String, "/yolo/targets", 10)
        self.pub_center = self.create_publisher(PointStamped, "/yolo/center_px", 10)  # publish từng target (optional)

        self.last_t = time.time()
        self.fps = 0.0

        cv2.namedWindow("YOLO MULTI WOOD - PARALLELOGRAM", cv2.WINDOW_NORMAL)
        self.get_logger().info("READY: multi targets")

    def find_largest_parallelogram(self, roi_bgr: np.ndarray):
        """
        Return:
          best_quad (4,2) in ROI coords, center (cx,cy), and optional lines (for debug)
        """
        # 1) convert to grayscale
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        # 2) gamma correction on grayscale
        gray = gamma_correction(gray, self.gamma)

        # 3) blur
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 4) Otsu threshold
        _, bw = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # 4) edges + HoughLinesP (để vẽ line nếu cần)
        edges = cv2.Canny(bw, self.canny1, self.canny2)

        lines = None
        if self.draw_hough_lines:
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180,
                threshold=self.hough_thresh,
                minLineLength=self.hough_min_len,
                maxLineGap=self.hough_max_gap
            )

        # 5) contour -> quad -> parallelogram
        kernel = np.ones((3, 3), np.uint8)
        bw2 = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

        cnts, _ = cv2.findContours(bw2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_quad = None
        best_area = 0.0
        best_center = None

        for c in cnts:
            area = cv2.contourArea(c)
            if area < self.min_quad_area:
                continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) != 4:
                continue
            if not cv2.isContourConvex(approx):
                continue

            quad = approx.reshape(4, 2).astype(np.float32)
            quad = order_quad(quad)

            if not is_parallelogram(quad, self.parallel_cos_tol):
                continue

            a = poly_area(quad)
            if a > best_area:
                best_area = a
                best_quad = quad
                best_center = (int(np.mean(quad[:, 0])), int(np.mean(quad[:, 1])))

        return best_quad, best_center, lines

    def cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        vis = frame.copy()
        H, W = frame.shape[:2]

        # YOLO predict (nhiều bbox)
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            imgsz=self.imgsz,
            device="cpu",
            verbose=False
        )
        r = results[0]

        boxes_out = []
        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), cid, cf in zip(boxes, cls, confs):
                if r.names[cid] != self.target_class:
                    continue
                boxes_out.append((int(x1), int(y1), int(x2), int(y2), float(cf)))

        # sort theo conf giảm dần (để ID ổn định hơn)
        boxes_out.sort(key=lambda t: t[4], reverse=True)

        all_targets = []  # list target hợp lệ

        for (x1, y1, x2, y2, cf) in boxes_out:
            x1 = clamp(x1, 0, W - 1)
            x2 = clamp(x2, 0, W - 1)
            y1 = clamp(y1, 0, H - 1)
            y2 = clamp(y2, 0, H - 1)
            if x2 <= x1 or y2 <= y1:
                continue

            if self.draw_yolo_bbox:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 1)
                cv2.putText(vis, f"{self.target_class} {cf:.2f}",
                            (x1, max(20, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            quad, center, lines = self.find_largest_parallelogram(roi)
            if quad is None or center is None:
                continue

            # debug: vẽ hough trong ROI (nếu bật)
            if self.draw_hough_lines and lines is not None:
                for l in lines[: self.limit_hough_draw]:
                    xa, ya, xb, yb = l[0]
                    cv2.line(vis, (x1 + xa, y1 + ya), (x1 + xb, y1 + yb), (255, 0, 0), 1)

            quad_g = quad.copy()
            quad_g[:, 0] += x1
            quad_g[:, 1] += y1

            cx_g = center[0] + x1
            cy_g = center[1] + y1

            area_g = poly_area(quad_g)

            all_targets.append({
                "conf": cf,
                "area": area_g,
                "center": (cx_g, cy_g),
                "quad": quad_g
            })

        # ===== VẼ + PUBLISH TẤT CẢ target hợp lệ =====
        if len(all_targets) == 0:
            cv2.putText(vis, "No valid parallelogram found", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            for i, t in enumerate(all_targets):
                quad_g = t["quad"].reshape(-1, 1, 2).astype(np.int32)
                cx_g, cy_g = t["center"]
                cf = t["conf"]
                area = t["area"]

                cv2.polylines(vis, [quad_g], True, (0, 255, 0), 2)
                cv2.circle(vis, (cx_g, cy_g), 6, (0, 0, 255), -1)
                cv2.putText(vis, f"ID:{i} conf:{cf:.2f} A:{int(area)}",
                            (cx_g + 5, cy_g - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # publish từng center (optional) — robot node có thể subscribe và pick theo ID trong /yolo/targets
                pt = PointStamped()
                pt.header = msg.header
                pt.header.frame_id = "image"
                pt.point.x = float(cx_g)
                pt.point.y = float(cy_g)
                pt.point.z = float(i)  # nhét ID vào z cho nhanh (nếu không muốn thì bỏ)
                self.pub_center.publish(pt)

            if self.publish_json:
                payload = []
                for i, t in enumerate(all_targets):
                    quad_list = t["quad"].astype(int).tolist()  # [[x,y],[x,y]...]
                    payload.append({
                        "id": i,
                        "conf": round(float(t["conf"]), 3),
                        "area": round(float(t["area"]), 1),
                        "center_px": [int(t["center"][0]), int(t["center"][1])],
                        "quad_px": quad_list
                    })
                s = String()
                s.data = json.dumps(payload)
                self.pub_targets.publish(s)

        # FPS
        now = time.time()
        dt = now - self.last_t
        self.last_t = now
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        cv2.putText(vis, f"FPS {self.fps:.1f}  conf>={self.conf:.2f}  targets:{len(all_targets)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("YOLO MULTI WOOD - PARALLELOGRAM", vis)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = YoloParallelogramMulti()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()