#!/usr/bin/env python3
import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
from cv_bridge import CvBridge

from ultralytics import YOLO


class Yolo3DGuiPickBest(Node):
    def __init__(self):
        super().__init__("yolo_3d_gui")

        # ================= PARAMETERS =================
        self.declare_parameter("rgb_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("info_topic", "/camera/color/camera_info")

        self.declare_parameter("model_path", "/home/asus/Downloads/best(1).pt")
        self.declare_parameter("target_class", "wood")

        # YOLO params
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.4)
        self.declare_parameter("max_det", 20)
        self.declare_parameter("imgsz", 480)

        # Depth params
        self.declare_parameter("depth_scale", 0.001)   # 16UC1 mm->m
        self.declare_parameter("roi_half", 2)          # median ROI = (2*roi_half+1)
        self.declare_parameter("search_radius", 25)    # px, search near pick point
        self.declare_parameter("search_step", 3)       # px

        # Pick point inside bbox (0..1 from top to bottom)
        self.declare_parameter("pick_v_ratio", 0.35)   # 0.35 ~ 1/3 from top

        self.rgb_topic   = self.get_parameter("rgb_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.info_topic  = self.get_parameter("info_topic").value

        self.model_path  = self.get_parameter("model_path").value
        self.target_cls  = str(self.get_parameter("target_class").value)

        self.conf        = float(self.get_parameter("conf").value)
        self.iou         = float(self.get_parameter("iou").value)
        self.max_det     = int(self.get_parameter("max_det").value)
        self.imgsz       = int(self.get_parameter("imgsz").value)

        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.roi_half    = int(self.get_parameter("roi_half").value)
        self.search_radius = int(self.get_parameter("search_radius").value)
        self.search_step = int(self.get_parameter("search_step").value)

        self.pick_v_ratio = float(self.get_parameter("pick_v_ratio").value)

        # ================= YOLO =================
        self.get_logger().info(f"Load YOLO model: {self.model_path}")
        self.model = YOLO(self.model_path)

        # ================= ROS =================
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self.bridge = CvBridge()

        self.sub_rgb = self.create_subscription(Image, self.rgb_topic, self.rgb_cb, qos)
        self.sub_depth = self.create_subscription(Image, self.depth_topic, self.depth_cb, qos)
        self.sub_info = self.create_subscription(CameraInfo, self.info_topic, self.info_cb, 10)

        self.pub_point = self.create_publisher(PointStamped, "/yolo/point_3d", 10)
        self.pub_info  = self.create_publisher(String, "/yolo/object_info", 10)

        # ================= BUFFERS =================
        self.depth_img = None
        self.fx = self.fy = self.cx = self.cy = None

        # ================= FPS =================
        self.last_t = time.time()
        self.fps = 0.0

        self.get_logger().info("YOLO 3D GUI (pick best conf) READY")

    # ==================================================
    def info_cb(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def depth_cb(self, msg: Image):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    # ==================================================
    def _median_depth_at(self, u: int, v: int):
        """Median depth (meters) around (u,v). None if invalid/0."""
        if self.depth_img is None:
            return None

        h, w = self.depth_img.shape[:2]
        r = self.roi_half
        if u < r or v < r or u >= w - r or v >= h - r:
            return None

        patch = self.depth_img[v - r:v + r + 1, u - r:u + r + 1]
        valid = patch[patch > 0]
        if valid.size == 0:
            return None

        Z = float(np.median(valid)) * self.depth_scale
        # lọc giá trị vô lý (tuỳ bạn chỉnh)
        if Z <= 0.05 or Z > 5.0:
            return None
        return Z

    def find_depth_near_uv(self, u0: int, v0: int):
        """
        Find depth near (u0,v0) to avoid holes.
        Returns: Z (meters) or None.
        """
        Z = self._median_depth_at(u0, v0)
        if Z is not None:
            return Z

        step = self.search_step
        radius = self.search_radius

        for r in range(step, radius + 1, step):
            candidates = [
                (u0 + r, v0), (u0 - r, v0), (u0, v0 + r), (u0, v0 - r),
                (u0 + r, v0 + r), (u0 + r, v0 - r), (u0 - r, v0 + r), (u0 - r, v0 - r)
            ]
            for u, v in candidates:
                Z = self._median_depth_at(u, v)
                if Z is not None:
                    return Z
        return None

    # ==================================================
    def rgb_cb(self, msg: Image):
        if self.fx is None or self.depth_img is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        annotated = frame.copy()

        # YOLO inference
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
        best = None  # (x1,y1,x2,y2, conf, cls_id)

        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), cls_id, c in zip(boxes, classes, confs):
                cls_name = r.names[cls_id]
                if cls_name != self.target_cls:
                    continue
                if best is None or c > best[4]:
                    best = (x1, y1, x2, y2, float(c), int(cls_id))

        if best is not None:
            x1, y1, x2, y2, best_conf, cls_id = best
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # draw only BEST bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # pick point: center in X, and 1/3 from top in Y
            u = int((x1 + x2) / 2)
            v = int(y1 + self.pick_v_ratio * (y2 - y1))

            # find depth near pick point
            Z = self.find_depth_near_uv(u, v)

            label = f"{self.target_cls} conf:{best_conf:.2f}"

            if Z is not None:
                X = (u - self.cx) * Z / self.fx
                Y = (v - self.cy) * Z / self.fy

                label += f"  X:{X:.2f} Y:{Y:.2f} Z:{Z:.2f}m"

                # publish ONE point
                pt = PointStamped()
                pt.header = msg.header
                pt.header.frame_id = "camera_color_optical_frame"
                pt.point.x = float(X)
                pt.point.y = float(Y)
                pt.point.z = float(Z)
                self.pub_point.publish(pt)

                info = String()
                info.data = f"PICK {self.target_cls} conf={best_conf:.2f} XYZ=({X:.3f},{Y:.3f},{Z:.3f}) u={u} v={v}"
                self.pub_info.publish(info)

                # mark pick point
                cv2.circle(annotated, (u, v), 7, (0, 0, 255), -1)
            else:
                label += "  Z=?"
                cv2.circle(annotated, (u, v), 7, (0, 0, 255), 2)

            cv2.putText(
                annotated, label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        else:
            cv2.putText(
                annotated, f"No '{self.target_cls}' detected",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
            )

        # FPS
        now = time.time()
        dt = now - self.last_t
        self.last_t = now
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

        cv2.putText(
            annotated, f"FPS: {self.fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        cv2.imshow("YOLO 3D GUI - PICK BEST", annotated)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = Yolo3DGuiPickBest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()