#!/usr/bin/env python3
import math
import time
from typing import Optional, List, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose2D

from cv_bridge import CvBridge

from ultralytics import YOLO


def clamp_angle_deg(a: float) -> float:
    """Normalize angle to [-180, 180)."""
    a = (a + 180.0) % 360.0 - 180.0
    return a


def obb_poly_area(poly: np.ndarray) -> float:
    """poly: (4,2) float"""
    return float(cv2.contourArea(poly.astype(np.float32)))


def draw_arrow(img, center: Tuple[int, int], yaw_deg: float, length: int = 50,
               color=(255, 0, 0), thickness: int = 2):
    """Draw yaw arrow in image coords (x right, y down). yaw_deg CCW from +x."""
    cx, cy = center
    ang = math.radians(yaw_deg)
    x2 = int(cx + length * math.cos(ang))
    y2 = int(cy + length * math.sin(ang))
    cv2.arrowedLine(img, (cx, cy), (x2, y2), color, thickness, tipLength=0.25)


class YoloOBBCenterYaw(Node):
    def __init__(self):
        super().__init__('yolo_obb_center_yaw')

        # Parameters
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('model_path', '')
        self.declare_parameter('target_class', 'wood')
        self.declare_parameter('conf', 0.5)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('max_det', 20)
        self.declare_parameter('publish_pose', True)
        self.declare_parameter('pose_topic', '/wood_pose2d')
        self.declare_parameter('pick_one', False)  # True: only best (highest conf)

        self.image_topic  = self.get_parameter('image_topic').get_parameter_value().string_value
        self.model_path   = self.get_parameter('model_path').get_parameter_value().string_value
        self.target_class = self.get_parameter('target_class').get_parameter_value().string_value
        self.conf_th      = self.get_parameter('conf').get_parameter_value().double_value
        self.imgsz        = int(self.get_parameter('imgsz').get_parameter_value().integer_value or 640)
        self.max_det      = int(self.get_parameter('max_det').get_parameter_value().integer_value or 20)
        self.publish_pose = self.get_parameter('publish_pose').get_parameter_value().bool_value
        self.pose_topic   = self.get_parameter('pose_topic').get_parameter_value().string_value
        self.pick_one     = self.get_parameter('pick_one').get_parameter_value().bool_value

        if not self.model_path:
            raise RuntimeError("model_path is empty. Pass -p model_path:=/path/to/best.pt")

        self.get_logger().info(f"Load YOLO-OBB model: {self.model_path}")
        self.model = YOLO(self.model_path)

        # map class name -> id
        self.names = self.model.names  # dict or list
        self.target_id = None
        if isinstance(self.names, dict):
            for k, v in self.names.items():
                if v == self.target_class:
                    self.target_id = int(k)
                    break
        else:  # list
            for i, v in enumerate(self.names):
                if v == self.target_class:
                    self.target_id = i
                    break

        if self.target_id is None:
            self.get_logger().warn(f"target_class '{self.target_class}' not in model names {self.names}. "
                                   f"Will not filter by class id.")
        else:
            self.get_logger().info(f"Target class '{self.target_class}' id={self.target_id}")

        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image, self.image_topic, self.image_cb, qos_profile_sensor_data
        )

        self.pose_pub = None
        if self.publish_pose:
            self.pose_pub = self.create_publisher(Pose2D, self.pose_topic, 10)

        # simple FPS
        self._t_last = time.time()
        self._fps = 0.0

        # display
        self.win = "YOLO OBB (center + yaw)"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)

        self.get_logger().info(f"Subscribed: {self.image_topic}")

    def image_cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = frame.shape[:2]

        # YOLO inference (OBB)
        # Ultralytics will return Results with .obb for OBB models
        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf_th,
            max_det=self.max_det,
            verbose=False
        )

        out = frame.copy()

        # FPS
        t = time.time()
        dt = t - self._t_last
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
        self._t_last = t

        picks = []  # list of (conf, cx, cy, yaw_deg, poly4)

        r0 = results[0]
        if getattr(r0, "obb", None) is None or r0.obb is None:
            cv2.putText(out, "Model output has no OBB. Check you are using YOLOv8-OBB weights.",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            obb = r0.obb

            # Ultralytics OBB commonly provides:
            # - obb.cls, obb.conf
            # - obb.xyxyxyxy : (N, 4, 2)
            # - obb.xywhr : (N, 5) [cx, cy, w, h, r] where r in radians
            cls = obb.cls.cpu().numpy() if obb.cls is not None else None
            conf = obb.conf.cpu().numpy() if obb.conf is not None else None

            poly = None
            if hasattr(obb, "xyxyxyxy") and obb.xyxyxyxy is not None:
                poly = obb.xyxyxyxy.cpu().numpy()  # (N,4,2)
            xywhr = None
            if hasattr(obb, "xywhr") and obb.xywhr is not None:
                xywhr = obb.xywhr.cpu().numpy()  # (N,5) r rad

            N = 0
            if poly is not None:
                N = poly.shape[0]
            elif xywhr is not None:
                N = xywhr.shape[0]

            for i in range(N):
                if conf is None:
                    c = 1.0
                else:
                    c = float(conf[i])

                if c < self.conf_th:
                    continue

                if cls is not None and self.target_id is not None:
                    if int(cls[i]) != int(self.target_id):
                        continue

                # Get polygon 4 points
                if poly is not None:
                    p4 = poly[i].astype(np.float32)  # (4,2)
                    # center from polygon
                    cx = float(np.mean(p4[:, 0]))
                    cy = float(np.mean(p4[:, 1]))
                    # yaw from minAreaRect on polygon (robust)
                    rect = cv2.minAreaRect(p4.astype(np.float32))
                    (rcx, rcy), (rw, rh), ang = rect  # ang in degrees
                    # OpenCV angle convention: [-90,0), depends on w/h
                    # Normalize yaw to represent LONG side direction:
                    if rw < rh:
                        yaw = ang + 90.0
                    else:
                        yaw = ang
                    yaw = clamp_angle_deg(yaw)
                    p4i = p4
                else:
                    cx, cy, rw, rh, rrad = xywhr[i].tolist()
                    # Convert radians->deg
                    yaw = clamp_angle_deg(math.degrees(rrad))
                    # Build polygon from rotated rect
                    rect = ((cx, cy), (rw, rh), yaw)
                    p4i = cv2.boxPoints(rect).astype(np.float32)

                picks.append((c, cx, cy, yaw, p4i))

        # If you want only best pick
        if self.pick_one and len(picks) > 0:
            picks = [max(picks, key=lambda x: x[0])]

        # Draw & publish
        cv2.putText(out, f"FPS: {self._fps:.1f}  n={len(picks)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        for (c, cx, cy, yaw, p4) in picks:
            pts = p4.reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(out, [pts], True, (0, 255, 255), 2)

            icx, icy = int(round(cx)), int(round(cy))
            cv2.circle(out, (icx, icy), 5, (0, 0, 255), -1)

            # Arrow: yaw
            draw_arrow(out, (icx, icy), yaw_deg=yaw, length=60, color=(255, 0, 0), thickness=2)

            cv2.putText(out, f"{self.target_class} {c:.2f} yaw:{yaw:.1f}",
                        (icx + 8, icy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if self.pose_pub is not None:
                m = Pose2D()
                m.x = float(cx)
                m.y = float(cy)
                m.theta = math.radians(yaw)  # rad
                self.pose_pub.publish(m)

        cv2.imshow(self.win, out)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = YoloOBBCenterYaw()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


if __name__ == '__main__':
    main()