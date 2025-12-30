#!/usr/bin/env python3
import time
from typing import List

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

import cv2
from ultralytics import YOLO


class YoloRealtimeNode(Node):
    def __init__(self):
        super().__init__("yolo_realtime_node")

        # ===== Parameters =====
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("annotated_topic", "/yolo/annotated")
        self.declare_parameter("detections_topic", "/yolo/detections")

        # Model + inference
        self.declare_parameter("model_path", "/home/asus/Downloads/best(1).pt")
        self.declare_parameter("conf", 0.4)
        self.declare_parameter("imgsz", 480)
        self.declare_parameter("device", "cpu")  # force CPU

        self.image_topic = self.get_parameter("image_topic").value
        self.annotated_topic = self.get_parameter("annotated_topic").value
        self.detections_topic = self.get_parameter("detections_topic").value

        self.model_path = self.get_parameter("model_path").value
        self.conf = float(self.get_parameter("conf").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.device = str(self.get_parameter("device").value)

        # ===== YOLO load =====
        self.get_logger().info(f"Load model: {self.model_path}")
        self.model = YOLO(self.model_path)

        # ===== ROS I/O =====
        # Camera topics usually publish sensor QoS => BEST_EFFORT
        qos_cam = QoSProfile(depth=10)
        qos_cam.reliability = ReliabilityPolicy.BEST_EFFORT

        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image,
            self.image_topic,
            self.on_image,
            qos_cam
        )

        self.pub_annot = self.create_publisher(Image, self.annotated_topic, 10)
        self.pub_det = self.create_publisher(String, self.detections_topic, 10)

        self.get_logger().info(f"Subscribed to {self.image_topic}")
        self.get_logger().info(f"Publishing annotated to {self.annotated_topic}")
        self.get_logger().info(f"Publishing detections to {self.detections_topic}")

        # ===== FPS =====
        self._t_last = time.time()
        self._fps = 0.0

    def on_image(self, msg: Image):
        # ROS Image -> OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # YOLO inference
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )

        r = results[0]
        annotated = r.plot()  # draw bboxes/labels

        # FPS overlay
        t_now = time.time()
        dt = t_now - self._t_last
        self._t_last = t_now
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)

        cv2.putText(
            annotated,
            f"FPS: {self._fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        # Publish annotated image
        out = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        out.header = msg.header
        self.pub_annot.publish(out)

        # Publish detections text (optional but useful)
        lines: List[str] = []
        if r.boxes is not None and len(r.boxes) > 0:
            names = r.names
            for b in r.boxes:
                cls_id = int(b.cls.item())
                conf = float(b.conf.item())
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                cls_name = names.get(cls_id, str(cls_id))
                lines.append(f"{cls_name} conf={conf:.2f} xyxy=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

        det_msg = String()
        det_msg.data = "\n".join(lines)
        self.pub_det.publish(det_msg)


def main():
    rclpy.init()
    node = YoloRealtimeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()