#!/usr/bin/env python3
import sys
import math
import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ultralytics import YOLO

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QHBoxLayout, QVBoxLayout,
    QSlider, QDoubleSpinBox, QCheckBox, QPushButton, QGroupBox, QFormLayout
)


# -------------------- Image helpers --------------------
def clamp_angle_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def draw_arrow(img, center: Tuple[int, int], yaw_deg: float, length: int = 60,
               color=(255, 0, 0), thickness: int = 2):
    cx, cy = center
    ang = math.radians(yaw_deg)
    x2 = int(cx + length * math.cos(ang))
    y2 = int(cy + length * math.sin(ang))
    cv2.arrowedLine(img, (cx, cy), (x2, y2), color, thickness, tipLength=0.25)


def gamma_correction(bgr: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(gamma, 1e-6)
    inv = 1.0 / gamma
    table = (np.arange(256) / 255.0) ** inv
    table = (table * 255.0).astype(np.uint8)
    return cv2.LUT(bgr, table)


def otsu_binary(bgr: np.ndarray, gamma: float = 1.2, invert: bool = False) -> np.ndarray:
    img_g = gamma_correction(bgr, gamma=gamma)
    gray = cv2.cvtColor(img_g, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    th_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, bw = cv2.threshold(gray, 0, 255, th_type | cv2.THRESH_OTSU)
    return bw


def bgr_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def gray_to_qpixmap(gray: np.ndarray) -> QPixmap:
    if gray.ndim == 2:
        h, w = gray.shape
        qimg = QImage(gray.data, w, h, w, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimg)
    return bgr_to_qpixmap(gray)


# -------------------- Shared frame buffer --------------------
@dataclass
class FrameBuffer:
    frame_bgr: Optional[np.ndarray] = None
    stamp: float = 0.0


# -------------------- ROS2 Image Source Node --------------------
class ImageSourceNode(Node):
    def __init__(self):
        super().__init__("yolo_obb_gui")

        # ROS params
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("model_path", "")
        self.declare_parameter("target_class", "wood")

        self.image_topic = self.get_parameter("image_topic").value
        self.model_path = self.get_parameter("model_path").value
        self.target_class = self.get_parameter("target_class").value

        if not self.model_path:
            raise RuntimeError("/home/asus/ros2_arm_ws/src/vision_tracking/resource/obb_v2.pt")

        self.bridge = CvBridge()
        self.buf = FrameBuffer()
        self.lock = threading.Lock()

        self.sub = self.create_subscription(
            Image, self.image_topic, self.on_image, qos_profile_sensor_data
        )
        self.get_logger().info(f"Subscribed: {self.image_topic}")
        self.get_logger().info(f"Model: {self.model_path} | target_class: {self.target_class}")

    def on_image(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self.lock:
            self.buf.frame_bgr = frame
            self.buf.stamp = time.time()

    def get_latest(self) -> Tuple[Optional[np.ndarray], float]:
        with self.lock:
            if self.buf.frame_bgr is None:
                return None, 0.0
            return self.buf.frame_bgr.copy(), self.buf.stamp


# -------------------- Inference Worker Thread --------------------
class InferenceWorker(QThread):
    new_images = pyqtSignal(object, object, float, int)  # det_bgr, otsu_gray_or_None, fps, n_det

    def __init__(self, node: ImageSourceNode):
        super().__init__()
        self.node = node

        # runtime settings (GUI chỉnh)
        self.conf = 0.5
        self.pick_one = False
        self.show_otsu = False
        self.gamma = 1.2
        self.otsu_invert = False

        self._stop = False
        self._last_stamp = 0.0

        # load model in this thread context
        self.model = YOLO(self.node.model_path)

        # map class name -> id (optional filter)
        self.names = self.model.names
        self.target_id = None
        if isinstance(self.names, dict):
            for k, v in self.names.items():
                if v == self.node.target_class:
                    self.target_id = int(k)
                    break
        else:
            for i, v in enumerate(self.names):
                if v == self.node.target_class:
                    self.target_id = i
                    break

        self._t_last = time.time()
        self._fps = 0.0

    def stop(self):
        self._stop = True

    def run(self):
        while not self._stop:
            frame, stamp = self.node.get_latest()
            if frame is None or stamp == self._last_stamp:
                time.sleep(0.005)
                continue
            self._last_stamp = stamp

            # FPS (worker loop)
            t = time.time()
            dt = t - self._t_last
            if dt > 0:
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
            self._t_last = t

            # YOLO predict
            results = self.model.predict(source=frame, conf=self.conf, verbose=False)
            r0 = results[0]
            out = frame.copy()

            picks = []  # (conf, cx, cy, yaw_deg, poly4, cls_id)

            if getattr(r0, "obb", None) is not None and r0.obb is not None:
                obb = r0.obb
                cls = obb.cls.cpu().numpy().astype(int) if obb.cls is not None else None
                conf = obb.conf.cpu().numpy() if obb.conf is not None else None
                poly = obb.xyxyxyxy.cpu().numpy() if hasattr(obb, "xyxyxyxy") and obb.xyxyxyxy is not None else None

                if poly is not None:
                    N = poly.shape[0]
                    for i in range(N):
                        c = float(conf[i]) if conf is not None else 1.0
                        if c < self.conf:
                            continue
                        cid = int(cls[i]) if cls is not None else -1
                        if self.target_id is not None and cid != self.target_id:
                            continue

                        p4 = poly[i].astype(np.float32)  # (4,2)
                        cx = float(np.mean(p4[:, 0]))
                        cy = float(np.mean(p4[:, 1]))

                        # yaw from minAreaRect (long side)
                        rect = cv2.minAreaRect(p4)
                        (_, _), (rw, rh), ang = rect
                        yaw = ang + 90.0 if rw < rh else ang
                        yaw = clamp_angle_deg(yaw)

                        picks.append((c, cx, cy, yaw, p4, cid))
                else:
                    cv2.putText(out, "OBB exists but no polygon output.", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                # fallback boxes
                if r0.boxes is not None and len(r0.boxes) > 0:
                    xyxy = r0.boxes.xyxy.cpu().numpy()
                    conf = r0.boxes.conf.cpu().numpy()
                    cls = r0.boxes.cls.cpu().numpy().astype(int)
                    for i in range(xyxy.shape[0]):
                        c = float(conf[i])
                        if c < self.conf:
                            continue
                        cid = int(cls[i])
                        x1, y1, x2, y2 = xyxy[i]
                        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        picks.append((c, cx, cy, 0.0, None, cid))

                cv2.putText(out, "No OBB output (use YOLOv8-OBB weights).", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if self.pick_one and len(picks) > 0:
                picks = [max(picks, key=lambda x: x[0])]

            # draw
            cv2.putText(out, f"FPS: {self._fps:.1f}  n={len(picks)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            for (c, cx, cy, yaw, p4, cid) in picks:
                icx, icy = int(round(cx)), int(round(cy))

                if p4 is not None:
                    pts = p4.reshape(-1, 1, 2).astype(np.int32)
                    cv2.polylines(out, [pts], True, (0, 255, 255), 2)
                    draw_arrow(out, (icx, icy), yaw_deg=yaw, length=70)

                cv2.circle(out, (icx, icy), 4, (0, 0, 255), -1)

                name = str(cid)
                if isinstance(self.names, dict):
                    name = self.names.get(cid, str(cid))
                else:
                    if 0 <= cid < len(self.names):
                        name = self.names[cid]

                cv2.putText(out, f"{name} {c:.2f} yaw:{yaw:.1f}", (icx + 8, icy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # optional otsu preview (whole frame)
            otsu_img = None
            if self.show_otsu:
                bw = otsu_binary(frame, gamma=self.gamma, invert=self.otsu_invert)
                otsu_img = bw

            self.new_images.emit(out, otsu_img, float(self._fps), int(len(picks)))


# -------------------- PyQt GUI --------------------
class MainWindow(QMainWindow):
    def __init__(self, node: ImageSourceNode, worker: InferenceWorker):
        super().__init__()
        self.node = node
        self.worker = worker

        self.setWindowTitle("YOLO-OBB GUI (ROS2)")

        # Image views
        self.lbl_det = QLabel("Waiting for images...")
        self.lbl_det.setAlignment(Qt.AlignCenter)
        self.lbl_det.setMinimumSize(640, 360)

        self.lbl_otsu = QLabel("Otsu off")
        self.lbl_otsu.setAlignment(Qt.AlignCenter)
        self.lbl_otsu.setMinimumSize(320, 180)

        # Stats
        self.lbl_stats = QLabel("FPS: 0.0 | n=0")

        # Controls
        box = QGroupBox("Controls")
        form = QFormLayout()

        self.sld_conf = QSlider(Qt.Horizontal)
        self.sld_conf.setRange(1, 99)
        self.sld_conf.setValue(int(worker.conf * 100))
        self.sld_conf.valueChanged.connect(self.on_conf)

        self.chk_pick_one = QCheckBox("Pick one (max conf)")
        self.chk_pick_one.setChecked(worker.pick_one)
        self.chk_pick_one.stateChanged.connect(self.on_pick_one)

        self.chk_otsu = QCheckBox("Show Otsu (whole frame)")
        self.chk_otsu.setChecked(worker.show_otsu)
        self.chk_otsu.stateChanged.connect(self.on_show_otsu)

        self.spn_gamma = QDoubleSpinBox()
        self.spn_gamma.setRange(0.2, 3.0)
        self.spn_gamma.setSingleStep(0.1)
        self.spn_gamma.setValue(worker.gamma)
        self.spn_gamma.valueChanged.connect(self.on_gamma)

        self.chk_otsu_inv = QCheckBox("Otsu invert")
        self.chk_otsu_inv.setChecked(worker.otsu_invert)
        self.chk_otsu_inv.stateChanged.connect(self.on_otsu_inv)

        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)

        self.btn_start.clicked.connect(self.start_worker)
        self.btn_stop.clicked.connect(self.stop_worker)

        form.addRow("Conf", self.sld_conf)
        form.addRow(self.chk_pick_one)
        form.addRow(self.chk_otsu)
        form.addRow("Gamma", self.spn_gamma)
        form.addRow(self.chk_otsu_inv)
        form.addRow(self.btn_start, self.btn_stop)
        box.setLayout(form)

        # Layout
        left = QVBoxLayout()
        left.addWidget(self.lbl_det, stretch=1)
        left.addWidget(self.lbl_stats)

        right = QVBoxLayout()
        right.addWidget(box)
        right.addWidget(self.lbl_otsu)

        root = QHBoxLayout()
        root.addLayout(left, stretch=3)
        root.addLayout(right, stretch=1)

        w = QWidget()
        w.setLayout(root)
        self.setCentralWidget(w)

        # Connect worker signal
        self.worker.new_images.connect(self.on_new_images)

    def start_worker(self):
        if not self.worker.isRunning():
            self.worker._stop = False
            self.worker.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop_worker(self):
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1500)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def closeEvent(self, event):
        self.stop_worker()
        event.accept()

    def on_conf(self, v: int):
        self.worker.conf = float(v) / 100.0

    def on_pick_one(self, state: int):
        self.worker.pick_one = (state == Qt.Checked)

    def on_show_otsu(self, state: int):
        self.worker.show_otsu = (state == Qt.Checked)
        if not self.worker.show_otsu:
            self.lbl_otsu.setText("Otsu off")

    def on_gamma(self, v: float):
        self.worker.gamma = float(v)

    def on_otsu_inv(self, state: int):
        self.worker.otsu_invert = (state == Qt.Checked)

    def on_new_images(self, det_bgr, otsu_gray, fps: float, n: int):
        # detection view
        pm = bgr_to_qpixmap(det_bgr)
        self.lbl_det.setPixmap(pm.scaled(self.lbl_det.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # otsu view
        if otsu_gray is not None:
            pm2 = gray_to_qpixmap(otsu_gray)
            self.lbl_otsu.setPixmap(pm2.scaled(self.lbl_otsu.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self.lbl_stats.setText(f"FPS: {fps:.1f} | n={n}")


# -------------------- main --------------------
def main():
    # Let ROS parse ROS args, and give Qt non-ROS args
    qt_argv = remove_ros_args(sys.argv)

    rclpy.init(args=sys.argv)
    node = ImageSourceNode()

    # spin node in background thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    app = QApplication(qt_argv)

    worker = InferenceWorker(node)
    win = MainWindow(node, worker)
    win.resize(1200, 700)
    win.show()

    ret = app.exec_()

    # shutdown
    worker.stop()
    if worker.isRunning():
        worker.wait(1500)

    node.destroy_node()
    rclpy.shutdown()
    sys.exit(ret)


if __name__ == "__main__":
    main()