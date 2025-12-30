#!/usr/bin/env python3
import os
import re
import sys
import threading
from dataclasses import dataclass

import cv2
import numpy as np

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QWidget, QFileDialog, QMessageBox, QComboBox
)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def cv_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    """Convert OpenCV BGR image to Qt QPixmap."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def ensure_dir(path: str) -> None:
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def next_index_in_folder(folder: str, prefix: str, ext: str) -> int:
    """
    Scan folder for files like prefix_<number>.<ext> and return next number.
    Example: prefix="anh", ext="png" => anh_1.png, anh_2.png ...
    """
    if not folder or not os.path.isdir(folder):
        return 1

    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.{re.escape(ext)}$", re.IGNORECASE)
    max_n = 0
    for name in os.listdir(folder):
        m = pattern.match(name)
        if m:
            try:
                max_n = max(max_n, int(m.group(1)))
            except ValueError:
                pass
    return max_n + 1


@dataclass
class SharedFrame:
    lock: threading.Lock
    frame_bgr: np.ndarray | None = None


class ImageSubNode(Node):
    def __init__(self, shared: SharedFrame, topic: str):
        super().__init__("cam_capture_gui_node")
        self.shared = shared
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, topic, self.cb, 10)
        self.get_logger().info(f"Subscribed to: {topic}")

    def cb(self, msg: Image):
        try:
            # Most ROS camera topics are "bgr8" or "rgb8". Use "passthrough" for safety.
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            # If passthrough gives RGB, convert to BGR for consistent saving/view:
            if cv_img.ndim == 3 and cv_img.shape[2] == 3:
                # Heuristic: assume it's RGB and convert; if already BGR, this is still ok visually
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

        with self.shared.lock:
            self.shared.frame_bgr = cv_img


class MainWindow(QWidget):
    def __init__(self, shared: SharedFrame):
        super().__init__()
        self.shared = shared

        self.setWindowTitle("ROS2 Astra Camera - Capture GUI")
        self.resize(980, 620)

        # --- UI ---
        self.video_label = QLabel("No image yet")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #111; color: #ddd;")
        self.video_label.setMinimumSize(640, 480)

        self.topic_combo = QComboBox()
        # Bạn có thể thêm topic khác nếu cần
        self.topic_combo.addItems([
            "/astra/rgb/image_raw",
            "/camera/image_raw",
            "/image_raw",
        ])
        self.topic_combo.setEditable(True)  # cho phép bạn tự sửa topic

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Chọn thư mục để lưu ảnh dataset YOLO...")

        self.prefix_edit = QLineEdit("anh")
        self.ext_combo = QComboBox()
        self.ext_combo.addItems(["png", "jpg"])

        self.btn_browse = QPushButton("Chọn thư mục")
        self.btn_capture = QPushButton("Chụp ảnh")
        self.btn_capture.setEnabled(False)

        # Layout
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Topic:"))
        controls.addWidget(self.topic_combo)

        save_row = QHBoxLayout()
        save_row.addWidget(QLabel("Folder:"))
        save_row.addWidget(self.path_edit)
        save_row.addWidget(self.btn_browse)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Prefix:"))
        name_row.addWidget(self.prefix_edit)
        name_row.addWidget(QLabel("Ext:"))
        name_row.addWidget(self.ext_combo)
        name_row.addStretch(1)
        name_row.addWidget(self.btn_capture)

        root = QVBoxLayout()
        root.addLayout(controls)
        root.addWidget(self.video_label, stretch=1)
        root.addLayout(save_row)
        root.addLayout(name_row)
        self.setLayout(root)

        # Timer to refresh UI
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~33 FPS GUI refresh

        # Signals
        self.btn_browse.clicked.connect(self.pick_folder)
        self.btn_capture.clicked.connect(self.capture)

    def pick_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu ảnh", os.getcwd())
        if folder:
            self.path_edit.setText(folder)
            self.btn_capture.setEnabled(True)

    def update_frame(self):
        with self.shared.lock:
            frame = None if self.shared.frame_bgr is None else self.shared.frame_bgr.copy()

        if frame is None:
            return

        # Fit display
        pix = cv_to_qpixmap(frame)
        self.video_label.setPixmap(pix.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def capture(self):
        folder = self.path_edit.text().strip()
        prefix = self.prefix_edit.text().strip() or "anh"
        ext = self.ext_combo.currentText().strip().lower()

        if not folder:
            QMessageBox.warning(self, "Thiếu đường dẫn", "Bạn chưa chọn thư mục lưu ảnh.")
            return

        ensure_dir(folder)

        with self.shared.lock:
            frame = None if self.shared.frame_bgr is None else self.shared.frame_bgr.copy()

        if frame is None:
            QMessageBox.warning(self, "Chưa có ảnh", "Chưa nhận được frame từ camera topic.")
            return

        idx = next_index_in_folder(folder, prefix, ext)
        filename = f"{prefix}_{idx}.{ext}"
        out_path = os.path.join(folder, filename)

        ok = cv2.imwrite(out_path, frame)
        if ok:
            QMessageBox.information(self, "Đã lưu", f"Saved: {out_path}")
        else:
            QMessageBox.critical(self, "Lỗi", f"Không lưu được ảnh: {out_path}")


def run_gui(topic: str):
    shared = SharedFrame(lock=threading.Lock())

    # ROS thread
    rclpy.init(args=None)
    node = ImageSubNode(shared=shared, topic=topic)

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # Qt App
    app = QApplication(sys.argv)
    w = MainWindow(shared)
    # set default topic in UI
    w.topic_combo.setCurrentText(topic)
    w.show()

    exit_code = app.exec_()

    # Shutdown ROS
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    # Bạn có thể truyền topic qua command line: python3 cam_capture_gui.py /astra/rgb/image_raw
    topic_arg = sys.argv[1] if len(sys.argv) > 1 else "/astra/rgb/image_raw"
    run_gui(topic_arg)