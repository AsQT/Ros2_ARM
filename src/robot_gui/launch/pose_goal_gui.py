#!/usr/bin/env python3
import sys
import threading
import math

from PyQt5 import QtWidgets, QtCore

import rclpy
from rclpy.node import Node

from pymoveit2 import MoveIt2

# ====== Robot semantic (từ SRDF của bạn) ======
ARM_GROUP_NAME = "arm"
BASE_LINK_NAME = "base_link"
EE_LINK_NAME   = "tcp_link"
ARM_JOINTS     = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]


def rpy_to_quat(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return (qx, qy, qz, qw)


class MoveItGuiNode(Node):
    def __init__(self, log_cb):
        super().__init__("pose_goal_gui")
        self.log_cb = log_cb

        self.moveit2 = MoveIt2(
            node=self,
            joint_names=ARM_JOINTS,
            base_link_name=BASE_LINK_NAME,
            end_effector_name=EE_LINK_NAME,
            group_name=ARM_GROUP_NAME,
        )

        self._last_plan_ok = False
        self.log("MoveIt2 ready.")
        self.log(f"Group={ARM_GROUP_NAME}, base={BASE_LINK_NAME}, ee={EE_LINK_NAME}")
        self.log(f"Joints={ARM_JOINTS}")

    def log(self, s: str):
        if self.log_cb:
            self.log_cb(s)
        self.get_logger().info(s)

    # ====== PLAN TO POSE (BASE_LINK) ======
    def plan_to_pose(self, x, y, z, roll, pitch, yaw, ignore_rpy=True):
        if ignore_rpy:
            roll = pitch = yaw = 0.0

        quat = rpy_to_quat(roll, pitch, yaw)

        self.moveit2.set_pose_goal(
            position=[x, y, z],
            quat_xyzw=list(quat),
            frame_id="base_link",   # ÉP base_link
        )

        self.log(
            f"PLAN pose (base_link): "
            f"p=({x:.3f},{y:.3f},{z:.3f}), "
            f"rpy=({roll:.3f},{pitch:.3f},{yaw:.3f})"
        )

        ok = self.moveit2.plan()
        self._last_plan_ok = bool(ok)
        self.log("Plan: " + ("SUCCESS ✅" if ok else "FAILED ❌"))
        return ok

    # ====== EXECUTE ======
    def execute(self):
        if not self._last_plan_ok:
            self.log("Execute blocked: plan chưa OK.")
            return False

        self.log("EXECUTE trajectory...")
        ok = self.moveit2.execute()
        self.log("Execute: " + ("SUCCESS ✅" if ok else "FAILED ❌"))
        return ok

    # ====== HOME ======
    def go_home(self):
        # Try named target first
        try:
            self.moveit2.set_named_target("home")
            self.log("PLAN named target: home")
            ok = self.moveit2.plan()
            self._last_plan_ok = bool(ok)
            self.log("Plan(home): " + ("SUCCESS ✅" if ok else "FAILED ❌"))
            return ok
        except Exception as e:
            self.log(f"Named target not supported → fallback joint home ({e})")

        # Fallback: joint home
        try:
            self.moveit2.set_joint_goal([0, 0, 0, 0, 0, 0])
            ok = self.moveit2.plan()
            self._last_plan_ok = bool(ok)
            self.log("Plan(joint home): " + ("SUCCESS ✅" if ok else "FAILED ❌"))
            return ok
        except Exception as e:
            self.log(f"Joint home failed: {e}")
            return False


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot Pose Goal GUI (MoveIt2 + Gazebo)")

        layout = QtWidgets.QGridLayout(self)

        def mk_spin(minv, maxv, dec=4, step=0.01):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(minv, maxv)
            s.setDecimals(dec)
            s.setSingleStep(step)
            return s

        self.x = mk_spin(-1.5, 1.5)
        self.y = mk_spin(-1.5, 1.5)
        self.z = mk_spin(0.0,  1.5)

        self.r = mk_spin(-math.pi, math.pi, step=0.05)
        self.p = mk_spin(-math.pi, math.pi, step=0.05)
        self.yaw = mk_spin(-math.pi, math.pi, step=0.05)

        # ====== Ignore RPY ======
        self.ignore_rpy = QtWidgets.QCheckBox("Ignore RPY (position only)")
        self.ignore_rpy.setChecked(True)

        # Default test pose (dễ plan)
        self.x.setValue(0.30)
        self.y.setValue(0.00)
        self.z.setValue(0.60)

        layout.addWidget(self.ignore_rpy, 0, 0, 1, 4)

        layout.addWidget(QtWidgets.QLabel("x (m)"), 1, 0); layout.addWidget(self.x, 1, 1)
        layout.addWidget(QtWidgets.QLabel("y (m)"), 2, 0); layout.addWidget(self.y, 2, 1)
        layout.addWidget(QtWidgets.QLabel("z (m)"), 3, 0); layout.addWidget(self.z, 3, 1)

        layout.addWidget(QtWidgets.QLabel("roll (rad)"), 1, 2); layout.addWidget(self.r, 1, 3)
        layout.addWidget(QtWidgets.QLabel("pitch (rad)"), 2, 2); layout.addWidget(self.p, 2, 3)
        layout.addWidget(QtWidgets.QLabel("yaw (rad)"), 3, 2); layout.addWidget(self.yaw, 3, 3)

        self.btn_home = QtWidgets.QPushButton("Home")
        self.btn_plan = QtWidgets.QPushButton("Plan")
        self.btn_exec = QtWidgets.QPushButton("Execute")

        layout.addWidget(self.btn_home, 4, 0)
        layout.addWidget(self.btn_plan, 4, 1)
        layout.addWidget(self.btn_exec, 4, 3)

        self.logbox = QtWidgets.QPlainTextEdit()
        self.logbox.setReadOnly(True)
        layout.addWidget(self.logbox, 5, 0, 1, 4)

        # ROS
        rclpy.init()
        self.node = MoveItGuiNode(self.append_log)

        self.spin_thread = threading.Thread(target=self._spin_ros, daemon=True)
        self.spin_thread.start()

        self.btn_plan.clicked.connect(self.on_plan)
        self.btn_exec.clicked.connect(self.on_exec)
        self.btn_home.clicked.connect(self.on_home)

    def append_log(self, s: str):
        QtCore.QMetaObject.invokeMethod(
            self.logbox,
            "appendPlainText",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, s),
        )

    def _spin_ros(self):
        rclpy.spin(self.node)

    def on_plan(self):
        self.node.plan_to_pose(
            self.x.value(),
            self.y.value(),
            self.z.value(),
            self.r.value(),
            self.p.value(),
            self.yaw.value(),
            ignore_rpy=self.ignore_rpy.isChecked(),
        )

    def on_exec(self):
        self.node.execute()

    def on_home(self):
        self.node.go_home()

    def closeEvent(self, event):
        try:
            self.node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass
        super().closeEvent(event)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(800, 480)
    w.show()
    sys.exit(app.exec_())