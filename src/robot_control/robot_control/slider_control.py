#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import tkinter as tk
import threading

class SliderControl(Node):
    def __init__(self):
        super().__init__('slider_control')
        
        # 1. Publisher cho Cánh tay (6 khớp)
        self.arm_pub = self.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.arm_joints = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        self.arm_values = [0.0] * 6

        # 2. Publisher cho Kẹp (2 ngón)
        self.gripper_pub = self.create_publisher(JointTrajectory, '/gripper_controller/joint_trajectory', 10)
        self.gripper_joints = ['joint_gl', 'joint_gr']
        self.gripper_value = 0.0 # Giá trị chung cho cả 2 ngón

        # --- GIAO DIỆN ---
        self.root = tk.Tk()
        self.root.title("Full Robot Control")
        
        # --- PHẦN ARM ---
        tk.Label(self.root, text="--- ARM CONTROL ---", fg="blue", font=("Arial", 10, "bold")).pack(pady=5)
        self.arm_sliders = []
        for i, name in enumerate(self.arm_joints):
            frame = tk.Frame(self.root)
            frame.pack()
            tk.Label(frame, text=name, width=10).pack(side=tk.LEFT)
            s = tk.Scale(frame, from_=-3.14, to=3.14, resolution=0.01, orient=tk.HORIZONTAL, length=250, 
                         command=lambda val, idx=i: self.on_arm_change(val, idx))
            s.set(0)
            s.pack(side=tk.LEFT)
            self.arm_sliders.append(s)

        # --- PHẦN GRIPPER ---
        tk.Label(self.root, text="--- GRIPPER CONTROL ---", fg="red", font=("Arial", 10, "bold")).pack(pady=10)
        frame_g = tk.Frame(self.root)
        frame_g.pack()
        tk.Label(frame_g, text="Open/Close", width=10).pack(side=tk.LEFT)
        
        # Kẹp chạy từ 0.0 (đóng) đến 0.02 (mở hết cỡ - theo URDF của bạn)
        self.gripper_slider = tk.Scale(frame_g, from_=0.0, to=0.02, resolution=0.001, orient=tk.HORIZONTAL, length=250, 
                                       command=self.on_gripper_change)
        self.gripper_slider.set(0)
        self.gripper_slider.pack(side=tk.LEFT)

        # Nút Reset
        btn = tk.Button(self.root, text="RESET ALL HOME", command=self.reset_home, bg="orange", height=2)
        btn.pack(pady=20, fill=tk.X, padx=20)

    def on_arm_change(self, val, idx):
        self.arm_values[idx] = float(val)
        self.publish_arm()

    def on_gripper_change(self, val):
        self.gripper_value = float(val)
        self.publish_gripper()

    def reset_home(self):
        for s in self.arm_sliders: s.set(0)
        self.gripper_slider.set(0)
        
        self.arm_values = [0.0] * 6
        self.gripper_value = 0.0
        
        self.publish_arm()
        self.publish_gripper()

    def publish_arm(self):
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = self.arm_joints
        point = JointTrajectoryPoint()
        point.positions = self.arm_values
        point.time_from_start = Duration(sec=0, nanosec=500000000)
        msg.points.append(point)
        self.arm_pub.publish(msg)

    def publish_gripper(self):
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = self.gripper_joints
        
        point = JointTrajectoryPoint()
        # Cả 2 ngón cùng di chuyển giá trị giống nhau
        point.positions = [self.gripper_value, self.gripper_value]
        point.time_from_start = Duration(sec=0, nanosec=500000000)
        
        msg.points.append(point)
        self.gripper_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = SliderControl()
    
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        node.root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
