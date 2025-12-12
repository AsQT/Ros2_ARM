#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, PositionConstraint, OrientationConstraint, BoundingVolume
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Pose # <--- Đã thêm Pose
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation as R
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time

class CommanderGUI(Node):
    def __init__(self):
        super().__init__('commander_gui')
        
        self._action_client = ActionClient(self, MoveGroup, 'move_action')
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.group_name = "arm"
        self.base_frame = "world"
        self.tip_frame = "tcp_link"

        # --- GIAO DIỆN TKINTER ---
        self.root = tk.Tk()
        self.root.title("MoveIt Commander")
        self.root.geometry("500x700")

        lf_fk = tk.LabelFrame(self.root, text="Current Position (FK)", font=("Arial", 10, "bold"), fg="blue")
        lf_fk.pack(fill="x", padx=10, pady=5)
        self.lbl_fk = tk.Label(lf_fk, text="Waiting for TF...", justify=tk.LEFT, font=("Consolas", 10))
        self.lbl_fk.pack(padx=5, pady=5)

        lf_input = tk.LabelFrame(self.root, text="Input Goal (XYZ - RPY)", font=("Arial", 10, "bold"))
        lf_input.pack(fill="x", padx=10, pady=5)

        self.entries = {}
        labels = ['x (m)', 'y (m)', 'z (m)', 'roll (rad)', 'pitch (rad)', 'yaw (rad)']
        # Tọa độ mặc định an toàn
        defaults = ['0.3', '0.0', '0.3', '0.0', '1.57', '0.0']
        
        for i, (lbl, default) in enumerate(zip(labels, defaults)):
            frame = tk.Frame(lf_input)
            frame.pack(fill="x", padx=5, pady=2)
            tk.Label(frame, text=lbl, width=10).pack(side=tk.LEFT)
            e = tk.Entry(frame)
            e.insert(0, default)
            e.pack(side=tk.RIGHT, expand=True, fill="x")
            self.entries[lbl] = e

        btn_add = tk.Button(lf_input, text="Add Goal to List", command=self.add_goal)
        btn_add.pack(pady=5)

        lf_list = tk.LabelFrame(self.root, text="Goal List", font=("Arial", 10, "bold"))
        lf_list.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.goal_listbox = tk.Listbox(lf_list, height=6)
        self.goal_listbox.pack(side=tk.LEFT, fill="both", expand=True, padx=5, pady=5)
        
        btn_clear = tk.Button(lf_list, text="Clear", command=lambda: self.goal_listbox.delete(0, tk.END))
        btn_clear.pack(side=tk.RIGHT, padx=5)

        frame_btns = tk.Frame(self.root)
        frame_btns.pack(fill="x", padx=10, pady=10)
        
        tk.Button(frame_btns, text="PLAN ONLY", bg="lightblue", command=lambda: self.send_request(plan_only=True)).pack(side=tk.LEFT, expand=True, fill="x", padx=2)
        tk.Button(frame_btns, text="EXECUTE", bg="orange", command=self.execute_plan).pack(side=tk.LEFT, expand=True, fill="x", padx=2)
        tk.Button(frame_btns, text="PLAN & EXECUTE", bg="lightgreen", command=lambda: self.send_request(plan_only=False)).pack(side=tk.LEFT, expand=True, fill="x", padx=2)

        self.latest_trajectory = None
        self.update_fk_display()

    def update_fk_display(self):
        try:
            t = self.tf_buffer.lookup_transform(self.base_frame, self.tip_frame, rclpy.time.Time())
            p = t.transform.translation
            q = t.transform.rotation
            r = R.from_quat([q.x, q.y, q.z, q.w])
            rpy = r.as_euler('xyz', degrees=False)
            text = (f"X: {p.x:.3f} | Y: {p.y:.3f} | Z: {p.z:.3f}\n"
                    f"R: {rpy[0]:.3f} | P: {rpy[1]:.3f} | Y: {rpy[2]:.3f}")
            self.lbl_fk.config(text=text, fg="black")
        except Exception:
            self.lbl_fk.config(text=f"Looking for transform {self.base_frame}->{self.tip_frame}...", fg="red")
        self.root.after(500, self.update_fk_display)

    def add_goal(self):
        try:
            val_str = " | ".join([e.get() for e in self.entries.values()])
            self.goal_listbox.insert(tk.END, val_str)
        except:
            pass

    def get_current_target(self):
        selection = self.goal_listbox.curselection()
        if not selection:
            if self.goal_listbox.size() > 0:
                idx = 0 
            else:
                return [float(e.get()) for e in self.entries.values()]
        else:
            idx = selection[0]
        raw = self.goal_listbox.get(idx)
        return [float(x) for x in raw.split(" | ")]

    def send_request(self, plan_only=False):
        try:
            vals = self.get_current_target()
            x, y, z, r, p, y_yaw = vals
        except ValueError:
            messagebox.showerror("Error", "Invalid coordinates!")
            return

        goal_msg = MoveGroup.Goal()
        goal_msg.request.workspace_parameters.header.frame_id = self.base_frame
        goal_msg.request.group_name = self.group_name
        goal_msg.request.num_planning_attempts = 10
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.planning_options.plan_only = plan_only
        goal_msg.planning_options.replan = True

        # --- 1. Position Constraint ---
        pcm = PositionConstraint()
        pcm.header.frame_id = self.base_frame
        pcm.link_name = self.tip_frame
        
        # Tạo hộp sai số (Tolerance Box)
        bv = BoundingVolume()
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.005, 0.005, 0.005]
        bv.primitives.append(box)
        
        # --- KHẮC PHỤC LỖI TYPE ERROR TẠI ĐÂY ---
        target_pose = Pose()
        target_pose.position.x = x
        target_pose.position.y = y
        target_pose.position.z = z
        target_pose.orientation.w = 1.0 # Quaternion chuẩn
        
        # Nạp Pose vào primitive_poses (MoveIt bắt buộc phải là Pose, không được là Point)
        bv.primitive_poses.append(target_pose)
        pcm.constraint_region = bv
        pcm.weight = 1.0

        # --- 2. Orientation Constraint ---
        ocm = OrientationConstraint()
        ocm.header.frame_id = self.base_frame
        ocm.link_name = self.tip_frame
        
        rot = R.from_euler('xyz', [r, p, y_yaw], degrees=False)
        q = rot.as_quat()
        ocm.orientation.x = q[0]
        ocm.orientation.y = q[1]
        ocm.orientation.z = q[2]
        ocm.orientation.w = q[3]
        
        ocm.absolute_x_axis_tolerance = 0.1
        ocm.absolute_y_axis_tolerance = 0.1
        ocm.absolute_z_axis_tolerance = 0.1
        ocm.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(pcm)
        constraints.orientation_constraints.append(ocm)
        goal_msg.request.goal_constraints.append(constraints)

        if not self._action_client.wait_for_server(timeout_sec=1.0):
            messagebox.showerror("Error", "MoveGroup Action Server not found! Is MoveIt running?")
            return

        print(f"Sending Goal: {vals}")
        self._future = self._action_client.send_goal_async(goal_msg)
        self._future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            print("Goal rejected :(")
            return
        print("Goal accepted! Planning...")
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        if result.error_code.val == 1:
            print("SUCCESS!")
            self.latest_trajectory = result.planned_trajectory
        else:
            print(f"FAILED with error code: {result.error_code.val}")
            messagebox.showerror("MoveIt Error", f"Planning failed! Code: {result.error_code.val}")

    def execute_plan(self):
        messagebox.showinfo("Info", "Please use 'Plan & Execute' for now.")

def main(args=None):
    rclpy.init(args=args)
    node = CommanderGUI()
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