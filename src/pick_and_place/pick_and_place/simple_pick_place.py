import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty
from moveit.planning import MoveItPy
import time

class SimplePickPlace(Node):
    def __init__(self):
        super().__init__('simple_pick_place')
        
        # 1. Khởi tạo MoveIt
        try:
            self.robot = MoveItPy(node_name="moveit_py")
            self.arm = self.robot.planning_component("arm")
            self.gripper = self.robot.planning_component("gripper")
        except Exception as e:
            self.get_logger().error(f"Lỗi MoveIt: {e}")
            return

        # 2. Publisher Plugin "Dính"
        self.attach_pub = self.create_publisher(Empty, '/model/robot/detachable_joint/toggle', 1)
        self.get_logger().info("--- SẴN SÀNG ---")

    def plan_and_execute(self, component, pose_goal, pose_link="tcp_link"):
        component.set_start_state_to_current_state()
        component.set_goal_state(pose_stamped_msg=pose_goal, pose_link=pose_link)
        
        plan_result = component.plan()
        if plan_result:
            self.get_logger().info("Đang chạy...")
            self.robot.execute(plan_result.trajectory, controllers=[])
            return True
        else:
            self.get_logger().error("Lập kế hoạch thất bại!")
            return False

    def run_routine(self):
        # Tọa độ hộp đỏ (cần chỉnh theo thực tế)
        BOX_X, BOX_Y, BOX_Z = 0.5, 0.1, 1.025
        HOVER_Z = 1.2
        
        # Quaternion chúi đầu xuống (Mũi kẹp hướng xuống đất)
        # Giá trị này tương đương xoay 180 độ quanh trục X
        QX, QY, QZ, QW = 1.0, 0.0, 0.0, 0.0 

        # 1. Home
        self.get_logger().info("1. Về Home")
        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(configuration_name="home")
        self.arm.set_planner_id("PTP") # Set PTP cho Home luôn
        plan = self.arm.plan()
        if plan: self.robot.execute(plan.trajectory, controllers=[])
        time.sleep(1.0)

        # 2. Đến trên hộp
        self.get_logger().info("2. Đến trên hộp")
        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.pose.position.x = BOX_X
        pose.pose.position.y = BOX_Y
        pose.pose.position.z = HOVER_Z
        pose.pose.orientation.x = QX
        pose.pose.orientation.y = QY
        pose.pose.orientation.z = QZ
        pose.pose.orientation.w = QW
        self.plan_and_execute(self.arm, pose)
        time.sleep(1.0)

        # 3. Lao xuống
        self.get_logger().info("3. Lao xuống")
        pose.pose.position.z = BOX_Z + 0.05
        self.plan_and_execute(self.arm, pose)
        time.sleep(1.0)

        # 4. Dính vật
        self.get_logger().info("4. DÍNH VẬT")
        self.attach_pub.publish(Empty())
        time.sleep(0.5)

        # 5. Nhấc lên
        self.get_logger().info("5. Nhấc lên")
        pose.pose.position.z = HOVER_Z
        self.plan_and_execute(self.arm, pose)
        
        # 6. Thả vật
        self.get_logger().info("6. Thả vật")
        pose.pose.position.y = -0.3
        self.plan_and_execute(self.arm, pose)
        self.attach_pub.publish(Empty())
        
        self.get_logger().info("--- XONG ---")

def main():
    rclpy.init()
    node = SimplePickPlace()
    time.sleep(2.0) # Đợi MoveIt kết nối
    node.run_routine()
    rclpy.shutdown()
