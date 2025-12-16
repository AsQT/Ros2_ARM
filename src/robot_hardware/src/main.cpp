#include <rclcpp/rclcpp.hpp>
#include "robot_hw_node.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<robot_hardware::RobotHwNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
