#pragma once
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/u_int64.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <std_srvs/srv/set_bool.hpp>

#include "serial_port.hpp"
#include "fas_protocol.hpp"
#include "types.hpp"

namespace robot_hardware {

class RobotHwNode : public rclcpp::Node {
public:
  RobotHwNode();

private:
  void onCmdPos(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
  void onCmdVel(const std_msgs::msg::Float64MultiArray::SharedPtr msg);

  void pollLoop(); // timer callback

  void sendOp(OpCode op, const std::vector<uint8_t>& payload = {});
  bool requestTelemetry();

  // services
  void onHome(const   std::shared_ptr<std_srvs::srv::Trigger::Request>,
                      std::shared_ptr<std_srvs::srv::Trigger::Response> res);
  void onStop(const   std::shared_ptr<std_srvs::srv::Trigger::Request>,
                      std::shared_ptr<std_srvs::srv::Trigger::Response> res);
  void onEstop(const  std::shared_ptr<std_srvs::srv::SetBool::Request> req,
                      std::shared_ptr<std_srvs::srv::SetBool::Response> res);

private:
  SerialPort serial_;
  std::string port_;
  int baud_{115200};
  double poll_hz_{200.0};

  std::vector<std::string> joints_;
  int protocol_id_base_{1};

  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_js_;
  rclcpp::Publisher<std_msgs::msg::UInt64>::SharedPtr pub_status_;

  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_cmd_pos_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_cmd_vel_;

  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_home_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_stop_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr srv_estop_;

  rclcpp::TimerBase::SharedPtr timer_;
  std::vector<uint8_t> rx_accum_;
};

}
