#include "robot_hw_node.hpp"
#include <algorithm>

namespace robot_hardware {

RobotHwNode::RobotHwNode() : rclcpp::Node("robot_hardware_node") {
  port_             = declare_parameter<std::string>("port", "/dev/ttyUSB0");
  baud_             = declare_parameter<int>("baud", 115200);
  poll_hz_          = declare_parameter<double>("poll_hz", 200.0);
  protocol_id_base_ = declare_parameter<int>("protocol_id_base", 1);

  joints_           = declare_parameter<std::vector<std::string>>(
    "joints", {"joint_1","joint_2","joint_3","joint_4","joint_5","joint_6"}
  );

  pub_js_           = create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10);
  pub_status_       = create_publisher<std_msgs::msg::UInt64>("/robot_hardware/status", 10);

  sub_cmd_pos_      = create_subscription<std_msgs::msg::Float64MultiArray>(
    "/robot_hardware/cmd_pos", 10,
    std::bind(&RobotHwNode::onCmdPos, this, std::placeholders::_1));

  sub_cmd_vel_      = create_subscription<std_msgs::msg::Float64MultiArray>(
    "/robot_hardware/cmd_vel", 10,
    std::bind(&RobotHwNode::onCmdVel, this, std::placeholders::_1));

  srv_home_         = create_service<std_srvs::srv::Trigger>(
    "/robot_hardware/home",
    std::bind(&RobotHwNode::onHome, this, std::placeholders::_1, std::placeholders::_2));

  srv_stop_         = create_service<std_srvs::srv::Trigger>(
    "/robot_hardware/stop",
    std::bind(&RobotHwNode::onStop, this, std::placeholders::_1, std::placeholders::_2));

  srv_estop_        = create_service<std_srvs::srv::SetBool>(
    "/robot_hardware/estop",
    std::bind(&RobotHwNode::onEstop, this, std::placeholders::_1, std::placeholders::_2));

  if (!serial_.open(port_, baud_)) {
    RCLCPP_ERROR(get_logger(), "Failed to open serial port: %s @%d", port_.c_str(), baud_);
  } else {
    RCLCPP_INFO(get_logger(), "Serial opened: %s @%d", port_.c_str(), baud_);
  }

  const auto period = std::chrono::duration<double>(1.0 / std::max(1.0, poll_hz_));
  timer_ = create_wall_timer(
    std::chrono::duration_cast<std::chrono::nanoseconds>(period),
    std::bind(&RobotHwNode::pollLoop, this)
  );
}

void RobotHwNode::sendOp(OpCode op, const std::vector<uint8_t>& payload) {
  Frame f{op, payload};
  auto bytes = packFrame(f);
  if (!serial_.writeBytes(bytes)) {
    RCLCPP_WARN(get_logger(), "Serial write failed");
  }
}

void RobotHwNode::onCmdPos(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
  if (msg->data.size() != joints_.size()) {
    RCLCPP_WARN(get_logger(), "cmd_pos size mismatch: got %zu need %zu", msg->data.size(), joints_.size());
    return;
  }
  std::vector<float> posf(joints_.size());
  for (size_t i = 0; i < joints_.size(); i++) posf[i] = (float)msg->data[i]; // rad
  auto payload = packFloat32Array(posf);
  sendOp(OpCode::CMD_POS, payload);
}

void RobotHwNode::onCmdVel(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
  if (msg->data.size() != joints_.size()) {
    RCLCPP_WARN(get_logger(), "cmd_vel size mismatch: got %zu need %zu", msg->data.size(), joints_.size());
    return;
  }
  std::vector<float> velf(joints_.size());
  for (size_t i = 0; i < joints_.size(); i++) velf[i] = (float)msg->data[i]; // rad/s
  auto payload = packFloat32Array(velf);
  sendOp(OpCode::CMD_VEL, payload);
}

bool RobotHwNode::requestTelemetry() {
  // gửi READ_STATE, rồi đọc về 1 frame telemetry
  sendOp(OpCode::READ_STATE);

  std::vector<uint8_t> rx;
  if (!serial_.readSome(rx, 1024, 5)) return false;

  // nếu bạn dùng stream thật: rx_accum_.insert(...); rồi parse nhiều frame
  auto f = tryUnpackFrame(rx);
  if (!f.has_value()) return false;

  JointSample js;
  StatusFlags st;
  if (!unpackTelemetry(*f, js, st, joints_.size())) return false;

  sensor_msgs::msg::JointState out;
  out.header.stamp = now();
  out.name = joints_;
  out.position = js.pos;
  out.velocity = js.vel;
  pub_js_->publish(out);

  std_msgs::msg::UInt64 stmsg;
  stmsg.data = st.flags;
  pub_status_->publish(stmsg);

  return true;
}

void RobotHwNode::pollLoop() {
  if (!serial_.isOpen()) return;
  (void)requestTelemetry();
}

void RobotHwNode::onHome(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                         std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
  sendOp(OpCode::HOME);
  res->success = true;
  res->message = "HOME sent";
}

void RobotHwNode::onStop(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                         std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
  // bạn có thể map STOP -> CMD_VEL all zero hoặc opcode riêng
  sendOp(OpCode::NOP);
  res->success = true;
  res->message = "STOP sent (NOP placeholder)";
}

void RobotHwNode::onEstop(const std::shared_ptr<std_srvs::srv::SetBool::Request> req,
                          std::shared_ptr<std_srvs::srv::SetBool::Response> res) {
  sendOp(req->data ? OpCode::ESTOP : OpCode::CLEAR_ESTOP);
  res->success = true;
  res->message = req->data ? "ESTOP sent" : "CLEAR_ESTOP sent";
}

} // namespace robot_hardware
