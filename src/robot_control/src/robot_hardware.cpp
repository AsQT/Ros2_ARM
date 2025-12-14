#include "robot_control/robot_hardware.hpp"

#include <stdexcept>

namespace robot_control
{

hardware_interface::CallbackReturn RobotHardware::on_init(const hardware_interface::HardwareInfo & info)
{
  if (hardware_interface::SystemInterface::on_init(info) != hardware_interface::CallbackReturn::SUCCESS) {
    return hardware_interface::CallbackReturn::ERROR;
  }

  num_joints_ = info_.joints.size();
  joint_names_.reserve(num_joints_);

  pos_.assign(num_joints_, 0.0);
  vel_.assign(num_joints_, 0.0);
  eff_.assign(num_joints_, 0.0);

  cmd_pos_.assign(num_joints_, 0.0);
  cmd_vel_.assign(num_joints_, 0.0);

  // Read params from <ros2_control><hardware><param ...>
  port_ = info_.hardware_parameters.count("port") ? info_.hardware_parameters.at("port") : "/dev/ttyUSB0";
  baud_ = info_.hardware_parameters.count("baud") ? std::stoi(info_.hardware_parameters.at("baud")) : 115200;
  pos_scale_ = info_.hardware_parameters.count("pos_scale") ? std::stod(info_.hardware_parameters.at("pos_scale")) : 1.0;

  for (const auto & j : info_.joints) {
    joint_names_.push_back(j.name);
  }

  RCLCPP_INFO(rclcpp::get_logger("RobotHardware"),
              "Initialized RobotHardware with %zu joints, port=%s baud=%d",
              num_joints_, port_.c_str(), baud_);

  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> RobotHardware::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;
  state_interfaces.reserve(num_joints_ * 3);

  for (size_t i = 0; i < num_joints_; ++i) {
    state_interfaces.emplace_back(joint_names_[i], hardware_interface::HW_IF_POSITION, &pos_[i]);
    state_interfaces.emplace_back(joint_names_[i], hardware_interface::HW_IF_VELOCITY, &vel_[i]);
    state_interfaces.emplace_back(joint_names_[i], hardware_interface::HW_IF_EFFORT,   &eff_[i]);
  }
  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface> RobotHardware::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> cmd_interfaces;
  cmd_interfaces.reserve(num_joints_ * 2);

  for (size_t i = 0; i < num_joints_; ++i) {
    cmd_interfaces.emplace_back(joint_names_[i], hardware_interface::HW_IF_POSITION, &cmd_pos_[i]);
    cmd_interfaces.emplace_back(joint_names_[i], hardware_interface::HW_IF_VELOCITY, &cmd_vel_[i]);
  }
  return cmd_interfaces;
}

hardware_interface::CallbackReturn RobotHardware::on_configure(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(rclcpp::get_logger("RobotHardware"), "Configuring...");
  if (!connect_hw()) {
    RCLCPP_ERROR(rclcpp::get_logger("RobotHardware"), "Failed to connect HW");
    return hardware_interface::CallbackReturn::ERROR;
  }
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn RobotHardware::on_activate(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(rclcpp::get_logger("RobotHardware"), "Activating...");
  // Optionally sync command with current state
  cmd_pos_ = pos_;
  std::fill(cmd_vel_.begin(), cmd_vel_.end(), 0.0);
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn RobotHardware::on_deactivate(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(rclcpp::get_logger("RobotHardware"), "Deactivating...");
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::return_type RobotHardware::read(const rclcpp::Time &, const rclcpp::Duration &)
{
  if (!hw_read(pos_, vel_)) {
    RCLCPP_WARN(rclcpp::get_logger("RobotHardware"), "HW read failed");
    return hardware_interface::return_type::ERROR;
  }
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type RobotHardware::write(const rclcpp::Time &, const rclcpp::Duration &)
{
  if (!hw_write(cmd_pos_, cmd_vel_)) {
    RCLCPP_WARN(rclcpp::get_logger("RobotHardware"), "HW write failed");
    return hardware_interface::return_type::ERROR;
  }
  return hardware_interface::return_type::OK;
}

/* --------------------- TODO: implement real RS-485 --------------------- */
bool RobotHardware::connect_hw()
{
  // TODO: open serial/RS485, init STM32 protocol
  RCLCPP_INFO(rclcpp::get_logger("RobotHardware"), "Connecting to %s @ %d ...", port_.c_str(), baud_);
  return true;
}

void RobotHardware::disconnect_hw()
{
  // TODO: close port
}

bool RobotHardware::hw_read(std::vector<double> & pos, std::vector<double> & vel)
{
  // TODO: read from STM32, convert to rad
  (void)vel;
  // fake: keep pos unchanged
  return true;
}

bool RobotHardware::hw_write(const std::vector<double> & cmd_pos, const std::vector<double> & cmd_vel)
{
  // TODO: send cmd to STM32
  (void)cmd_pos;
  (void)cmd_vel;
  return true;
}

}  // namespace robot_control
