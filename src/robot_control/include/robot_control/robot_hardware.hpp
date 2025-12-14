#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

#include "rclcpp/rclcpp.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/handle.hpp"
#include "hardware_interface/component_parser.hpp"

namespace robot_control
{

class RobotHardware : public hardware_interface::SystemInterface
{
public:
  RCLCPP_SHARED_PTR_DEFINITIONS(RobotHardware)

  hardware_interface::CallbackReturn on_init(const hardware_interface::HardwareInfo & info) override;

  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  hardware_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::return_type read(const rclcpp::Time & time, const rclcpp::Duration & period) override;
  hardware_interface::return_type write(const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  // Joint data
  size_t num_joints_{0};
  std::vector<std::string> joint_names_;

  // State
  std::vector<double> pos_;
  std::vector<double> vel_;
  std::vector<double> eff_;

  // Command
  std::vector<double> cmd_pos_;
  std::vector<double> cmd_vel_;

  // HW params
  std::string port_;
  int baud_{115200};
  double pos_scale_{1.0};  // example: rad -> device units

  // TODO: replace with your real RS-485 driver class
  bool connect_hw();
  void disconnect_hw();
  bool hw_read(std::vector<double> & pos, std::vector<double> & vel);
  bool hw_write(const std::vector<double> & cmd_pos, const std::vector<double> & cmd_vel);
};

}  // namespace robot_control
