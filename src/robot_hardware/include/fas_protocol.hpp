#pragma once
#include <cstdint>
#include <vector>
#include <optional>
#include "types.hpp"

namespace robot_hardware {

enum class OpCode : uint8_t {
  NOP         = 0x00,
  CMD_POS     = 0x10,
  CMD_VEL     = 0x11,
  HOME        = 0x20,
  JOG         = 0x21,
  READ_STATE  = 0x30,
  ESTOP       = 0xE0,
  CLEAR_ESTOP = 0xE1,
};

struct Frame {
  OpCode op;
  std::vector<uint8_t> payload;
};

std::vector<uint8_t> packFrame(const Frame& f);
std::optional<Frame> tryUnpackFrame(const std::vector<uint8_t>& rx_stream);

std::vector<uint8_t> packFloat32Array(const std::vector<float>& v);
bool unpackTelemetry(const Frame& f, JointSample& js, StatusFlags& st, size_t n_joints);

} 
