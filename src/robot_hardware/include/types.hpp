#pragma once
#include <cstdint>
#include <vector>
#include <string>

namespace robot_hardware {

struct JointSample {
  std::vector<double> pos; // rad
  std::vector<double> vel; // rad/s
};

struct StatusFlags {
  uint64_t flags = 0; // 8 bytes -> 64 flags
};

}
