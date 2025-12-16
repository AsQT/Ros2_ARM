#include "fas_protocol.hpp"
#include <cstring>

namespace robot_hardware {

// Frame demo: [0xAA][op][len][payload...][0x55]
std::vector<uint8_t> packFrame(const Frame& f) {
  std::vector<uint8_t> out;
  out.push_back(0xAA);
  out.push_back(static_cast<uint8_t>(f.op));
  out.push_back(static_cast<uint8_t>(f.payload.size()));
  out.insert(out.end(), f.payload.begin(), f.payload.end());
  out.push_back(0x55);
  return out;
}

std::optional<Frame> tryUnpackFrame(const std::vector<uint8_t>& rx) {
  if (rx.size() < 4) return std::nullopt;
  // tìm header/footer đơn giản
  size_t i = 0;
  while (i < rx.size() && rx[i] != 0xAA) i++;
  if (i + 3 >= rx.size()) return std::nullopt;

  uint8_t op  = rx[i + 1];
  uint8_t len = rx[i + 2];
  if (i + 3 + len >= rx.size()) return std::nullopt;
  if (rx[i + 3 + len] != 0x55) return std::nullopt;

  Frame f;
  f.op = static_cast<OpCode>(op);
  f.payload.assign(rx.begin() + (long)(i + 3), rx.begin() + (long)(i + 3 + len));
  return f;
}

std::vector<uint8_t> packFloat32Array(const std::vector<float>& v) {
  std::vector<uint8_t> out;
  out.resize(v.size() * 4);
  std::memcpy(out.data(), v.data(), out.size());
  return out;
}

// Telemetry demo: op=READ_STATE trả về payload:
// [N*float32 pos][N*float32 vel][uint64 flags]
bool unpackTelemetry(const Frame& f, JointSample& js, StatusFlags& st, size_t n) {
  const size_t need = n*4 + n*4 + 8;
  if (f.payload.size() < need) return false;

  js.pos.resize(n);
  js.vel.resize(n);

  std::vector<float> posf(n), velf(n);
  std::memcpy(posf.data(), f.payload.data(), n*4);
  std::memcpy(velf.data(), f.payload.data() + n*4, n*4);

  for (size_t i = 0; i < n; i++) {
    js.pos[i] = (double)posf[i]; // giả sử payload đã là rad
    js.vel[i] = (double)velf[i];
  }

  uint64_t flags = 0;
  std::memcpy(&flags, f.payload.data() + n*8, 8);
  st.flags = flags;
  return true;
}

} // namespace robot_hardware
