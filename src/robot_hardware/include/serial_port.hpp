#pragma once
#include <string>
#include <vector>
#include <mutex>
#include <chrono>

namespace robot_hardware {

class SerialPort {
public:
  SerialPort();
  ~SerialPort();

  bool open(const std::string& port, int baud);
  void close();
  bool isOpen() const;

  bool writeBytes(const std::vector<uint8_t>& data);

  bool readSome(std::vector<uint8_t>& out, int max_bytes, int timeout_ms);

private:
  int fd_ = -1;
  mutable std::mutex mtx_;
};

} 