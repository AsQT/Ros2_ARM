#include "serial_port.hpp"

#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <sys/select.h>
#include <cstring>

namespace robot_hardware {

static speed_t to_speed(int baud) {
  switch (baud) {
    case 9600: return B9600;
    case 19200: return B19200;
    case 38400: return B38400;
    case 57600: return B57600;
    case 115200: return B115200;
    case 230400: return B230400;
    default: return B115200;
  }
}

SerialPort::SerialPort() = default;
SerialPort::~SerialPort() { close(); }

bool SerialPort::open(const std::string& port, int baud) {
  std::lock_guard<std::mutex> lk(mtx_);
  if (fd_ >= 0) return true;

  fd_ = ::open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
  if (fd_ < 0) return false;

  termios tty{};
  if (tcgetattr(fd_, &tty) != 0) {
    ::close(fd_); fd_ = -1;
    return false;
  }

  cfmakeraw(&tty);
  tty.c_cflag |= (CLOCAL | CREAD);
  tty.c_cflag &= ~CSTOPB;
  tty.c_cflag &= ~CRTSCTS;
  tty.c_cflag &= ~PARENB;
  tty.c_cflag &= ~CSIZE;
  tty.c_cflag |= CS8;

  speed_t spd = to_speed(baud);
  cfsetispeed(&tty, spd);
  cfsetospeed(&tty, spd);

  tty.c_cc[VMIN]  = 0;
  tty.c_cc[VTIME] = 0;

  if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
    ::close(fd_); fd_ = -1;
    return false;
  }
  return true;
}

void SerialPort::close() {
  std::lock_guard<std::mutex> lk(mtx_);
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

bool SerialPort::isOpen() const {
  std::lock_guard<std::mutex> lk(mtx_);
  return fd_ >= 0;
}

bool SerialPort::writeBytes(const std::vector<uint8_t>& data) {
  std::lock_guard<std::mutex> lk(mtx_);
  if (fd_ < 0) return false;
  const uint8_t* p = data.data();
  size_t n = data.size();
  while (n > 0) {
    ssize_t w = ::write(fd_, p, n);
    if (w < 0) return false;
    p += (size_t)w;
    n -= (size_t)w;
  }
  return true;
}

bool SerialPort::readSome(std::vector<uint8_t>& out, int max_bytes, int timeout_ms) {
  out.clear();
  std::lock_guard<std::mutex> lk(mtx_);
  if (fd_ < 0) return false;

  fd_set rfds;
  FD_ZERO(&rfds);
  FD_SET(fd_, &rfds);

  timeval tv{};
  tv.tv_sec  = timeout_ms / 1000;
  tv.tv_usec = (timeout_ms % 1000) * 1000;

  int ret = select(fd_ + 1, &rfds, nullptr, nullptr, &tv);
  if (ret <= 0) return false;

  std::vector<uint8_t> buf((size_t)max_bytes);
  ssize_t r = ::read(fd_, buf.data(), buf.size());
  if (r <= 0) return false;

  buf.resize((size_t)r);
  out = std::move(buf);
  return true;
}

} // namespace robot_hardware
