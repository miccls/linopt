#include "metal_details.h"

namespace pdhg::metal {

Context::Context() {
  device_ = MTL::CreateSystemDefaultDevice();
  if (device_ == nullptr) {
    throw MetalError("Could not create default Metal device");
  }

  command_queue_ = device_->newCommandQueue();
  if (command_queue_ == nullptr) {
    releaseResources();
    throw MetalError("Could not create Metal command queue");
  }
}

Context::~Context() { releaseResources(); }

Context::Context(Context&& other) noexcept
    : device_(other.device_), command_queue_(other.command_queue_) {
  other.device_ = nullptr;
  other.command_queue_ = nullptr;
}

Context& Context::operator=(Context&& other) noexcept {
  if (this != &other) {
    releaseResources();

    device_ = other.device_;
    command_queue_ = other.command_queue_;

    other.device_ = nullptr;
    other.command_queue_ = nullptr;
  }
  return *this;
}

MTL::Device* Context::device() const {
  if (device_ == nullptr) {
    throw MetalError("Metal context has no device");
  }
  return device_;
}

MTL::CommandQueue* Context::commandQueue() const {
  if (command_queue_ == nullptr) {
    throw MetalError("Metal context has no command queue");
  }
  return command_queue_;
}

void Context::releaseResources() noexcept {
  if (command_queue_ != nullptr) {
    command_queue_->release();
    command_queue_ = nullptr;
  }
  if (device_ != nullptr) {
    device_->release();
    device_ = nullptr;
  }
}

}  // namespace pdhg::metal
