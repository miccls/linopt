#pragma once

#include <Metal/MTLCommandQueue.hpp>
#include <Metal/MTLDevice.hpp>
#include <stdexcept>

namespace pdhg::metal {

class MetalError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class Context {
 public:
  Context();
  ~Context();

  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  Context(Context&& other) noexcept;
  Context& operator=(Context&& other) noexcept;

  [[nodiscard]] MTL::Device* device() const;
  [[nodiscard]] MTL::CommandQueue* commandQueue() const;

 private:
  void releaseResources() noexcept;

  MTL::Device* device_ = nullptr;
  MTL::CommandQueue* command_queue_ = nullptr;
};

}  // namespace pdhg::metal
