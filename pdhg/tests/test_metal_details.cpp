#include <gtest/gtest.h>

#include <Metal/MTLDevice.hpp>
#include <concepts>
#include <string_view>
#include <utility>

#include "metal_details.h"

namespace pdhg::metal {
namespace {

bool hasDefaultMetalDevice() {
  MTL::Device* device = MTL::CreateSystemDefaultDevice();
  if (device == nullptr) {
    return false;
  }
  device->release();
  return true;
}

template <typename Exception, std::invocable Callable>
void expectThrowsWithMessage(Callable&& callable, std::string_view expected_message) {
  try {
    std::forward<Callable>(callable)();
    FAIL() << "Expected exception";
  } catch (const Exception& exception) {
    EXPECT_EQ(exception.what(), expected_message);
  } catch (...) {
    FAIL() << "Expected a different exception type";
  }
}

TEST(MetalContext, CreatesDeviceAndCommandQueue) {
  if (!hasDefaultMetalDevice()) {
    GTEST_SKIP() << "No default Metal device is available";
  }

  Context context;

  EXPECT_NE(context.device(), nullptr);
  EXPECT_NE(context.commandQueue(), nullptr);
}

TEST(MetalContext, MoveConstructorTransfersOwnership) {
  if (!hasDefaultMetalDevice()) {
    GTEST_SKIP() << "No default Metal device is available";
  }

  Context source;
  Context destination(std::move(source));

  EXPECT_NE(destination.device(), nullptr);
  EXPECT_NE(destination.commandQueue(), nullptr);
  expectThrowsWithMessage<MetalError>([&] { (void)source.device(); },
                                      "Metal context has no device");
  expectThrowsWithMessage<MetalError>([&] { (void)source.commandQueue(); },
                                      "Metal context has no command queue");
}

TEST(MetalContext, MoveAssignmentTransfersOwnership) {
  if (!hasDefaultMetalDevice()) {
    GTEST_SKIP() << "No default Metal device is available";
  }

  Context source;
  Context destination;

  destination = std::move(source);

  EXPECT_NE(destination.device(), nullptr);
  EXPECT_NE(destination.commandQueue(), nullptr);
  expectThrowsWithMessage<MetalError>([&] { (void)source.device(); },
                                      "Metal context has no device");
  expectThrowsWithMessage<MetalError>([&] { (void)source.commandQueue(); },
                                      "Metal context has no command queue");
}

}  // namespace
}  // namespace pdhg::metal
