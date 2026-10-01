#include "vulcao/fence.h"

#include "vulcao/detail/check.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace vulcao {

Fence::~Fence() {
    destroy();
}

Fence::Fence(Fence&& other) noexcept
    : device_(std::exchange(other.device_, vk::Device{})),
      fence_(std::exchange(other.fence_, vk::Fence{})) {}

Fence& Fence::operator=(Fence&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = std::exchange(other.device_, vk::Device{});
        fence_ = std::exchange(other.fence_, vk::Fence{});
    }
    return *this;
}

Fence Fence::create(vk::Device device, vk::FenceCreateFlags flags) {
    Fence fence;
    fence.device_ = device;
    fence.fence_ = device.createFence(vk::FenceCreateInfo{.flags = flags});
    return fence;
}

void Fence::wait(uint64_t timeout) const {
    detail::check(device_.waitForFences(fence_, VK_TRUE, timeout), "wait for fence");
}

bool Fence::wait_for(uint64_t timeout) const {
    const vk::Result result = device_.waitForFences(fence_, VK_TRUE, timeout);
    if (result == vk::Result::eSuccess)
        return true;
    if (result == vk::Result::eTimeout)
        return false;
    throw std::runtime_error("Fence::wait_for failed: " + vk::to_string(result));
}

bool Fence::signaled() const {
    return device_.getFenceStatus(fence_) == vk::Result::eSuccess;
}

void Fence::reset() const {
    device_.resetFences(fence_);
}

void Fence::destroy() {
    if (fence_)
        device_.destroyFence(fence_);

    device_ = nullptr;
    fence_ = nullptr;
}

}
