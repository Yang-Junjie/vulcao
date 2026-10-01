#include "vulcao/allocator.h"

#include "vulcao/check.h"

#include <utility>

namespace vulcao {

Allocator::~Allocator() {
    destroy();
}

Allocator::Allocator(Allocator&& other) noexcept
    : allocator_(std::exchange(other.allocator_, nullptr)),
      device_(std::exchange(other.device_, vk::Device{})) {}

Allocator& Allocator::operator=(Allocator&& other) noexcept {
    if (this != &other) {
        destroy();
        allocator_ = std::exchange(other.allocator_, nullptr);
        device_ = std::exchange(other.device_, vk::Device{});
    }
    return *this;
}

void Allocator::create(vk::Instance instance,
                       vk::PhysicalDevice physical_device,
                       vk::Device device,
                       uint32_t api_version,
                       bool buffer_device_address) {
    VmaAllocatorCreateInfo create_info{};
    create_info.instance = instance;
    create_info.physicalDevice = physical_device;
    create_info.device = device;
    create_info.vulkanApiVersion = api_version;
    if (buffer_device_address)
        create_info.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

    check(static_cast<vk::Result>(vmaCreateAllocator(&create_info, &allocator_)), "create allocator");
    device_ = device;
}

void Allocator::destroy() {
    if (allocator_)
        vmaDestroyAllocator(allocator_);
    allocator_ = nullptr;
    device_ = nullptr;
}

}
