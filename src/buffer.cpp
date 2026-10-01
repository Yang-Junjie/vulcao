#include "vulcao/buffer.h"

#include "vulcao/check.h"
#include "vulcao/context.h"

#include <stdexcept>
#include <utility>

namespace vulcao {

Buffer::~Buffer() {
    destroy();
}

Buffer::Buffer(Buffer&& other) noexcept
    : allocator_(std::exchange(other.allocator_, nullptr)),
      device_(std::exchange(other.device_, vk::Device{})),
      buffer_(std::exchange(other.buffer_, VK_NULL_HANDLE)),
      allocation_(std::exchange(other.allocation_, nullptr)),
      info_(std::exchange(other.info_, VmaAllocationInfo{})),
      size_(std::exchange(other.size_, 0)),
      usage_(std::exchange(other.usage_, vk::BufferUsageFlags{})),
      sharing_mode_(std::exchange(other.sharing_mode_, vk::SharingMode::eExclusive)),
      mapped_data_(std::exchange(other.mapped_data_, nullptr)),
      mapped_(std::exchange(other.mapped_, false)),
      host_visible_(std::exchange(other.host_visible_, false)) {}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        destroy();
        allocator_ = std::exchange(other.allocator_, nullptr);
        device_ = std::exchange(other.device_, vk::Device{});
        buffer_ = std::exchange(other.buffer_, VK_NULL_HANDLE);
        allocation_ = std::exchange(other.allocation_, nullptr);
        info_ = std::exchange(other.info_, VmaAllocationInfo{});
        size_ = std::exchange(other.size_, 0);
        usage_ = std::exchange(other.usage_, vk::BufferUsageFlags{});
        sharing_mode_ = std::exchange(other.sharing_mode_, vk::SharingMode::eExclusive);
        mapped_data_ = std::exchange(other.mapped_data_, nullptr);
        mapped_ = std::exchange(other.mapped_, false);
        host_visible_ = std::exchange(other.host_visible_, false);
    }
    return *this;
}

Buffer Buffer::create(Allocator& allocator,
                      vk::DeviceSize size,
                      vk::BufferUsageFlags usage,
                      VmaMemoryUsage memory_usage,
                      VmaAllocationCreateFlags flags,
                      vk::ArrayProxy<const uint32_t> concurrent_families) {
    if (!allocator.valid())
        throw std::runtime_error("Buffer::create: invalid allocator");

    Buffer buffer;
    buffer.allocator_ = allocator.handle();
    buffer.device_ = allocator.device();

    const bool concurrent = concurrent_families.size() >= 2;
    const vk::BufferCreateInfo create_info{
        .size = size,
        .usage = usage,
        .sharingMode = concurrent ? vk::SharingMode::eConcurrent : vk::SharingMode::eExclusive,
        .queueFamilyIndexCount = concurrent ? static_cast<uint32_t>(concurrent_families.size()) : 0u,
        .pQueueFamilyIndices = concurrent ? concurrent_families.data() : nullptr,
    };

    VmaAllocationCreateInfo allocation_info{};
    allocation_info.usage = memory_usage;
    allocation_info.flags = flags;

    const VkBufferCreateInfo raw_create_info = create_info;
    check(static_cast<vk::Result>(vmaCreateBuffer(buffer.allocator_, &raw_create_info, &allocation_info,
                                                  &buffer.buffer_, &buffer.allocation_, &buffer.info_)),
          "create buffer");
    buffer.size_ = size;
    buffer.usage_ = usage;
    buffer.sharing_mode_ = create_info.sharingMode;

    VkMemoryPropertyFlags memory_flags = 0;
    vmaGetMemoryTypeProperties(buffer.allocator_, buffer.info_.memoryType, &memory_flags);
    buffer.host_visible_ = (memory_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;

    return buffer;
}

Buffer Buffer::create_with_data(Context& context,
                                const void* data,
                                vk::DeviceSize size,
                                vk::BufferUsageFlags usage,
                                VmaMemoryUsage memory_usage,
                                VmaAllocationCreateFlags flags) {
    Buffer buffer = create(context.allocator(), size,
                           usage | vk::BufferUsageFlagBits::eTransferDst, memory_usage, flags);
    context.upload(buffer, data, size);
    return buffer;
}

vk::DeviceAddress Buffer::device_address() const {
    if (buffer_ == VK_NULL_HANDLE)
        throw std::runtime_error("Buffer::device_address: invalid buffer");
    if (!(usage_ & vk::BufferUsageFlagBits::eShaderDeviceAddress))
        throw std::runtime_error("Buffer::device_address: buffer requires eShaderDeviceAddress usage");

    return device_.getBufferAddress(vk::BufferDeviceAddressInfo{.buffer = vk::Buffer{buffer_}});
}

void* Buffer::map() {
    if (info_.pMappedData)
        return info_.pMappedData;

    if (!host_visible_)
        throw std::runtime_error("Buffer::map: buffer is not host visible");

    if (!mapped_) {
        check(static_cast<vk::Result>(vmaMapMemory(allocator_, allocation_, &mapped_data_)), "map buffer");
        mapped_ = true;
    }
    return mapped_data_;
}

void Buffer::unmap() {
    if (mapped_) {
        vmaUnmapMemory(allocator_, allocation_);
        mapped_ = false;
        mapped_data_ = nullptr;
    }
}

void Buffer::flush(vk::DeviceSize offset, vk::DeviceSize size) {
    check(static_cast<vk::Result>(vmaFlushAllocation(allocator_, allocation_, offset, size)), "flush buffer");
}

void Buffer::invalidate(vk::DeviceSize offset, vk::DeviceSize size) {
    check(static_cast<vk::Result>(vmaInvalidateAllocation(allocator_, allocation_, offset, size)),
          "invalidate buffer");
}

void Buffer::write_bytes(const void* data, vk::DeviceSize size, vk::DeviceSize offset) {
    if (size == 0)
        return;
    if (offset + size > size_)
        throw std::runtime_error("Buffer::write_bytes: out of range");
    if (!host_visible_)
        throw std::runtime_error("Buffer::write_bytes: buffer is not host visible, use Context::upload");

    check(static_cast<vk::Result>(vmaCopyMemoryToAllocation(allocator_, data, allocation_, offset, size)),
          "write buffer");
}

void Buffer::destroy() {
    if (mapped_)
        vmaUnmapMemory(allocator_, allocation_);
    if (buffer_ != VK_NULL_HANDLE)
        vmaDestroyBuffer(allocator_, buffer_, allocation_);

    allocator_ = nullptr;
    device_ = nullptr;
    buffer_ = VK_NULL_HANDLE;
    allocation_ = nullptr;
    info_ = {};
    size_ = 0;
    usage_ = {};
    sharing_mode_ = vk::SharingMode::eExclusive;
    mapped_data_ = nullptr;
    mapped_ = false;
    host_visible_ = false;
}

BufferView::~BufferView() {
    destroy();
}

BufferView::BufferView(BufferView&& other) noexcept
    : device_(std::exchange(other.device_, vk::Device{})),
      view_(std::exchange(other.view_, vk::BufferView{})) {}

BufferView& BufferView::operator=(BufferView&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = std::exchange(other.device_, vk::Device{});
        view_ = std::exchange(other.view_, vk::BufferView{});
    }
    return *this;
}

BufferView BufferView::create(vk::Device device,
                              const Buffer& buffer,
                              vk::Format format,
                              vk::DeviceSize offset,
                              vk::DeviceSize range) {
    if (!buffer.valid())
        throw std::runtime_error("BufferView::create: invalid buffer");
    if (!(buffer.usage() & (vk::BufferUsageFlagBits::eUniformTexelBuffer |
                            vk::BufferUsageFlagBits::eStorageTexelBuffer)))
        throw std::runtime_error(
            "BufferView::create: buffer requires UniformTexelBuffer or StorageTexelBuffer usage");

    BufferView view;
    view.device_ = device;
    view.view_ = device.createBufferView(vk::BufferViewCreateInfo{
        .buffer = buffer.handle(),
        .format = format,
        .offset = offset,
        .range = range,
    });
    return view;
}

void BufferView::destroy() {
    if (view_)
        device_.destroyBufferView(view_);

    device_ = nullptr;
    view_ = nullptr;
}

}
