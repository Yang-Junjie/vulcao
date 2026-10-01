#pragma once

#include <cstddef>
#include <ranges>
#include <type_traits>

#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.h>

#include "vulcao/allocator.h"

namespace vulcao {

class Context;

/// @brief RAII wrapper around a VMA-allocated buffer.
class Buffer {
public:
    /// @brief Creates an empty buffer.
    Buffer() = default;

    /// @brief Frees the buffer and its memory.
    ~Buffer();

    /// @brief Not copyable.
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    /// @brief Moves the buffer, leaving the source empty.
    Buffer(Buffer&& other) noexcept;

    /// @brief Move assignment. Destroys the current buffer first.
    Buffer& operator=(Buffer&& other) noexcept;

    /// @brief Creates a buffer and allocates its memory.
    /// @param allocator Allocator used for the memory.
    /// @param size Buffer size in bytes.
    /// @param usage Buffer usage flags.
    /// @param memory_usage VMA memory usage hint.
    /// @param flags VMA allocation flags.
    /// @param concurrent_families Queue families that share the buffer; two or
    ///        more make it concurrent (see Context::transfer_sharing_families
    ///        for asynchronous transfer uploads), empty keeps it exclusive.
    /// @return The created buffer.
    /// @throws std::runtime_error if the allocator is invalid or the buffer cannot be created.
    static Buffer create(Allocator& allocator,
                         vk::DeviceSize size,
                         vk::BufferUsageFlags usage,
                         VmaMemoryUsage memory_usage = VMA_MEMORY_USAGE_AUTO,
                         VmaAllocationCreateFlags flags = 0,
                         vk::ArrayProxy<const uint32_t> concurrent_families = {});

    /// @brief Creates a buffer and uploads data into it through the context.
    ///
    /// Folds the recurring create-then-upload pair into one call. TransferDst is
    /// added to @p usage automatically; the default memory usage keeps the
    /// buffer device local.
    /// @param context Context used for the allocation and the upload.
    /// @param data Source pointer.
    /// @param size Number of bytes to create and upload.
    /// @param usage Buffer usage flags, TransferDst is added.
    /// @param memory_usage VMA memory usage hint.
    /// @param flags VMA allocation flags.
    /// @return The created and uploaded buffer.
    /// @throws std::runtime_error if creation or upload fails.
    static Buffer create_with_data(Context& context,
                                   const void* data,
                                   vk::DeviceSize size,
                                   vk::BufferUsageFlags usage,
                                   VmaMemoryUsage memory_usage = VMA_MEMORY_USAGE_AUTO,
                                   VmaAllocationCreateFlags flags = 0);

    /// @brief Creates a buffer and uploads a contiguous range into it.
    /// @tparam Container Contiguous range of trivially copyable values.
    /// @param context Context used for the allocation and the upload.
    /// @param data Source range.
    /// @param usage Buffer usage flags, TransferDst is added.
    /// @param memory_usage VMA memory usage hint.
    /// @param flags VMA allocation flags.
    /// @return The created and uploaded buffer.
    /// @throws std::runtime_error if creation or upload fails.
    template <typename Container>
        requires std::ranges::contiguous_range<Container>
    static Buffer create_with_data(Context& context,
                                   const Container& data,
                                   vk::BufferUsageFlags usage,
                                   VmaMemoryUsage memory_usage = VMA_MEMORY_USAGE_AUTO,
                                   VmaAllocationCreateFlags flags = 0) {
        using T = std::ranges::range_value_t<Container>;
        static_assert(std::is_trivially_copyable_v<T>, "buffer data must be trivially copyable");
        // Only forwards the context by reference, so the forward declaration of
        // Context above is enough; the member access lives in the raw overload.
        return create_with_data(context, std::ranges::data(data),
                                static_cast<vk::DeviceSize>(std::ranges::size(data)) * sizeof(T),
                                usage, memory_usage, flags);
    }

    /// @brief Returns true if the buffer holds a valid handle.
    bool valid() const { return buffer_ != VK_NULL_HANDLE; }

    /// @brief Returns true if the buffer holds a valid handle.
    explicit operator bool() const { return valid(); }

    /// @brief Returns the raw Vulkan buffer handle.
    vk::Buffer handle() const { return vk::Buffer{buffer_}; }

    /// @brief Returns the buffer size in bytes.
    vk::DeviceSize size() const { return size_; }

    /// @brief Returns the usage flags the buffer was created with.
    vk::BufferUsageFlags usage() const { return usage_; }

    /// @brief Returns the sharing mode the buffer was created with.
    vk::SharingMode sharing_mode() const { return sharing_mode_; }

    /// @brief Returns the VMA allocation of the buffer.
    VmaAllocation allocation() const { return allocation_; }

    /// @brief Returns information about the VMA allocation.
    const VmaAllocationInfo& allocation_info() const { return info_; }

    /// @brief Returns the device address of the buffer.
    ///
    /// The buffer must have been created with eShaderDeviceAddress, which the
    /// buffer device address feature makes valid.
    /// @return The buffer device address.
    /// @throws std::runtime_error if the buffer is invalid or lacks
    ///         eShaderDeviceAddress usage.
    vk::DeviceAddress device_address() const;

    /// @brief Returns true if the memory is host visible.
    bool host_visible() const { return host_visible_; }

    /// @brief Maps the memory and returns a pointer to it.
    /// @return Pointer to the mapped memory.
    /// @throws std::runtime_error if the buffer is not host visible or mapping fails.
    void* map();

    /// @brief Unmaps the memory if it was mapped by map().
    void unmap();

    /// @brief Flushes a range of the allocation to make writes visible to the device.
    /// @param offset Byte offset of the range.
    /// @param size Size of the range, or VK_WHOLE_SIZE.
    /// @throws std::runtime_error if the flush fails.
    void flush(vk::DeviceSize offset = 0, vk::DeviceSize size = VK_WHOLE_SIZE);

    /// @brief Invalidates a range of the allocation to make device writes visible to the host.
    /// @param offset Byte offset of the range.
    /// @param size Size of the range, or VK_WHOLE_SIZE.
    /// @throws std::runtime_error if the invalidate fails.
    void invalidate(vk::DeviceSize offset = 0, vk::DeviceSize size = VK_WHOLE_SIZE);

    /// @brief Copies raw bytes into a host visible buffer.
    /// @param data Source pointer.
    /// @param size Number of bytes to copy.
    /// @param offset Byte offset in the buffer.
    /// @throws std::runtime_error if the range is out of bounds, the buffer is not host
    ///         visible, or the copy fails.
    void write_bytes(const void* data, vk::DeviceSize size, vk::DeviceSize offset = 0);

    /// @brief Copies a contiguous range of trivially copyable values into a host visible buffer.
    /// @tparam Container Contiguous range of trivially copyable values.
    /// @param data Source range.
    /// @param offset Byte offset in the buffer.
    template <typename Container>
        requires std::ranges::contiguous_range<Container>
    void write(const Container& data, vk::DeviceSize offset = 0) {
        using T = std::ranges::range_value_t<Container>;
        static_assert(std::is_trivially_copyable_v<T>, "buffer data must be trivially copyable");
        write_bytes(std::ranges::data(data),
                    static_cast<vk::DeviceSize>(std::ranges::size(data)) * sizeof(T),
                    offset);
    }

private:
    /// @brief Frees the buffer and resets the wrapper.
    void destroy();

    VmaAllocator allocator_ = nullptr;
    vk::Device device_;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = nullptr;
    VmaAllocationInfo info_{};
    vk::DeviceSize size_ = 0;
    vk::BufferUsageFlags usage_;
    vk::SharingMode sharing_mode_ = vk::SharingMode::eExclusive;
    void* mapped_data_ = nullptr;
    bool mapped_ = false;
    bool host_visible_ = false;
};

/// @brief RAII wrapper around a Vulkan buffer view, used for texel buffers.
class BufferView {
public:
    /// @brief Creates an empty buffer view.
    BufferView() = default;

    /// @brief Destroys the buffer view.
    ~BufferView();

    /// @brief Not copyable.
    BufferView(const BufferView&) = delete;
    BufferView& operator=(const BufferView&) = delete;

    /// @brief Moves the buffer view, leaving the source empty.
    BufferView(BufferView&& other) noexcept;

    /// @brief Move assignment. Destroys the current buffer view first.
    BufferView& operator=(BufferView&& other) noexcept;

    /// @brief Creates a view of a buffer in a texel format.
    ///
    /// The buffer must have been created with UniformTexelBuffer or
    /// StorageTexelBuffer usage and must outlive the view.
    /// @param device Device that creates the view.
    /// @param buffer Buffer to view.
    /// @param format Format the texels are interpreted as.
    /// @param offset Byte offset of the viewed range.
    /// @param range Byte size of the viewed range, or VK_WHOLE_SIZE.
    /// @return The created buffer view.
    /// @throws std::runtime_error if the buffer is invalid or lacks texel buffer usage.
    static BufferView create(vk::Device device,
                             const Buffer& buffer,
                             vk::Format format,
                             vk::DeviceSize offset = 0,
                             vk::DeviceSize range = VK_WHOLE_SIZE);

    /// @brief Returns true if the view holds a valid handle.
    bool valid() const { return static_cast<bool>(view_); }

    /// @brief Returns true if the view holds a valid handle.
    explicit operator bool() const { return valid(); }

    /// @brief Returns the raw Vulkan buffer view handle.
    vk::BufferView handle() const { return view_; }

    /// @brief Destroys the buffer view and resets the wrapper.
    void destroy();

private:
    vk::Device device_;
    vk::BufferView view_;
};

}
