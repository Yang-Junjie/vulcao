#pragma once

#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.h>

namespace vulcao {

/// @brief RAII wrapper around a VMA allocator.
class Allocator {
public:
    /// @brief Creates an empty allocator.
    Allocator() = default;

    /// @brief Destroys the allocator and all remaining allocations.
    ~Allocator();

    /// @brief Not copyable.
    Allocator(const Allocator&) = delete;
    Allocator& operator=(const Allocator&) = delete;

    /// @brief Moves the allocator, leaving the source empty.
    Allocator(Allocator&& other) noexcept;

    /// @brief Move assignment. Destroys the current allocator first.
    Allocator& operator=(Allocator&& other) noexcept;

    /// @brief Returns true if the allocator holds a valid handle.
    bool valid() const { return allocator_ != nullptr; }

    /// @brief Returns true if the allocator holds a valid handle.
    explicit operator bool() const { return valid(); }

    /// @brief Returns the raw VMA allocator handle.
    VmaAllocator handle() const { return allocator_; }

    /// @brief Returns the device the allocator was created with.
    vk::Device device() const { return device_; }

private:
    friend class Context;

    /// @brief Creates the allocator. Called by Context.
    /// @param buffer_device_address True to allow buffers with device addresses.
    void create(vk::Instance instance,
                vk::PhysicalDevice physical_device,
                vk::Device device,
                uint32_t api_version,
                bool buffer_device_address);

    /// @brief Destroys the allocator if it exists.
    void destroy();

    VmaAllocator allocator_ = nullptr;
    vk::Device device_;
};

}
