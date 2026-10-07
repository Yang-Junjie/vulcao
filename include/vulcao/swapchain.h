#pragma once

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.hpp>

namespace vulcao {

/// @brief Parameters used to create a swapchain for an externally owned surface.
struct SwapchainCreateInfo {
    vk::PhysicalDevice physical_device;
    vk::Device device;
    vk::SurfaceKHR surface;
    vk::Extent2D extent;
    vk::Format format = vk::Format::eB8G8R8A8Srgb;
    vk::ColorSpaceKHR color_space = vk::ColorSpaceKHR::eSrgbNonlinear;
    vk::PresentModeKHR present_mode = vk::PresentModeKHR::eFifo;
    uint32_t min_image_count = 2;
    vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eColorAttachment;
    uint32_t graphics_queue_family_index = 0;
    uint32_t present_queue_family_index = 0;
    vk::SwapchainKHR old_swapchain;
};

/// @brief Owns a swapchain and its image views, but not the presentation surface.
class Swapchain {
public:
    struct Acquired {
        uint32_t image_index = 0;
        bool suboptimal = false;
        bool out_of_date = false;
    };

    Swapchain() = default;
    ~Swapchain();

    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;
    Swapchain(Swapchain&& other) noexcept;
    Swapchain& operator=(Swapchain&& other) noexcept;

    static Swapchain create(const SwapchainCreateInfo& info);

    Acquired acquire_next_image(vk::Semaphore signal_semaphore, uint64_t timeout = UINT64_MAX) const;
    bool present(vk::Queue queue, uint32_t image_index, vk::Semaphore wait_semaphore) const;

    bool valid() const { return static_cast<bool>(swapchain_); }
    vk::SwapchainKHR handle() const { return swapchain_; }
    vk::SurfaceKHR surface() const { return surface_; }
    vk::Format format() const { return format_; }
    vk::Extent2D extent() const { return extent_; }
    uint32_t image_count() const { return static_cast<uint32_t>(images_.size()); }
    const std::vector<vk::Image>& images() const { return images_; }
    const std::vector<vk::ImageView>& image_views() const { return image_views_; }

    void destroy();

private:
    vk::Device device_;
    vk::SurfaceKHR surface_;
    vk::SwapchainKHR swapchain_;
    vk::Format format_ = vk::Format::eUndefined;
    vk::Extent2D extent_{};
    std::vector<vk::Image> images_;
    std::vector<vk::ImageView> image_views_;
};

} // namespace vulcao
