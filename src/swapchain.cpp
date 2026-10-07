#include "vulcao/swapchain.h"

#include <stdexcept>
#include <utility>

#include <VkBootstrap.h>

namespace vulcao {

Swapchain::~Swapchain() { destroy(); }

Swapchain::Swapchain(Swapchain&& other) noexcept
        : device_(std::exchange(other.device_, vk::Device{})),
          surface_(std::exchange(other.surface_, vk::SurfaceKHR{})),
          swapchain_(std::exchange(other.swapchain_, vk::SwapchainKHR{})),
          format_(std::exchange(other.format_, vk::Format::eUndefined)),
          extent_(std::exchange(other.extent_, vk::Extent2D{})),
          images_(std::move(other.images_)),
          image_views_(std::move(other.image_views_)) {}

Swapchain& Swapchain::operator=(Swapchain&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = std::exchange(other.device_, vk::Device{});
        surface_ = std::exchange(other.surface_, vk::SurfaceKHR{});
        swapchain_ = std::exchange(other.swapchain_, vk::SwapchainKHR{});
        format_ = std::exchange(other.format_, vk::Format::eUndefined);
        extent_ = std::exchange(other.extent_, vk::Extent2D{});
        images_ = std::move(other.images_);
        image_views_ = std::move(other.image_views_);
    }
    return *this;
}

Swapchain Swapchain::create(const SwapchainCreateInfo& info) {
    if (!info.physical_device || !info.device || !info.surface)
        throw std::runtime_error("Swapchain::create: physical device, device, and surface are required");
    if (info.extent.width == 0 || info.extent.height == 0)
        throw std::runtime_error("Swapchain::create: extent must be non-zero");

    const vk::SurfaceCapabilitiesKHR capabilities =
        info.physical_device.getSurfaceCapabilitiesKHR(info.surface);
    if (capabilities.currentExtent.width == 0 || capabilities.currentExtent.height == 0)
        throw std::runtime_error("Swapchain::create: surface extent must be non-zero");

    vkb::SwapchainBuilder builder{
        static_cast<VkPhysicalDevice>(info.physical_device),
        static_cast<VkDevice>(info.device),
        static_cast<VkSurfaceKHR>(info.surface),
        info.graphics_queue_family_index,
        info.present_queue_family_index,
    };
    builder.set_desired_extent(info.extent.width, info.extent.height)
        .set_desired_format(VkSurfaceFormatKHR{
            .format = static_cast<VkFormat>(info.format),
            .colorSpace = static_cast<VkColorSpaceKHR>(info.color_space),
        })
        .set_desired_present_mode(static_cast<VkPresentModeKHR>(info.present_mode))
        .set_desired_min_image_count(info.min_image_count)
        .set_image_usage_flags(static_cast<VkImageUsageFlags>(info.usage));
    if (info.old_swapchain)
        builder.set_old_swapchain(static_cast<VkSwapchainKHR>(info.old_swapchain));

    auto built = builder.build();
    if (!built)
        throw std::runtime_error("Swapchain::create: vk-bootstrap failed to create swapchain");

    vkb::Swapchain& raw = built.value();
    Swapchain result;
    result.device_ = info.device;
    result.surface_ = info.surface;
    result.swapchain_ = vk::SwapchainKHR{raw.swapchain};
    result.format_ = static_cast<vk::Format>(raw.image_format);
    result.extent_ = vk::Extent2D{raw.extent.width, raw.extent.height};

    auto images = raw.get_images();
    if (!images)
        throw std::runtime_error("Swapchain::create: failed to retrieve swapchain images");
    result.images_.reserve(images.value().size());
    for (VkImage image : images.value())
        result.images_.emplace_back(image);

    auto image_views = raw.get_image_views();
    if (!image_views)
        throw std::runtime_error("Swapchain::create: failed to create swapchain image views");
    result.image_views_.reserve(image_views.value().size());
    for (VkImageView image_view : image_views.value())
        result.image_views_.emplace_back(image_view);

    if (result.format_ != info.format) {
        throw std::runtime_error("Swapchain::create: requested surface format is not supported");
    }
    return result;
}

Swapchain::Acquired Swapchain::acquire_next_image(vk::Semaphore signal_semaphore, uint64_t timeout) const {
    uint32_t image_index = 0;
    const VkResult result = vkAcquireNextImageKHR(
        static_cast<VkDevice>(device_), static_cast<VkSwapchainKHR>(swapchain_), timeout,
        static_cast<VkSemaphore>(signal_semaphore), VK_NULL_HANDLE, &image_index);
    if (result == VK_ERROR_OUT_OF_DATE_KHR)
        return {.out_of_date = true};
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
        throw std::runtime_error("Swapchain::acquire_next_image: vkAcquireNextImageKHR failed");
    return {.image_index = image_index, .suboptimal = result == VK_SUBOPTIMAL_KHR};
}

bool Swapchain::present(vk::Queue queue, uint32_t image_index, vk::Semaphore wait_semaphore) const {
    const VkSemaphore wait = static_cast<VkSemaphore>(wait_semaphore);
    const VkSwapchainKHR swapchain = static_cast<VkSwapchainKHR>(swapchain_);
    const VkPresentInfoKHR info{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &wait,
        .swapchainCount = 1,
        .pSwapchains = &swapchain,
        .pImageIndices = &image_index,
    };
    const VkResult result = vkQueuePresentKHR(static_cast<VkQueue>(queue), &info);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
        return false;
    if (result != VK_SUCCESS)
        throw std::runtime_error("Swapchain::present: vkQueuePresentKHR failed");
    return true;
}

void Swapchain::destroy() {
    for (vk::ImageView view : image_views_)
        device_.destroyImageView(view);
    image_views_.clear();
    images_.clear();
    if (swapchain_)
        device_.destroySwapchainKHR(swapchain_);
    device_ = nullptr;
    surface_ = nullptr;
    swapchain_ = nullptr;
    format_ = vk::Format::eUndefined;
    extent_ = vk::Extent2D{};
}

} // namespace vulcao
