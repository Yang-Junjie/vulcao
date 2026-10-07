#include "vulcao/window_presenter.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <utility>

#include "vulcao/context.h"
#include "vulcao/frame_manager.h"

namespace vulcao {
namespace {

constexpr std::array<vk::Format, 2> kPreferredSrgbFormats{{
    vk::Format::eB8G8R8A8Srgb,
    vk::Format::eR8G8B8A8Srgb,
}};

/// Picks an sRGB nonlinear surface format, preferring the requested one.
std::optional<vk::SurfaceFormatKHR> select_srgb_format(
    const std::vector<vk::SurfaceFormatKHR>& formats, vk::Format requested) {
    const auto matches = [](const vk::SurfaceFormatKHR& format, vk::Format wanted) {
        return format.format == wanted && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
    };

    if (const auto found = std::find_if(formats.begin(), formats.end(),
                                        [&](const auto& candidate) { return matches(candidate, requested); });
        found != formats.end())
        return *found;

    for (vk::Format wanted : kPreferredSrgbFormats) {
        if (const auto found = std::find_if(formats.begin(), formats.end(),
                                            [&](const auto& candidate) { return matches(candidate, wanted); });
            found != formats.end())
            return *found;
    }

    // VK_FORMAT_UNDEFINED with a nonlinear colorspace means "any format"; sRGB
    // is then what the presentation path wants.
    if (formats.size() == 1 && formats.front().format == vk::Format::eUndefined &&
        formats.front().colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
        return vk::SurfaceFormatKHR{.format = vk::Format::eB8G8R8A8Srgb,
                                    .colorSpace = vk::ColorSpaceKHR::eSrgbNonlinear};
    }
    return std::nullopt;
}

} // namespace

WindowPresenter::WindowPresenter(Context& context,
                                 Surface& surface,
                                 const WindowPresenterInfo& info)
        : context_(&context), surface_(&surface), info_(info) {
    // initialize() can fail after the swapchain exists, so everything it created
    // is released before the exception leaves. The surface is not ours to free.
    try {
        initialize();
    } catch (...) {
        destroy();
        throw;
    }
}

void WindowPresenter::initialize() {
    if (surface_ == nullptr || !surface_->valid())
        throw std::runtime_error("WindowPresenter: a valid surface is required");

    const vk::SurfaceKHR surface = surface_->handle();
    const uint32_t graphics_family = context_->graphics_queue_family_index();
    const uint32_t present_family = context_->present_queue_family_index();
    if (context_->physical_device().getSurfaceSupportKHR(graphics_family, surface)) {
        present_queue_family_ = graphics_family;
        present_queue_ = context_->graphics_queue();
    } else if (present_family != graphics_family &&
               context_->physical_device().getSurfaceSupportKHR(present_family, surface)) {
        present_queue_family_ = present_family;
        present_queue_ = context_->present_queue();
    } else {
        throw std::runtime_error("WindowPresenter: no engine queue family can present to the surface");
    }

    const std::vector<vk::SurfaceFormatKHR> formats =
        context_->physical_device().getSurfaceFormatsKHR(surface);
    const auto selected = select_srgb_format(formats, info_.format);
    if (!selected)
        throw std::runtime_error("WindowPresenter: the surface has no sRGB nonlinear format");

    swapchain_ = Swapchain::create(SwapchainCreateInfo{
        .physical_device = context_->physical_device(),
        .device = context_->device(),
        .surface = surface,
        .extent = info_.extent,
        .format = selected->format,
        .color_space = selected->colorSpace,
        .present_mode = info_.present_mode,
        .min_image_count = info_.min_image_count,
        .usage = info_.usage,
        .graphics_queue_family_index = graphics_family,
        .present_queue_family_index = present_queue_family_,
        .old_swapchain = {},
    });
    create_render_finished();
    acquire_semaphore_ = Semaphore::create(context_->device());
}

WindowPresenter::~WindowPresenter() { destroy(); }

void WindowPresenter::create_render_finished() {
    render_finished_.clear();
    render_finished_.reserve(swapchain_.image_count());
    for (uint32_t i = 0; i < swapchain_.image_count(); ++i)
        render_finished_.push_back(Semaphore::create(context_->device()));
}

std::optional<WindowPresenter::Acquired> WindowPresenter::acquire(const Frame& frame) {
    if (!swapchain_.valid())
        return std::nullopt;

    // A minimized or collapsed window reports a zero current extent; acquiring
    // would block until it is restored, so the caller gets "no image this frame"
    // and keeps its offscreen work moving instead.
    const vk::SurfaceCapabilitiesKHR capabilities =
        context_->physical_device().getSurfaceCapabilitiesKHR(surface_->handle());
    if (capabilities.currentExtent.width == 0 || capabilities.currentExtent.height == 0)
        return std::nullopt;

    const vk::Extent2D extent = swapchain_.extent();
    if (extent.width == 0 || extent.height == 0)
        return std::nullopt;

    const Swapchain::Acquired acquired = swapchain_.acquire_next_image(frame.image_available);
    if (acquired.out_of_date)
        return std::nullopt;

    return Acquired{.image_index = acquired.image_index, .suboptimal = acquired.suboptimal};
}

std::optional<WindowPresenter::Acquired> WindowPresenter::acquire() {
    if (!swapchain_.valid())
        return std::nullopt;

    const vk::SurfaceCapabilitiesKHR capabilities =
        context_->physical_device().getSurfaceCapabilitiesKHR(surface_->handle());
    if (capabilities.currentExtent.width == 0 || capabilities.currentExtent.height == 0)
        return std::nullopt;

    const Swapchain::Acquired acquired = swapchain_.acquire_next_image(acquire_semaphore_.handle());
    if (acquired.out_of_date)
        return std::nullopt;

    return Acquired{.image_index = acquired.image_index, .suboptimal = acquired.suboptimal};
}

vk::SemaphoreSubmitInfo WindowPresenter::acquire_wait(const Frame& frame) const {
    return vk::SemaphoreSubmitInfo{
        .semaphore = frame.image_available,
        .value = 0,
        .stageMask = FrameManager::acquire_wait_stage,
    };
}

vk::SemaphoreSubmitInfo WindowPresenter::acquire_wait() const {
    return vk::SemaphoreSubmitInfo{
        .semaphore = acquire_semaphore_.handle(),
        .value = 0,
        .stageMask = FrameManager::acquire_wait_stage,
    };
}

vk::SemaphoreSubmitInfo WindowPresenter::render_finished(uint32_t image_index) const {
    return vk::SemaphoreSubmitInfo{
        .semaphore = render_finished_.at(image_index).handle(),
        .value = 0,
        .stageMask = vk::PipelineStageFlagBits2::eAllCommands,
    };
}

bool WindowPresenter::present(vk::Queue queue, uint32_t image_index) {
    return swapchain_.present(queue, image_index, render_finished_.at(image_index).handle());
}

bool WindowPresenter::recreate(vk::Extent2D extent) {
    if (extent.width == 0 || extent.height == 0)
        return false;

    const vk::SurfaceKHR surface = surface_->handle();
    const vk::SurfaceCapabilitiesKHR capabilities =
        context_->physical_device().getSurfaceCapabilitiesKHR(surface);
    if (capabilities.currentExtent.width == 0 || capabilities.currentExtent.height == 0)
        return false;

    context_->wait_idle();

    const std::vector<vk::SurfaceFormatKHR> formats =
        context_->physical_device().getSurfaceFormatsKHR(surface);
    const auto selected = select_srgb_format(formats, info_.format);
    if (!selected)
        throw std::runtime_error("WindowPresenter: the surface has no sRGB nonlinear format");

    Swapchain new_swapchain = Swapchain::create(SwapchainCreateInfo{
        .physical_device = context_->physical_device(),
        .device = context_->device(),
        .surface = surface,
        .extent = extent,
        .format = selected->format,
        .color_space = selected->colorSpace,
        .present_mode = info_.present_mode,
        .min_image_count = info_.min_image_count,
        .usage = info_.usage,
        .graphics_queue_family_index = context_->graphics_queue_family_index(),
        .present_queue_family_index = present_queue_family_,
        .old_swapchain = swapchain_.handle(),
    });

    // The new swapchain replaced the old one, so the present semaphores of the
    // old image set are no longer referenced by anything.
    render_finished_.clear();
    swapchain_ = std::move(new_swapchain);
    create_render_finished();
    return true;
}

void WindowPresenter::destroy() {
    render_finished_.clear();
    swapchain_.destroy();
    // surface_ is borrowed: its owner destroys it.
}

} // namespace vulcao
