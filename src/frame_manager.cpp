#include "vulcao/frame_manager.h"

#include "vulcao/context.h"

#include <stdexcept>
#include <utility>

namespace vulcao {

FrameManager::FrameManager(Context& context, const FrameManagerInfo& info)
    : context_(context), device_(context.device()) {
    if (!context.initialized())
        throw std::runtime_error("FrameManager: the context must be initialized");
    if (!context.swapchain())
        throw std::runtime_error("FrameManager: the context must have a swapchain");
    if (info.frames_in_flight == 0)
        throw std::runtime_error("FrameManager: frames_in_flight must be at least 1");

    pool_ = CommandPool::create(device_, context_.graphics_queue_family_index(),
                                vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    create_slots(info.frames_in_flight);
    create_render_finished_semaphores();
}

FrameManager::~FrameManager() {
    if (!valid())
        return;

    device_.waitIdle();
    render_finished_.clear();
    slots_.clear();
    pool_.destroy();
}

void FrameManager::create_slots(uint32_t count) {
    slots_.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        Slot slot;
        slot.command_buffer = pool_.allocate();
        slot.in_flight_fence = Fence::create(device_, vk::FenceCreateFlagBits::eSignaled);
        slot.image_available = Semaphore::create(device_);
        slots_.push_back(std::move(slot));
    }
}

void FrameManager::create_render_finished_semaphores() {
    render_finished_.clear();

    const size_t count = context_.swapchain_images().size();
    render_finished_.reserve(count);
    for (size_t i = 0; i < count; ++i)
        render_finished_.push_back(Semaphore::create(device_));
}

void FrameManager::ensure_image_count() {
    if (render_finished_.size() == context_.swapchain_images().size())
        return;

    context_.wait_idle();
    create_render_finished_semaphores();
}

Frame FrameManager::begin_frame() {
    ensure_image_count();

    Slot& slot = slots_[next_slot_];
    slot.in_flight_fence.wait();

    // The slot's previous submission just completed, so resources retired
    // while it (or any earlier frame) was recording are no longer referenced.
    slot.deletion_queue.flush();

    uint32_t image_index = 0;
    const vk::ResultValue<uint32_t> acquired = device_.acquireNextImageKHR(
        context_.swapchain(), UINT64_MAX, slot.image_available.handle());
    if (acquired.result == vk::Result::eErrorOutOfDateKHR)
        throw vk::OutOfDateKHRError("acquireNextImageKHR");
    if (acquired.result != vk::Result::eSuccess &&
        acquired.result != vk::Result::eSuboptimalKHR)
        throw std::runtime_error("FrameManager::begin_frame failed: " +
                                 vk::to_string(acquired.result));
    image_index = acquired.value;

    // No eOneTimeSubmit: this buffer is re-recorded every frame.
    slot.command_buffer.reset();
    slot.command_buffer.begin(vk::CommandBufferUsageFlags{});

    Frame frame;
    frame.slot = next_slot_;
    frame.image_index = image_index;
    frame.command_buffer = &slot.command_buffer;
    frame.in_flight_fence = slot.in_flight_fence.handle();
    frame.image_available = slot.image_available.handle();
    frame.render_finished = render_finished_[image_index].handle();
    frame.active = true;

    current_slot_ = next_slot_;
    next_slot_ = (next_slot_ + 1) % static_cast<uint32_t>(slots_.size());
    return frame;
}

void FrameManager::end_frame(Frame& frame) {
    Slot& slot = slots_[frame.slot];
    if (frame.command_buffer)
        frame.command_buffer->end();

    slot.in_flight_fence.reset();

    const vk::CommandBufferSubmitInfo command_info{
        .commandBuffer = slot.command_buffer.handle(),
    };
    const vk::SemaphoreSubmitInfo wait_info{
        .semaphore = frame.image_available,
        .stageMask = acquire_wait_stage,
    };
    const vk::SemaphoreSubmitInfo signal_info{
        .semaphore = frame.render_finished,
        .stageMask = vk::PipelineStageFlagBits2::eAllCommands,
    };

    context_.submit(context_.graphics_queue(),
                    vk::SubmitInfo2{
                        .waitSemaphoreInfoCount = 1,
                        .pWaitSemaphoreInfos = &wait_info,
                        .commandBufferInfoCount = 1,
                        .pCommandBufferInfos = &command_info,
                        .signalSemaphoreInfoCount = 1,
                        .pSignalSemaphoreInfos = &signal_info,
                    },
                    slot.in_flight_fence.handle());

    frame.active = false;
}

bool FrameManager::present(Frame& frame) {
    const vk::Semaphore wait_semaphore = frame.render_finished;
    const vk::SwapchainKHR swapchain = context_.swapchain();
    const uint32_t image_index = frame.image_index;

    const vk::Result result = context_.present_queue().presentKHR(vk::PresentInfoKHR{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &wait_semaphore,
        .swapchainCount = 1,
        .pSwapchains = &swapchain,
        .pImageIndices = &image_index,
    });
    return result == vk::Result::eSuccess;
}

void FrameManager::wait_idle() {
    context_.wait_idle();
}

void FrameManager::recreate_swapchain(vk::Extent2D extent) {
    context_.recreate_swapchain(extent);
    create_render_finished_semaphores();
}

}
