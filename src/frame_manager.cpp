#include "vulcao/frame_manager.h"

#include "vulcao/context.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace vulcao {

FrameManager::FrameManager(Context& context, const FrameManagerInfo& info)
    : context_(context), device_(context.device()) {
    if (!context.initialized())
        throw std::runtime_error("FrameManager: the context must be initialized");
    if (info.frames_in_flight == 0)
        throw std::runtime_error("FrameManager: frames_in_flight must be at least 1");

    pool_ = CommandPool::create(device_, context_.graphics_queue_family_index(),
                                vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    create_slots(info.frames_in_flight);
}

FrameManager::~FrameManager() {
    if (!valid())
        return;

    device_.waitIdle();
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

Frame FrameManager::begin_frame() {
    Slot& slot = slots_[next_slot_];
    // Every submission registered against this slot in the previous frame has
    // completed, so its images, buffers and descriptors can be rewritten and
    // anything retired while it was recording can be destroyed.
    slot.in_flight_fence.wait();
    slot.deletion_queue.flush();

    // No eOneTimeSubmit: this buffer is re-recorded every frame.
    slot.command_buffer.reset();
    slot.command_buffer.begin(vk::CommandBufferUsageFlags{});

    Frame frame;
    frame.slot = next_slot_;
    frame.serial = next_serial_++;
    frame.command_buffer = &slot.command_buffer;
    frame.completion_fence = slot.in_flight_fence.handle();
    frame.image_available = slot.image_available.handle();
    frame.active = true;

    current_slot_ = next_slot_;
    current_serial_ = frame.serial;
    next_slot_ = (next_slot_ + 1) % static_cast<uint32_t>(slots_.size());
    return frame;
}

void FrameManager::submit(Frame& frame,
                          std::span<const vk::SemaphoreSubmitInfo> wait_semaphores,
                          std::span<const vk::SemaphoreSubmitInfo> signal_semaphores) {
    if (frame.command_buffer && frame.active)
        frame.command_buffer->end();

    const vk::CommandBufferSubmitInfo command_info{.commandBuffer = frame.command_buffer->handle()};
    context_.submit(context_.graphics_queue(),
                    vk::SubmitInfo2{
                        .waitSemaphoreInfoCount = static_cast<uint32_t>(wait_semaphores.size()),
                        .pWaitSemaphoreInfos = wait_semaphores.data(),
                        .commandBufferInfoCount = 1,
                        .pCommandBufferInfos = &command_info,
                        .signalSemaphoreInfoCount = static_cast<uint32_t>(signal_semaphores.size()),
                        .pSignalSemaphoreInfos = signal_semaphores.data(),
                    });
}

void FrameManager::submit(const vk::SubmitInfo2& info) {
    context_.submit(context_.graphics_queue(), info);
}

void FrameManager::end_frame(Frame& frame) {
    Slot& slot = slots_[frame.slot];

    // The tail batch carries no commands: it only signals the slot fence once
    // every graphics submission made before it has completed. Same-queue
    // submissions execute in order, so the fence covers the main frame, all
    // registered offscreen producers and the secondary window renderers.
    slot.in_flight_fence.reset();
    context_.submit(context_.graphics_queue(), vk::SubmitInfo2{}, slot.in_flight_fence.handle());

    // Recorded only here: from now on this slot describes the frame that was just
    // submitted, and completed_serial() can tell whether its fence has signaled.
    slot.last_serial = frame.serial;
    slot.used = true;
    frame.active = false;
}

uint64_t FrameManager::completed_serial() const {
    uint64_t earliest_incomplete = UINT64_MAX;
    uint64_t highest_submitted = 0;
    for (const Slot& slot : slots_) {
        if (!slot.used)
            continue;
        highest_submitted = std::max(highest_submitted, slot.last_serial);
        if (!slot.in_flight_fence.signaled())
            earliest_incomplete = std::min(earliest_incomplete, slot.last_serial);
    }

    // Every slot's fence has signaled, so every frame, being some slot's most
    // recent one, has completed.
    if (earliest_incomplete == UINT64_MAX)
        return highest_submitted;
    return earliest_incomplete - 1;
}

void FrameManager::wait_idle() {
    context_.wait_idle();
}

}
