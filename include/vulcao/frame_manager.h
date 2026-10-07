#pragma once

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include <vulkan/vulkan.hpp>

#include "vulcao/command_buffer.h"
#include "vulcao/command_pool.h"
#include "vulcao/deletion_queue.h"
#include "vulcao/fence.h"
#include "vulcao/semaphore.h"

namespace vulcao {

class Context;

/// @brief Data of a frame acquired by a FrameManager.
///
/// The pointer members are owned by the FrameManager and stay valid until the
/// next call to begin_frame() for the same slot or until the manager is
/// destroyed.
struct Frame {
    uint32_t slot = 0;                              ///< Frame in flight slot.
    /// @brief Monotonic logical frame counter. Never reused, unlike the slot.
    uint64_t serial = 0;
    CommandBuffer* command_buffer = nullptr;        ///< Command buffer to record into.
    /// @brief Signaled when every graphics submission of this frame has completed.
    ///
    /// begin_frame() waits on it before the slot is reused, so it gates the
    /// retirement of that slot's images, buffers and descriptors.
    vk::Fence completion_fence;
    /// @brief Per-slot acquire semaphore a window presenter signals on acquire.
    ///
    /// Reusable because begin_frame() only returns once the previous wait on it
    /// (made by end_frame()'s submission) has completed.
    vk::Semaphore image_available;
    bool active = false;                            ///< True between begin_frame and end_frame.
};

/// @brief Creation parameters of a FrameManager.
struct FrameManagerInfo {
    uint32_t frames_in_flight = 2; ///< Number of frames that may be in flight.
};

/// @brief Coordinates per-frame CPU/GPU resources independently of any window.
///
/// Owns the command pools, per-slot command buffers, per-slot completion fences
/// and per-slot acquire semaphores. The manager no longer acquires or presents a
/// swapchain: window presenters do that and register their submissions with the
/// manager, whose slot fence gates every graphics submission of the frame.
///
/// Typical order per frame:
/// @code
/// Frame frame = manager.begin_frame();
/// // record into frame.command_buffer, or submit extra command buffers
/// manager.submit(frame, waits, signals);   // ends and submits the main buffer
/// manager.submit(extra_submit_info);       // any number of extra submissions
/// manager.end_frame(frame);                // signals the slot completion fence
/// @endcode
/// @warning Single threaded. It must be destroyed before the Context it was
///          created from.
class FrameManager {
public:
    /// @brief Creates a FrameManager for an initialized context.
    /// @param context Context that owns the device and queues.
    /// @param info Frame manager creation parameters.
    /// @throws std::runtime_error if the context is not initialized or
    ///         frames_in_flight is zero.
    explicit FrameManager(Context& context, const FrameManagerInfo& info = {});

    /// @brief Waits for in-flight work and destroys the owned synchronization objects.
    ~FrameManager();

    /// @brief Not copyable.
    FrameManager(const FrameManager&) = delete;
    FrameManager& operator=(const FrameManager&) = delete;

    /// @brief Not movable.
    FrameManager(FrameManager&&) = delete;
    FrameManager& operator=(FrameManager&&) = delete;

    /// @brief Pipeline stage an acquire semaphore is waited at for image access.
    ///
    /// The first barrier that accesses an acquired swapchain image must include
    /// this stage in its source stage mask, otherwise the layout transition is
    /// not ordered after the acquire.
    static constexpr vk::PipelineStageFlags2 acquire_wait_stage =
        vk::PipelineStageFlagBits2::eColorAttachmentOutput;

    /// @brief Waits for the slot's fence, retires its resources and starts recording.
    ///
    /// Does not acquire any swapchain image; a window presenter does that with
    /// the returned Frame.
    /// @return The recycled frame slot.
    Frame begin_frame();

    /// @brief Ends the frame's command buffer and submits it on the graphics queue.
    ///
    /// The slot is not fenced here: end_frame() signals the completion fence
    /// after every submission of the frame has been recorded. Pass the acquire
    /// and render-finished semaphores of the window presenting this frame, or
    /// nothing for pure offscreen work.
    /// @param frame Frame returned by begin_frame.
    /// @param wait_semaphores Semaphores the submission waits on. A wait on an
    ///        acquire must use acquire_wait_stage in its stage mask.
    /// @param signal_semaphores Semaphores the submission signals.
    void submit(Frame& frame,
                std::span<const vk::SemaphoreSubmitInfo> wait_semaphores = {},
                std::span<const vk::SemaphoreSubmitInfo> signal_semaphores = {});

    /// @brief Registers an extra graphics submission whose completion the slot must cover.
    ///
    /// Use for offscreen producers or secondary window renderers that own their
    /// command buffers; call before end_frame().
    /// @param info Submission parameters.
    void submit(const vk::SubmitInfo2& info);

    /// @brief Signals the slot's completion fence after all submissions of the frame.
    ///
    /// Must be the last call for the frame, once the main submission and every
    /// registered extra submission have been made.
    /// @param frame Frame returned by begin_frame.
    void end_frame(Frame& frame);

    /// @brief Defers destruction of a resource until the GPU is done with it.
    ///
    /// The resource is moved into the deletion queue of the frame's slot and
    /// destroyed at the slot's next begin_frame(), after the fence wait: by
    /// then every submission of this frame, including secondary window work
    /// registered before end_frame(), has completed.
    /// @tparam T Move-only resource type (Buffer, Image, ...).
    /// @param frame Frame the resource is retired from.
    /// @param resource Resource to destroy later.
    template <typename T>
    void defer_destroy(Frame& frame, T&& resource) {
        slots_.at(frame.slot).deletion_queue.push(std::forward<T>(resource));
    }

    /// @brief Defers destruction of a resource to the most recently acquired slot.
    /// @tparam T Move-only resource type (Buffer, Image, ...).
    /// @param resource Resource to destroy later.
    template <typename T>
    void defer_destroy(T&& resource) {
        slots_.at(current_slot_).deletion_queue.push(std::forward<T>(resource));
    }

    /// @brief Waits for the device to become idle.
    void wait_idle();

    /// @brief Returns true if the manager holds valid objects.
    bool valid() const { return static_cast<bool>(pool_); }

    /// @brief Returns the number of frames in flight.
    uint32_t frames_in_flight() const { return static_cast<uint32_t>(slots_.size()); }

    /// @brief Returns the slot of the most recently acquired frame.
    uint32_t current_slot() const { return current_slot_; }

    /// @brief Returns the serial of the most recently acquired frame.
    uint64_t current_serial() const { return current_serial_; }

    /// @brief Returns the highest frame serial whose submissions have all completed.
    ///
    /// A slot's fence is signaled only after its most recent submission, and every
    /// frame belongs to exactly one slot, so once every slot's fence has signaled
    /// every frame has completed. While some slot is still running, the prefix
    /// stops one serial before that slot's frame.
    ///
    /// This is the retirement judgement for resources used by many frames: a
    /// resource whose last use was serial S is safe to destroy or rewrite once
    /// completed_serial() returns S or more. It queries fences rather than
    /// waiting, so it never stalls the CPU on GPU progress.
    uint64_t completed_serial() const;

private:
    struct Slot {
        CommandBuffer command_buffer;
        Fence in_flight_fence;
        Semaphore image_available;
        /// @brief Serial of the last frame this slot submitted its tail fence for.
        ///
        /// Updated in end_frame, not in begin_frame, so that while a frame is
        /// being built the slot still describes the previous completed one.
        uint64_t last_serial = 0;
        bool used = false;
        /// @brief Resources retired while this slot was recording, destroyed
        ///        after the slot's fence signals.
        DeletionQueue deletion_queue;
    };

    /// @brief Creates the per-frame command buffers, fences and semaphores.
    /// @param count Number of frames in flight.
    void create_slots(uint32_t count);

    Context& context_;
    vk::Device device_;
    CommandPool pool_;
    std::vector<Slot> slots_;
    uint32_t next_slot_ = 0;
    uint32_t current_slot_ = 0;
    /// @brief Serials start at 1 so that 0 can mean "no frame has run".
    uint64_t next_serial_ = 1;
    uint64_t current_serial_ = 0;
};

}
