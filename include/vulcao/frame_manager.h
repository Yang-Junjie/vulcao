#pragma once

#include <cstdint>
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
/// next call to begin_frame() for the same slot or until the manager is destroyed.
struct Frame {
    uint32_t slot = 0;                              ///< Frame in flight slot.
    uint32_t image_index = 0;                       ///< Acquired swapchain image index.
    CommandBuffer* command_buffer = nullptr;        ///< Command buffer to record into.
    vk::Fence in_flight_fence;                      ///< Fence signaled when the submission completes.
    vk::Semaphore image_available;                  ///< Semaphore the submission waits on.
    vk::Semaphore render_finished;                  ///< Semaphore the submission signals.
    bool active = false;                            ///< True between begin_frame and end_frame.
};

/// @brief Creation parameters of a FrameManager.
struct FrameManagerInfo {
    uint32_t frames_in_flight = 2; ///< Number of frames that may be in flight.
};

/// @brief Drives the per-frame synchronization of a swapchain rendering loop.
///
/// Owns the command pools, in-flight fences and semaphores used to record and
/// submit one frame at a time.
/// @warning Single threaded. It must be destroyed before the Context it was
///          created from.
class FrameManager {
public:
    /// @brief Creates a FrameManager for an initialized context.
    /// @param context Context that owns the device and swapchain.
    /// @param info Frame manager creation parameters.
    /// @throws std::runtime_error if the context is not initialized, has no swapchain, or
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

    /// @brief Pipeline stage the image_available semaphore is waited at in end_frame.
    ///
    /// The first barrier that accesses an acquired swapchain image must include
    /// this stage in its source stage mask, otherwise the layout transition is
    /// not ordered after the acquire.
    static constexpr vk::PipelineStageFlags2 acquire_wait_stage =
        vk::PipelineStageFlagBits2::eColorAttachmentOutput;

    /// @brief Waits for the slot's fence, acquires the next image and starts recording.
    ///
    /// The caller should call recreate_swapchain() on vk::OutOfDateKHRError.
    /// @return The acquired frame.
    /// @throws vk::OutOfDateKHRError if the swapchain is out of date.
    /// @throws std::runtime_error if acquiring the image fails for another reason.
    Frame begin_frame();

    /// @brief Ends recording and submits the frame on the graphics queue.
    /// @param frame Frame returned by begin_frame.
    void end_frame(Frame& frame);

    /// @brief Defers destruction of a resource until the GPU is done with it.
    ///
    /// The resource is moved into the deletion queue of the frame's slot and
    /// destroyed at the slot's next begin_frame(), after the fence wait: by
    /// then every submission recorded so far has completed, so it is safe to
    /// release a resource that earlier frames still referenced. Typical use is
    /// replacing a per-frame buffer or image while it may still be read by an
    /// in-flight frame.
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

    /// @brief Presents the frame.
    /// @param frame Frame returned by begin_frame.
    /// @return True if the image was presented, false if the swapchain should be recreated.
    bool present(Frame& frame);

    /// @brief Waits for the device to become idle.
    void wait_idle();

    /// @brief Recreates the swapchain and the per-image synchronization objects.
    /// @param extent New swapchain extent.
    /// @return False if the requested or current surface extent is zero; retry after restoration.
    bool recreate_swapchain(vk::Extent2D extent);

    /// @brief Returns true if the manager holds valid objects.
    bool valid() const { return static_cast<bool>(pool_); }

    /// @brief Returns the number of frames in flight.
    uint32_t frames_in_flight() const { return static_cast<uint32_t>(slots_.size()); }

    /// @brief Returns the slot of the most recently acquired frame.
    uint32_t current_slot() const { return current_slot_; }

private:
    struct Slot {
        CommandBuffer command_buffer;
        Fence in_flight_fence;
        Semaphore image_available;
        /// @brief Resources retired while this slot was recording, destroyed
        ///        after the slot's fence signals.
        DeletionQueue deletion_queue;
    };

    /// @brief Creates the per-frame command buffers, fences and semaphores.
    /// @param count Number of frames in flight.
    void create_slots(uint32_t count);

    /// @brief Recreates the per-swapchain-image render finished semaphores.
    void create_render_finished_semaphores();

    /// @brief Rebuilds the per-image semaphores if the swapchain image count changed.
    void ensure_image_count();

    Context& context_;
    vk::Device device_;
    CommandPool pool_;
    std::vector<Slot> slots_;
    std::vector<Semaphore> render_finished_;
    uint32_t next_slot_ = 0;
    uint32_t current_slot_ = 0;
};

}
