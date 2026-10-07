#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <vulkan/vulkan.hpp>

#include "vulcao/semaphore.h"
#include "vulcao/surface.h"
#include "vulcao/swapchain.h"

namespace vulcao {

class Context;
struct Frame;

/// @brief Creation parameters of a WindowPresenter.
struct WindowPresenterInfo {
    vk::Extent2D extent{};
    /// @brief Requested sRGB surface format.
    ///
    /// The presenter falls back to any available sRGB nonlinear format when the
    /// surface does not offer this one, and reports the chosen format through
    /// format(). Dynamic rendering bakes that format into its pipelines, so a
    /// caller must build them from the value it reads back, not from this field.
    vk::Format format = vk::Format::eB8G8R8A8Srgb;
    vk::PresentModeKHR present_mode = vk::PresentModeKHR::eFifo;
    /// @brief Extra usage flags the swapchain images are created with.
    vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eColorAttachment;
    /// @brief Minimum swapchain image count.
    uint32_t min_image_count = 2;
};

/// @brief Owns the presentation resources of one window and drives acquire/present.
///
/// Holds the swapchain and the per-image present synchronization for a borrowed
/// Surface, so the main window and every secondary window share one
/// acquire/resize/present implementation instead of each re-deriving it from the
/// device and surface. It does not own the surface: the caller keeps it in a
/// vulcao::Surface, which must outlive the presenter.
///
/// It never records commands and owns no command buffer: the caller records into
/// whatever frame it is building and feeds the submission semaphores back through
/// acquire_wait() and render_finished().
///
/// The presenter is not tied to a frame slot. acquire(const Frame&) takes the
/// frame's per-slot semaphore, which is what the main window needs because its
/// wait is only known complete once the whole frame's slot fence has signaled.
/// acquire() uses the presenter's own semaphore for a standalone window renderer
/// that fences its own submissions and waits that fence before acquiring again.
///
/// Present semaphores are keyed by swapchain image, so a present still waiting on
/// one image never blocks a submission writing another.
///
/// @warning Single threaded. It must be destroyed before the Context it was
///          created from.
class WindowPresenter {
public:
    /// @brief Creates the swapchain and the present synchronization for a surface.
    /// @param context Context that owns the device and the queues.
    /// @param surface Presentation surface, borrowed. It must stay valid for the
    ///        presenter's lifetime and must outlive it; a non-const reference is
    ///        taken on purpose, so a temporary cannot be passed.
    /// @param info Creation parameters.
    /// @throws std::runtime_error when the surface is null, no engine queue family
    ///         can present to it, it offers no sRGB nonlinear format, or swapchain
    ///         creation fails.
    WindowPresenter(Context& context, Surface& surface, const WindowPresenterInfo& info);

    /// @brief Destroys the swapchain and its semaphores. The surface is not owned.
    ~WindowPresenter();

    /// @brief Not copyable.
    WindowPresenter(const WindowPresenter&) = delete;
    WindowPresenter& operator=(const WindowPresenter&) = delete;

    /// @brief Not movable.
    WindowPresenter(WindowPresenter&&) = delete;
    WindowPresenter& operator=(WindowPresenter&&) = delete;

    /// @brief Returns true while a valid swapchain is held.
    bool valid() const { return swapchain_.valid(); }

    /// @brief Returns the borrowed presentation surface handle.
    vk::SurfaceKHR surface() const { return surface_->handle(); }

    /// @brief Returns the current swapchain extent.
    vk::Extent2D extent() const { return swapchain_.extent(); }

    /// @brief Returns the swapchain format the surface actually gave.
    vk::Format format() const { return swapchain_.format(); }

    /// @brief Returns the number of swapchain images.
    uint32_t image_count() const { return swapchain_.image_count(); }

    /// @brief Returns the swapchain, for the caller's image and view lookup.
    const Swapchain& swapchain() const { return swapchain_; }

    /// @brief Result of a successful acquire.
    struct Acquired {
        uint32_t image_index = 0; ///< Index into swapchain().images().
        /// The swapchain is still usable, but should be recreated at the next
        /// safe point; the caller decides when.
        bool suboptimal = false;
    };

    /// @brief Acquires the next image, signaling frame.image_available.
    ///
    /// Returns nullopt when the surface currently reports a zero extent or the
    /// swapchain is out of date. In that case no image may be recorded into, and
    /// the caller should keep any offscreen work running instead of idling.
    /// @param frame Frame whose per-slot image_available semaphore is signaled.
    std::optional<Acquired> acquire(const Frame& frame);

    /// @brief Acquires the next image using the presenter's own semaphore.
    ///
    /// For a standalone window renderer whose submission is fenced by its own
    /// fence and which waits that fence before acquiring again, so the semaphore
    /// is never reused while a previous wait is pending. A renderer that shares
    /// whole-frame completion must use acquire(const Frame&) instead.
    std::optional<Acquired> acquire();

    /// @brief Builds the wait info for a submission that consumes the acquired image.
    /// @param frame Frame passed to acquire(const Frame&).
    vk::SemaphoreSubmitInfo acquire_wait(const Frame& frame) const;

    /// @brief Builds the wait info matching acquire().
    vk::SemaphoreSubmitInfo acquire_wait() const;

    /// @brief Builds the signal info a submission writing @p image_index signals.
    /// @param image_index Image index returned by an acquire call.
    vk::SemaphoreSubmitInfo render_finished(uint32_t image_index) const;

    /// @brief Presents an image whose render-finished semaphore has been signaled.
    /// @param queue Queue to present on, normally present_queue().
    /// @param image_index Acquired image to present.
    /// @return False when the swapchain is out of date and must be recreated.
    bool present(vk::Queue queue, uint32_t image_index);

    /// @brief Recreates the swapchain with a new extent.
    ///
    /// Waits for the device to become idle, because the old swapchain's images and
    /// the semaphores tied to them are replaced. Call it only from an explicit
    /// swapchain-lifecycle path, never during steady-state rendering.
    /// @param extent Requested extent.
    /// @return False when the requested or current surface extent is zero, which
    ///         means the window is minimized or collapsed; retry after it is
    ///         restored and do not treat the swapchain as lost.
    /// @throws std::runtime_error when the surface no longer offers an sRGB format
    ///         or swapchain creation fails.
    bool recreate(vk::Extent2D extent);

    /// @brief Returns the queue family this presenter presents on.
    uint32_t present_queue_family() const { return present_queue_family_; }

    /// @brief Returns the queue this presenter presents on.
    vk::Queue present_queue() const { return present_queue_; }

    /// @brief Destroys the swapchain and its semaphores, leaving the surface alone.
    void destroy();

private:
    /// @brief Creates the swapchain and the presentation semaphores.
    void initialize();

    /// @brief Creates one present semaphore per swapchain image.
    void create_render_finished();

    Context* context_ = nullptr;
    /// @brief Borrowed surface; the owner is the caller's vulcao::Surface.
    Surface* surface_ = nullptr;
    WindowPresenterInfo info_;
    uint32_t present_queue_family_ = 0;
    vk::Queue present_queue_;
    Swapchain swapchain_;
    std::vector<Semaphore> render_finished_;
    /// @brief Acquire semaphore used by the acquire() overload.
    Semaphore acquire_semaphore_;
};

} // namespace vulcao
