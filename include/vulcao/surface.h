#pragma once

#include <vulkan/vulkan.hpp>

namespace vulcao {

/// @brief RAII owner of a VkSurfaceKHR created outside vulcao.
///
/// vulcao cannot create a presentation surface: that is platform specific
/// (SDL, Win32, Xlib, wayland) and belongs to whichever windowing layer the
/// application already uses. What it can do is own one once that layer hands it
/// over, so the destruction point is explicit instead of being a bool argument
/// threaded through every consumer.
///
/// A Context borrows the surface for device selection and a WindowPresenter
/// borrows it for its swapchain. The Surface itself must outlive both.
class Surface {
public:
    /// @brief Creates an empty wrapper that owns nothing.
    Surface() = default;

    /// @brief Destroys the adopted surface.
    ~Surface();

    /// @brief Not copyable.
    Surface(const Surface&) = delete;
    Surface& operator=(const Surface&) = delete;

    /// @brief Moves the surface, leaving the source empty.
    Surface(Surface&& other) noexcept;
    Surface& operator=(Surface&& other) noexcept;

    /// @brief Adopts a surface and takes responsibility for destroying it.
    /// @param instance Instance the surface was created from. It is what
    ///        destruction is issued through and must outlive the wrapper.
    /// @param surface Surface to adopt. A null handle yields an empty wrapper
    ///        that owns nothing, which lets a platform layer report "no surface"
    ///        without a second code path.
    /// @return The owning wrapper.
    /// @throws std::runtime_error if @p instance is null, which would create a
    ///         handle nothing can ever free.
    static Surface adopt(vk::Instance instance, vk::SurfaceKHR surface);

    /// @brief Returns true if a surface is held.
    bool valid() const { return static_cast<bool>(surface_); }

    /// @brief Returns true if a surface is held.
    explicit operator bool() const { return valid(); }

    /// @brief Returns the raw surface handle, for device selection and creation.
    vk::SurfaceKHR handle() const { return surface_; }

    /// @brief Destroys the held surface and resets the wrapper. No-op when empty.
    void destroy();

private:
    vk::Instance instance_;
    vk::SurfaceKHR surface_;
};

} // namespace vulcao
