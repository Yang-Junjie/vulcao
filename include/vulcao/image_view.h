#pragma once

#include <vulkan/vulkan.hpp>

namespace vulcao {

class Image;

/// @brief RAII wrapper around a Vulkan image view.
///
/// An Image owns the memory and the tracked layout, a view owns the
/// interpretation. Create as many views per image as needed: different formats,
/// mip or layer ranges and view types (1D, 2D array, cube, 3D).
class ImageView {
public:
    /// @brief Creates an empty view.
    ImageView() = default;

    /// @brief Destroys the view.
    ~ImageView();

    /// @brief Not copyable.
    ImageView(const ImageView&) = delete;
    ImageView& operator=(const ImageView&) = delete;

    /// @brief Moves the view, leaving the source empty.
    ImageView(ImageView&& other) noexcept;

    /// @brief Move assignment. Destroys the current view first.
    ImageView& operator=(ImageView&& other) noexcept;

    /// @brief Creates a view from a full create info.
    /// @param device Device that creates the view.
    /// @param info View creation parameters, including the image.
    /// @return The created view.
    /// @throws std::runtime_error if the image is invalid or creation fails.
    static ImageView create(vk::Device device, const vk::ImageViewCreateInfo& info);

    /// @brief Creates a view deriving the type, format and range from the image.
    /// @param device Device that creates the view.
    /// @param image Image to view.
    /// @return The created view.
    /// @throws std::runtime_error if the image is invalid or creation fails.
    static ImageView create(vk::Device device, const Image& image);

    /// @brief Creates a view with an explicit type, format and range.
    ///
    /// A zero levelCount or layerCount in @p range is replaced with the whole
    /// image, an undefined @p format with the image format, and an empty aspect
    /// mask with the aspect of @p format.
    /// @param device Device that creates the view.
    /// @param image Image to view.
    /// @param type View type.
    /// @param format View format, or undefined to use the image format.
    /// @param range Subresource range, or a default range for the whole image.
    /// @return The created view.
    /// @throws std::runtime_error if the image is invalid or creation fails.
    static ImageView create(vk::Device device,
                            const Image& image,
                            vk::ImageViewType type,
                            vk::Format format = vk::Format::eUndefined,
                            vk::ImageSubresourceRange range = {});

    /// @brief Returns true if the view holds a valid handle.
    bool valid() const { return static_cast<bool>(view_); }

    /// @brief Returns true if the view holds a valid handle.
    explicit operator bool() const { return valid(); }

    /// @brief Returns the raw Vulkan image view handle.
    vk::ImageView handle() const { return view_; }

    /// @brief Returns the format the view was created with.
    vk::Format format() const { return format_; }

    /// @brief Returns the subresource range the view covers.
    vk::ImageSubresourceRange range() const { return range_; }

    /// @brief Destroys the view and resets the wrapper.
    void destroy();

private:
    vk::Device device_;
    vk::ImageView view_;
    vk::Format format_ = vk::Format::eUndefined;
    vk::ImageSubresourceRange range_{};
};

}
