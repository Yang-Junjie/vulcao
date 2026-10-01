#pragma once

#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.h>

#include "vulcao/allocator.h"

namespace vulcao {

/// @brief Returns the image aspect mask that matches a format.
/// @param format Image format.
/// @return Color, depth or depth-stencil aspect flags.
vk::ImageAspectFlags image_aspect_for_format(vk::Format format);

/// @brief Returns the number of bytes a tightly packed image of that extent occupies.
///
/// Follows the texel block layout, so compressed formats round the extent up to
/// whole blocks. Describes one mip level of one array layer, which is what
/// Context::upload and Context::download move.
/// @param extent Image extent. Depth is ignored for non-3D images, which use 1.
/// @param format Image format.
/// @return Size in bytes.
vk::DeviceSize image_byte_size(vk::Extent3D extent, vk::Format format);

/// @brief RAII wrapper around a VMA-allocated image, its view and tracked layout.
class Image {
public:
    /// @brief Creates an empty image.
    Image() = default;

    /// @brief Destroys the view, the image and its memory.
    ~Image();

    /// @brief Not copyable.
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    /// @brief Moves the image, leaving the source empty.
    Image(Image&& other) noexcept;

    /// @brief Move assignment. Destroys the current image first.
    Image& operator=(Image&& other) noexcept;

    /// @brief Creates an image and allocates its memory.
    ///
    /// No view is created: build one with ImageView::create once the image is
    /// known to be valid. The full subresource range is tracked from
    /// @p image_info for the layout barriers.
    /// @param allocator Allocator used for the memory.
    /// @param image_info Image creation parameters.
    /// @return The created image.
    /// @throws std::runtime_error if the allocator is invalid or the image cannot be created.
    static Image create(Allocator& allocator, const vk::ImageCreateInfo& image_info);

    /// @brief Creates a 2D image.
    /// @param allocator Allocator used for the memory.
    /// @param extent Image width and height.
    /// @param format Image format.
    /// @param usage Image usage flags.
    /// @param mip_levels Number of mip levels.
    /// @param samples Sample count.
    /// @param concurrent_families Queue families that share the image; two or
    ///        more make it concurrent (see Context::transfer_sharing_families
    ///        for asynchronous transfer uploads), empty keeps it exclusive.
    /// @return The created image.
    /// @throws std::runtime_error if the allocator is invalid or the image cannot be created.
    static Image create_2d(Allocator& allocator,
                           vk::Extent2D extent,
                           vk::Format format,
                           vk::ImageUsageFlags usage,
                           uint32_t mip_levels = 1,
                           vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1,
                           vk::ArrayProxy<const uint32_t> concurrent_families = {});

    /// @brief Creates a 1D image.
    static Image create_1d(Allocator& allocator,
                           uint32_t width,
                           vk::Format format,
                           vk::ImageUsageFlags usage,
                           uint32_t mip_levels = 1,
                           uint32_t array_layers = 1,
                           vk::ArrayProxy<const uint32_t> concurrent_families = {});

    /// @brief Creates a 3D image.
    static Image create_3d(Allocator& allocator,
                           vk::Extent3D extent,
                           vk::Format format,
                           vk::ImageUsageFlags usage,
                           uint32_t mip_levels = 1,
                           vk::ArrayProxy<const uint32_t> concurrent_families = {});

    /// @brief Creates a 2D array image.
    static Image create_2d_array(Allocator& allocator,
                                 vk::Extent2D extent,
                                 uint32_t array_layers,
                                 vk::Format format,
                                 vk::ImageUsageFlags usage,
                                 uint32_t mip_levels = 1,
                                 vk::ArrayProxy<const uint32_t> concurrent_families = {});

    /// @brief Creates a cubemap image, six layers and cube compatible.
    static Image create_cube(Allocator& allocator,
                             vk::Extent2D extent,
                             vk::Format format,
                             vk::ImageUsageFlags usage,
                             uint32_t mip_levels = 1,
                             vk::ArrayProxy<const uint32_t> concurrent_families = {});

    /// @brief Creates a depth image.
    /// @param allocator Allocator used for the memory.
    /// @param extent Image width and height.
    /// @param format Depth format.
    /// @return The created image.
    /// @throws std::runtime_error if the allocator is invalid or the image cannot be created.
    static Image create_depth(Allocator& allocator,
                              vk::Extent2D extent,
                              vk::Format format = vk::Format::eD32Sfloat);

    /// @brief Returns true if the image holds a valid handle.
    bool valid() const { return image_ != VK_NULL_HANDLE; }

    /// @brief Returns true if the image holds a valid handle.
    explicit operator bool() const { return valid(); }

    /// @brief Returns the raw Vulkan image handle.
    vk::Image handle() const { return vk::Image{image_}; }

    /// @brief Returns the image extent.
    vk::Extent3D extent() const { return extent_; }

    /// @brief Returns the image type.
    vk::ImageType image_type() const { return image_type_; }

    /// @brief Returns the number of array layers.
    uint32_t array_layers() const { return array_layers_; }

    /// @brief Returns true if the image was created cube compatible.
    bool cube_compatible() const { return cube_compatible_; }

    /// @brief Returns the number of mip levels.
    uint32_t mip_levels() const { return range_.levelCount; }

    /// @brief Returns the image format.
    vk::Format format() const { return format_; }

    /// @brief Returns the usage flags the image was created with.
    vk::ImageUsageFlags usage() const { return usage_; }

    /// @brief Returns the sharing mode the image was created with.
    vk::SharingMode sharing_mode() const { return sharing_mode_; }

    /// @brief Returns the tracked current layout of the image.
    ///
    /// This is CPU-side bookkeeping, not something read back from the driver: it
    /// starts at the image's initialLayout and is updated by the CommandBuffer
    /// helpers that take an Image& (transition, generate_mipmaps). Anything that
    /// transitions the image through raw handles leaves it stale, and a stale
    /// value makes later barriers record the wrong oldLayout. Keep transitions
    /// on the Image& overloads so this stays accurate.
    vk::ImageLayout layout() const { return layout_; }

    /// @brief Returns the full subresource range of the image.
    vk::ImageSubresourceRange subresource_range() const { return range_; }

    /// @brief Returns the VMA allocation of the image.
    VmaAllocation allocation() const { return allocation_; }

private:
    friend class CommandBuffer;

    /// @brief Updates the tracked layout. Called by CommandBuffer.
    void set_layout(vk::ImageLayout layout) { layout_ = layout; }

    /// @brief Destroys the image and its memory, then resets the wrapper.
    void destroy();

    vk::Device device_;
    VmaAllocator allocator_ = nullptr;
    VkImage image_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = nullptr;
    vk::Extent3D extent_{};
    vk::ImageType image_type_ = vk::ImageType::e2D;
    uint32_t array_layers_ = 1;
    bool cube_compatible_ = false;
    vk::Format format_ = vk::Format::eUndefined;
    vk::ImageUsageFlags usage_;
    vk::SharingMode sharing_mode_ = vk::SharingMode::eExclusive;
    vk::ImageLayout layout_ = vk::ImageLayout::eUndefined;
    vk::ImageSubresourceRange range_{};
};

}
