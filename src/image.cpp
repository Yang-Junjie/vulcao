#include "vulcao/image.h"

#include "vulcao/check.h"

#include <array>
#include <stdexcept>
#include <utility>

#include <vulkan/vulkan_format_traits.hpp>

namespace vulcao {
namespace {

/// @brief Sharing fields of an image created concurrently across queue families.
struct Sharing {
    vk::SharingMode mode = vk::SharingMode::eExclusive;
    uint32_t count = 0;
    const uint32_t* indices = nullptr;
};

Sharing sharing_for(vk::ArrayProxy<const uint32_t> families) {
    const bool concurrent = families.size() >= 2;
    return Sharing{
        .mode = concurrent ? vk::SharingMode::eConcurrent : vk::SharingMode::eExclusive,
        .count = concurrent ? static_cast<uint32_t>(families.size()) : 0u,
        .indices = concurrent ? families.data() : nullptr,
    };
}

}

vk::ImageAspectFlags image_aspect_for_format(vk::Format format) {
    switch (format) {
        case vk::Format::eD16Unorm:
        case vk::Format::eX8D24UnormPack32:
        case vk::Format::eD32Sfloat:
            return vk::ImageAspectFlagBits::eDepth;
        case vk::Format::eD16UnormS8Uint:
        case vk::Format::eD24UnormS8Uint:
        case vk::Format::eD32SfloatS8Uint:
            return vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
        default:
            return vk::ImageAspectFlagBits::eColor;
    }
}

vk::DeviceSize image_byte_size(vk::Extent3D extent, vk::Format format) {
    const std::array<uint8_t, 3> block_extent = vk::blockExtent(format);
    const vk::DeviceSize block_size = vk::blockSize(format);

    const vk::DeviceSize blocks_x = (extent.width + block_extent[0] - 1) / block_extent[0];
    const vk::DeviceSize blocks_y = (extent.height + block_extent[1] - 1) / block_extent[1];
    const vk::DeviceSize blocks_z = (extent.depth + block_extent[2] - 1) / block_extent[2];
    return blocks_x * blocks_y * blocks_z * block_size;
}

Image::~Image() {
    destroy();
}

Image::Image(Image&& other) noexcept
    : device_(std::exchange(other.device_, vk::Device{})),
      allocator_(std::exchange(other.allocator_, nullptr)),
      image_(std::exchange(other.image_, VK_NULL_HANDLE)),
      allocation_(std::exchange(other.allocation_, nullptr)),
      extent_(std::exchange(other.extent_, vk::Extent3D{})),
      image_type_(std::exchange(other.image_type_, vk::ImageType::e2D)),
      array_layers_(std::exchange(other.array_layers_, 1u)),
      cube_compatible_(std::exchange(other.cube_compatible_, false)),
      format_(std::exchange(other.format_, vk::Format::eUndefined)),
      usage_(std::exchange(other.usage_, vk::ImageUsageFlags{})),
      sharing_mode_(std::exchange(other.sharing_mode_, vk::SharingMode::eExclusive)),
      layout_(std::exchange(other.layout_, vk::ImageLayout::eUndefined)),
      range_(std::exchange(other.range_, vk::ImageSubresourceRange{})) {}

Image& Image::operator=(Image&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = std::exchange(other.device_, vk::Device{});
        allocator_ = std::exchange(other.allocator_, nullptr);
        image_ = std::exchange(other.image_, VK_NULL_HANDLE);
        allocation_ = std::exchange(other.allocation_, nullptr);
        extent_ = std::exchange(other.extent_, vk::Extent3D{});
        image_type_ = std::exchange(other.image_type_, vk::ImageType::e2D);
        array_layers_ = std::exchange(other.array_layers_, 1u);
        cube_compatible_ = std::exchange(other.cube_compatible_, false);
        format_ = std::exchange(other.format_, vk::Format::eUndefined);
        usage_ = std::exchange(other.usage_, vk::ImageUsageFlags{});
        sharing_mode_ = std::exchange(other.sharing_mode_, vk::SharingMode::eExclusive);
        layout_ = std::exchange(other.layout_, vk::ImageLayout::eUndefined);
        range_ = std::exchange(other.range_, vk::ImageSubresourceRange{});
    }
    return *this;
}

Image Image::create(Allocator& allocator, const vk::ImageCreateInfo& image_info) {
    if (!allocator.valid())
        throw std::runtime_error("Image::create: invalid allocator");

    Image image;
    image.allocator_ = allocator.handle();
    image.device_ = allocator.device();

    VmaAllocationCreateInfo allocation_info{};
    allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    const VkImageCreateInfo raw_image_info = image_info;
    check(static_cast<vk::Result>(vmaCreateImage(image.allocator_, &raw_image_info, &allocation_info,
                                                 &image.image_, &image.allocation_, nullptr)),
          "create image");

    image.extent_ = image_info.extent;
    image.image_type_ = image_info.imageType;
    image.array_layers_ = image_info.arrayLayers;
    image.cube_compatible_ =
        static_cast<bool>(image_info.flags & vk::ImageCreateFlagBits::eCubeCompatible);
    image.format_ = image_info.format;
    image.usage_ = image_info.usage;
    image.sharing_mode_ = image_info.sharingMode;
    image.layout_ = image_info.initialLayout;
    image.range_ = vk::ImageSubresourceRange{
        .aspectMask = image_aspect_for_format(image_info.format),
        .baseMipLevel = 0,
        .levelCount = image_info.mipLevels,
        .baseArrayLayer = 0,
        .layerCount = image_info.arrayLayers,
    };
    return image;
}

Image Image::create_2d(Allocator& allocator,
                       vk::Extent2D extent,
                       vk::Format format,
                       vk::ImageUsageFlags usage,
                       uint32_t mip_levels,
                       vk::SampleCountFlagBits samples,
                       vk::ArrayProxy<const uint32_t> concurrent_families) {
    const Sharing sharing = sharing_for(concurrent_families);
    return create(allocator,
                  vk::ImageCreateInfo{
                      .imageType = vk::ImageType::e2D,
                      .format = format,
                      .extent = vk::Extent3D{extent.width, extent.height, 1},
                      .mipLevels = mip_levels,
                      .arrayLayers = 1,
                      .samples = samples,
                      .tiling = vk::ImageTiling::eOptimal,
                      .usage = usage,
                      .sharingMode = sharing.mode,
                      .queueFamilyIndexCount = sharing.count,
                      .pQueueFamilyIndices = sharing.indices,
                      .initialLayout = vk::ImageLayout::eUndefined,
                  });
}

Image Image::create_1d(Allocator& allocator,
                       uint32_t width,
                       vk::Format format,
                       vk::ImageUsageFlags usage,
                       uint32_t mip_levels,
                       uint32_t array_layers,
                       vk::ArrayProxy<const uint32_t> concurrent_families) {
    const Sharing sharing = sharing_for(concurrent_families);
    return create(allocator,
                  vk::ImageCreateInfo{
                      .imageType = vk::ImageType::e1D,
                      .format = format,
                      .extent = vk::Extent3D{width, 1, 1},
                      .mipLevels = mip_levels,
                      .arrayLayers = array_layers,
                      .samples = vk::SampleCountFlagBits::e1,
                      .tiling = vk::ImageTiling::eOptimal,
                      .usage = usage,
                      .sharingMode = sharing.mode,
                      .queueFamilyIndexCount = sharing.count,
                      .pQueueFamilyIndices = sharing.indices,
                      .initialLayout = vk::ImageLayout::eUndefined,
                  });
}

Image Image::create_3d(Allocator& allocator,
                       vk::Extent3D extent,
                       vk::Format format,
                       vk::ImageUsageFlags usage,
                       uint32_t mip_levels,
                       vk::ArrayProxy<const uint32_t> concurrent_families) {
    const Sharing sharing = sharing_for(concurrent_families);
    return create(allocator,
                  vk::ImageCreateInfo{
                      .imageType = vk::ImageType::e3D,
                      .format = format,
                      .extent = extent,
                      .mipLevels = mip_levels,
                      .arrayLayers = 1,
                      .samples = vk::SampleCountFlagBits::e1,
                      .tiling = vk::ImageTiling::eOptimal,
                      .usage = usage,
                      .sharingMode = sharing.mode,
                      .queueFamilyIndexCount = sharing.count,
                      .pQueueFamilyIndices = sharing.indices,
                      .initialLayout = vk::ImageLayout::eUndefined,
                  });
}

Image Image::create_2d_array(Allocator& allocator,
                            vk::Extent2D extent,
                            uint32_t array_layers,
                            vk::Format format,
                            vk::ImageUsageFlags usage,
                            uint32_t mip_levels,
                            vk::ArrayProxy<const uint32_t> concurrent_families) {
    const Sharing sharing = sharing_for(concurrent_families);
    return create(allocator,
                  vk::ImageCreateInfo{
                      .imageType = vk::ImageType::e2D,
                      .format = format,
                      .extent = vk::Extent3D{extent.width, extent.height, 1},
                      .mipLevels = mip_levels,
                      .arrayLayers = array_layers,
                      .samples = vk::SampleCountFlagBits::e1,
                      .tiling = vk::ImageTiling::eOptimal,
                      .usage = usage,
                      .sharingMode = sharing.mode,
                      .queueFamilyIndexCount = sharing.count,
                      .pQueueFamilyIndices = sharing.indices,
                      .initialLayout = vk::ImageLayout::eUndefined,
                  });
}

Image Image::create_cube(Allocator& allocator,
                         vk::Extent2D extent,
                         vk::Format format,
                         vk::ImageUsageFlags usage,
                         uint32_t mip_levels,
                         vk::ArrayProxy<const uint32_t> concurrent_families) {
    const Sharing sharing = sharing_for(concurrent_families);
    return create(allocator,
                  vk::ImageCreateInfo{
                      .flags = vk::ImageCreateFlagBits::eCubeCompatible,
                      .imageType = vk::ImageType::e2D,
                      .format = format,
                      .extent = vk::Extent3D{extent.width, extent.height, 1},
                      .mipLevels = mip_levels,
                      .arrayLayers = 6,
                      .samples = vk::SampleCountFlagBits::e1,
                      .tiling = vk::ImageTiling::eOptimal,
                      .usage = usage,
                      .sharingMode = sharing.mode,
                      .queueFamilyIndexCount = sharing.count,
                      .pQueueFamilyIndices = sharing.indices,
                      .initialLayout = vk::ImageLayout::eUndefined,
                  });
}

Image Image::create_depth(Allocator& allocator, vk::Extent2D extent, vk::Format format) {
    return create_2d(allocator, extent, format, vk::ImageUsageFlagBits::eDepthStencilAttachment);
}

void Image::destroy() {
    if (image_ != VK_NULL_HANDLE)
        vmaDestroyImage(allocator_, image_, allocation_);

    device_ = nullptr;
    allocator_ = nullptr;
    image_ = VK_NULL_HANDLE;
    allocation_ = nullptr;
    extent_ = vk::Extent3D{};
    image_type_ = vk::ImageType::e2D;
    array_layers_ = 1;
    cube_compatible_ = false;
    format_ = vk::Format::eUndefined;
    usage_ = {};
    sharing_mode_ = vk::SharingMode::eExclusive;
    layout_ = vk::ImageLayout::eUndefined;
    range_ = vk::ImageSubresourceRange{};
}

}
