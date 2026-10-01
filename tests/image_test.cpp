#include <cstdint>

#include <doctest/doctest.h>

#include <vulcao/context.h>
#include <vulcao/image.h>
#include <vulcao/image_view.h>
#include <vulcao/log.h>

#include "common.h"

TEST_CASE("image_aspect_for_format maps depth and stencil formats") {
    using vk::Format;
    using vk::ImageAspectFlagBits;

    CHECK(vulcao::image_aspect_for_format(Format::eD16Unorm) == ImageAspectFlagBits::eDepth);
    CHECK(vulcao::image_aspect_for_format(Format::eX8D24UnormPack32) == ImageAspectFlagBits::eDepth);
    CHECK(vulcao::image_aspect_for_format(Format::eD32Sfloat) == ImageAspectFlagBits::eDepth);

    CHECK(vulcao::image_aspect_for_format(Format::eD16UnormS8Uint) ==
          (ImageAspectFlagBits::eDepth | ImageAspectFlagBits::eStencil));
    CHECK(vulcao::image_aspect_for_format(Format::eD24UnormS8Uint) ==
          (ImageAspectFlagBits::eDepth | ImageAspectFlagBits::eStencil));
    CHECK(vulcao::image_aspect_for_format(Format::eD32SfloatS8Uint) ==
          (ImageAspectFlagBits::eDepth | ImageAspectFlagBits::eStencil));

    CHECK(vulcao::image_aspect_for_format(Format::eR8G8B8A8Unorm) == ImageAspectFlagBits::eColor);
}

TEST_CASE("image_byte_size follows the texel block layout") {
    CHECK(vulcao::image_byte_size(vk::Extent3D{8, 4, 1}, vk::Format::eR8G8B8A8Unorm) == 8u * 4u * 4u);
    CHECK(vulcao::image_byte_size(vk::Extent3D{8, 4, 1}, vk::Format::eR8Unorm) == 8u * 4u);
    CHECK(vulcao::image_byte_size(vk::Extent3D{8, 4, 1}, vk::Format::eR32G32B32A32Sfloat) ==
          8u * 4u * 16u);
    CHECK(vulcao::image_byte_size(vk::Extent3D{8, 4, 3}, vk::Format::eR8G8B8A8Unorm) ==
          8u * 4u * 3u * 4u);

    // BC1 packs a 4x4 texel block into 8 bytes, so partial blocks still count whole.
    CHECK(vulcao::image_byte_size(vk::Extent3D{8, 8, 1}, vk::Format::eBc1RgbaUnormBlock) ==
          2u * 2u * 8u);
    CHECK(vulcao::image_byte_size(vk::Extent3D{5, 5, 1}, vk::Format::eBc1RgbaUnormBlock) ==
          2u * 2u * 8u);
}

TEST_CASE("ImageView derives type, format and range from an image") {
    VULCAO_REQUIRE_DEVICE();

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    const vk::ImageCreateInfo image_info{
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR8G8B8A8Unorm,
        .extent = vk::Extent3D{16, 16, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eSampled,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined,
    };

    const vulcao::Image image = vulcao::Image::create(context.allocator(), image_info);
    CHECK(image.valid());
    CHECK(image.subresource_range().levelCount == 1);
    CHECK(image.subresource_range().layerCount == 1);
    CHECK(image.subresource_range().aspectMask == vk::ImageAspectFlagBits::eColor);

    // The view type, format and range are derived from the image.
    const vulcao::ImageView derived = vulcao::ImageView::create(context.device(), image);
    CHECK(derived.valid());
    CHECK(derived.format() == vk::Format::eR8G8B8A8Unorm);
    CHECK(derived.range().levelCount == 1);
    CHECK(derived.range().layerCount == 1);

    // An explicit subresource range must survive instead of being replaced.
    vk::ImageCreateInfo mipmapped = image_info;
    mipmapped.mipLevels = 4;
    const vulcao::Image mip_image = vulcao::Image::create(context.allocator(), mipmapped);

    const vulcao::ImageView mip_view = vulcao::ImageView::create(
        context.device(), mip_image, vk::ImageViewType::e2D, vk::Format::eR8G8B8A8Unorm,
        vk::ImageSubresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        });

    CHECK(mip_view.valid());
    CHECK(mip_view.range().levelCount == 1);
}
