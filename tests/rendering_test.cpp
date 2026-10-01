#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include <doctest/doctest.h>

#include <vulcao/command_buffer.h>
#include <vulcao/context.h>
#include <vulcao/image.h>
#include <vulcao/image_view.h>
#include <vulcao/log.h>
#include <vulcao/rendering.h>

#include "common.h"

TEST_CASE("attachment builders fill the rendering structures") {
    const vk::ClearColorValue clear{std::array<float, 4>{0.25f, 0.5f, 0.75f, 1.0f}};

    const vk::RenderingAttachmentInfo cleared = vulcao::color_attachment(
        vk::ImageView{}, vk::ImageLayout::eColorAttachmentOptimal, clear);
    CHECK(cleared.loadOp == vk::AttachmentLoadOp::eClear);
    CHECK(cleared.storeOp == vk::AttachmentStoreOp::eStore);
    CHECK(cleared.imageLayout == vk::ImageLayout::eColorAttachmentOptimal);
    CHECK(cleared.clearValue.color.float32[1] == 0.5f);

    const vk::RenderingAttachmentInfo loaded =
        vulcao::color_attachment(vk::ImageView{}, vk::ImageLayout::eGeneral);
    CHECK(loaded.loadOp == vk::AttachmentLoadOp::eLoad);
    CHECK(loaded.storeOp == vk::AttachmentStoreOp::eStore);
    CHECK(loaded.imageLayout == vk::ImageLayout::eGeneral);

    const vk::RenderingAttachmentInfo depth = vulcao::depth_attachment(
        vk::ImageView{}, vk::ImageLayout::eDepthStencilAttachmentOptimal, 0.5f, 3);
    CHECK(depth.loadOp == vk::AttachmentLoadOp::eClear);
    CHECK(depth.clearValue.depthStencil.depth == 0.5f);
    CHECK(depth.clearValue.depthStencil.stencil == 3);

    const vk::RenderingAttachmentInfo depth_clear = vulcao::depth_attachment(
        vk::ImageView{}, vk::ImageLayout::eDepthStencilAttachmentOptimal, 1.0f);
    CHECK(depth_clear.loadOp == vk::AttachmentLoadOp::eClear);
    CHECK(depth_clear.clearValue.depthStencil.stencil == 0);

    const vk::RenderingAttachmentInfo depth_loaded = vulcao::depth_attachment(
        vk::ImageView{}, vk::ImageLayout::eDepthStencilReadOnlyOptimal);
    CHECK(depth_loaded.loadOp == vk::AttachmentLoadOp::eLoad);
    CHECK(depth_loaded.imageLayout == vk::ImageLayout::eDepthStencilReadOnlyOptimal);
}

TEST_CASE("begin_rendering with a built attachment clears an offscreen image") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::test::ErrorCapture capture;

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    constexpr uint32_t width = 4;
    constexpr uint32_t height = 4;
    vulcao::Image image = vulcao::Image::create_2d(
        context.allocator(), vk::Extent2D{width, height}, vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc);
    const vulcao::ImageView image_view = vulcao::ImageView::create(context.device(), image);

    const vk::ClearColorValue clear{std::array<float, 4>{0.25f, 0.5f, 0.75f, 1.0f}};

    context.immediate([&](vulcao::CommandBuffer& cmd) {
        cmd.transition(image, vk::ImageLayout::eColorAttachmentOptimal);
        // The clear happens at the attachment load op, no draw is needed.
        cmd.begin_rendering(vk::Extent2D{width, height},
                            vulcao::color_attachment(image_view.handle(), image.layout(), clear));
        cmd.end_rendering();
    });

    // 0.25/0.5/0.75 map to 64/128/191 in unorm.
    const std::array<uint8_t, 4> expected{64, 128, 191, 255};
    std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * 4);
    context.download(image, pixels);
    for (size_t i = 0; i < pixels.size(); i += 4)
        CHECK(std::equal(pixels.begin() + static_cast<ptrdiff_t>(i),
                         pixels.begin() + static_cast<ptrdiff_t>(i + 4), expected.begin()));

    CHECK(capture.errors.empty());
}
