#include <array>
#include <cstdint>
#include <vector>

#include <doctest/doctest.h>

#include <vulcao/buffer.h>
#include <vulcao/command_buffer.h>
#include <vulcao/context.h>
#include <vulcao/event.h>
#include <vulcao/image.h>
#include <vulcao/image_view.h>
#include <vulcao/log.h>
#include <vulcao/rendering.h>

#include "common.h"

TEST_CASE("events signal from the host and across submissions") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::test::ErrorCapture capture;

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    vulcao::Event event = vulcao::Event::create(context.device());
    REQUIRE(event.valid());
    CHECK_FALSE(event.signaled());

    event.set();
    CHECK(event.signaled());
    event.reset();
    CHECK_FALSE(event.signaled());

    const vk::MemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
    };
    const vk::DependencyInfo dependency{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier,
    };

    // The event is set by the first submission, which immediate() waits on.
    context.immediate(
        [&](vulcao::CommandBuffer& cmd) { cmd.set_event(event.handle(), dependency); });
    CHECK(event.signaled());

    // Waiting on an already set event in a second submission must not hang.
    CHECK_NOTHROW(context.immediate([&](vulcao::CommandBuffer& cmd) {
        cmd.wait_event(event.handle(), vk::PipelineStageFlagBits2::eAllCommands,
                       vk::PipelineStageFlagBits2::eAllCommands);
        cmd.reset_event(event.handle(), vk::PipelineStageFlagBits2::eAllCommands);
    }));
    CHECK_FALSE(event.signaled());

    context.wait_idle();
    CHECK(capture.errors.empty());
}

TEST_CASE("copy_buffer copies several regions in one call") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::test::ErrorCapture capture;

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    const std::array<uint32_t, 8> source{1, 2, 3, 4, 5, 6, 7, 8};
    const vulcao::Buffer src = vulcao::Buffer::create_with_data(
        context, source, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc);
    vulcao::Buffer dst = vulcao::Buffer::create(
        context.allocator(), sizeof(source),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
            vk::BufferUsageFlagBits::eTransferDst,
        VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

    context.immediate([&](vulcao::CommandBuffer& cmd) {
        const std::array<vk::BufferCopy, 2> regions{{
            vk::BufferCopy{.srcOffset = 0, .dstOffset = 16, .size = 8},
            vk::BufferCopy{.srcOffset = 16, .dstOffset = 0, .size = 8},
        }};
        cmd.copy_buffer(src.handle(), dst.handle(), regions);
    });

    dst.invalidate();
    const auto* result = static_cast<const uint32_t*>(dst.map());
    REQUIRE(result != nullptr);
    CHECK(result[0] == 5);
    CHECK(result[1] == 6);
    CHECK(result[4] == 1);
    CHECK(result[5] == 2);
    dst.unmap();

    context.wait_idle();
    CHECK(capture.errors.empty());
}

TEST_CASE("resolve_image resolves a multisampled image") {
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

    vulcao::Image msaa = vulcao::Image::create_2d(
        context.allocator(), vk::Extent2D{width, height}, vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc, 1,
        vk::SampleCountFlagBits::e4);
    vulcao::Image resolved = vulcao::Image::create_2d(
        context.allocator(), vk::Extent2D{width, height}, vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst);
    const vulcao::ImageView msaa_view = vulcao::ImageView::create(context.device(), msaa);

    const vk::ClearColorValue red{std::array<float, 4>{1.0f, 0.0f, 0.0f, 1.0f}};
    const vk::ImageSubresourceLayers layers{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .mipLevel = 0,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };

    context.immediate([&](vulcao::CommandBuffer& cmd) {
        cmd.transition(msaa, vk::ImageLayout::eColorAttachmentOptimal);
        cmd.begin_rendering(vk::Extent2D{width, height},
                            vulcao::color_attachment(msaa_view.handle(), msaa.layout(), red));
        cmd.end_rendering();

        cmd.transition(msaa, vk::ImageLayout::eTransferSrcOptimal);
        cmd.transition(resolved, vk::ImageLayout::eTransferDstOptimal);
        cmd.resolve_image(msaa.handle(), vk::ImageLayout::eTransferSrcOptimal, resolved.handle(),
                          vk::ImageLayout::eTransferDstOptimal,
                          vk::ImageResolve{
                              .srcSubresource = layers,
                              .srcOffset = vk::Offset3D{0, 0, 0},
                              .dstSubresource = layers,
                              .dstOffset = vk::Offset3D{0, 0, 0},
                              .extent = vk::Extent3D{width, height, 1},
                          });
    });

    std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * 4);
    context.download(resolved, pixels);
    for (size_t i = 0; i < pixels.size(); i += 4) {
        CHECK(pixels[i + 0] == 255);
        CHECK(pixels[i + 1] == 0);
        CHECK(pixels[i + 2] == 0);
        CHECK(pixels[i + 3] == 255);
    }

    context.wait_idle();
    CHECK(capture.errors.empty());
}

TEST_CASE("clear_attachments clears a region of the attachment") {
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

    const vk::ClearColorValue black{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}};
    const vk::ClearColorValue red{std::array<float, 4>{1.0f, 0.0f, 0.0f, 1.0f}};

    vk::ClearValue clear_value;
    clear_value.color = red;
    const std::array<vk::ClearAttachment, 1> attachments{{
        vk::ClearAttachment{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .colorAttachment = 0,
            .clearValue = clear_value,
        },
    }};
    const std::array<vk::ClearRect, 1> rects{{
        vk::ClearRect{
            .rect = vk::Rect2D{.offset = vk::Offset2D{0, 0}, .extent = vk::Extent2D{2, 2}},
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    }};

    context.immediate([&](vulcao::CommandBuffer& cmd) {
        cmd.transition(image, vk::ImageLayout::eColorAttachmentOptimal);
        cmd.begin_rendering(
            vk::Extent2D{width, height},
            vulcao::color_attachment(image_view.handle(), image.layout(), black));
        cmd.clear_attachments(attachments, rects);
        cmd.end_rendering();
    });

    std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * 4);
    context.download(image, pixels);

    // Pixel (0, 0) is inside the cleared region, pixel (2, 2) is not.
    const auto pixel = [&](uint32_t x, uint32_t y) {
        return &pixels[(static_cast<size_t>(y) * width + x) * 4];
    };
    CHECK(pixel(0, 0)[0] == 255);
    CHECK(pixel(1, 1)[0] == 255);
    CHECK(pixel(2, 2)[0] == 0);
    CHECK(pixel(3, 3)[0] == 0);

    context.wait_idle();
    CHECK(capture.errors.empty());
}
