#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include <vulcao/buffer.h>
#include <vulcao/command_buffer.h>
#include <vulcao/context.h>
#include <vulcao/image.h>
#include <vulcao/log.h>

#include "common.h"

TEST_CASE("download round trips a buffer and a padded image") {
    VULCAO_REQUIRE_DEVICE();

    vulcao::test::ErrorCapture capture;

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    const std::vector<uint32_t> data{10, 20, 30, 40, 50};
    vulcao::Buffer buffer = vulcao::Buffer::create(
        context.allocator(), data.size() * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst);
    context.upload(buffer, data);

    std::vector<uint32_t> readback(data.size());
    context.download(buffer, readback);
    CHECK(readback == data);

    constexpr uint32_t width = 4;
    constexpr uint32_t height = 3;
    constexpr uint32_t padding = 2;
    const uint32_t padded = width + padding;

    std::vector<uint8_t> source(static_cast<size_t>(padded) * height * 4);
    std::vector<uint8_t> expected(static_cast<size_t>(width) * height * 4);
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const size_t src = (static_cast<size_t>(y) * padded + x) * 4;
            source[src + 0] = static_cast<uint8_t>(x * 20);
            source[src + 1] = static_cast<uint8_t>(y * 30);
            source[src + 2] = 90;
            source[src + 3] = 255;

            const size_t dst = (static_cast<size_t>(y) * width + x) * 4;
            expected[dst + 0] = source[src + 0];
            expected[dst + 1] = source[src + 1];
            expected[dst + 2] = source[src + 2];
            expected[dst + 3] = source[src + 3];
        }
    }

    vulcao::Buffer staging = vulcao::Buffer::create(
        context.allocator(), source.size(), vk::BufferUsageFlagBits::eTransferSrc,
        VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
    staging.write(source);

    vulcao::Image image = vulcao::Image::create_2d(
        context.allocator(), vk::Extent2D{width, height}, vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc |
            vk::ImageUsageFlagBits::eTransferDst);

    const vk::ImageSubresourceLayers layers{vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    context.immediate([&](vulcao::CommandBuffer& cmd) {
        cmd.transition(image, vk::ImageLayout::eTransferDstOptimal);
        cmd.copy_buffer_to_image(staging.handle(), image.handle(),
                                 vk::BufferImageCopy{
                                     .bufferOffset = 0,
                                     .bufferRowLength = padded,
                                     .bufferImageHeight = 0,
                                     .imageSubresource = layers,
                                     .imageOffset = vk::Offset3D{0, 0, 0},
                                     .imageExtent = vk::Extent3D{width, height, 1},
                                 });
        cmd.transition(image, vk::ImageLayout::eShaderReadOnlyOptimal);
    });

    std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * 4);
    context.download(image, pixels);
    CHECK(pixels == expected);
    CHECK(image.layout() == vk::ImageLayout::eShaderReadOnlyOptimal);

    CHECK(capture.errors.empty());
}

TEST_CASE("upload and download reject an image payload that is too small") {
    VULCAO_REQUIRE_DEVICE();

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    constexpr uint32_t width = 8;
    constexpr uint32_t height = 8;
    constexpr size_t texel = 4;

    vulcao::Image image = vulcao::Image::create_2d(
        context.allocator(), vk::Extent2D{width, height}, vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc |
            vk::ImageUsageFlagBits::eTransferDst);

    const size_t required = static_cast<size_t>(width) * height * texel;
    const std::vector<uint8_t> exact(required, 0x7f);
    const std::vector<uint8_t> too_small(required - texel, 0x7f);

    CHECK_NOTHROW(context.upload(image, exact));
    CHECK_THROWS_AS(context.upload(image, too_small), std::runtime_error);

    std::vector<uint8_t> readback(required);
    CHECK_NOTHROW(context.download(image, readback));
    CHECK(readback == exact);

    std::vector<uint8_t> short_readback(required - texel);
    CHECK_THROWS_AS(context.download(image, short_readback), std::runtime_error);

    // A payload larger than the image stays accepted, matching the buffer path.
    const std::vector<uint8_t> oversized(required + texel, 0x7f);
    CHECK_NOTHROW(context.upload(image, oversized));
}

TEST_CASE("upload with mip generation requires TransferSrc usage") {
    VULCAO_REQUIRE_DEVICE();

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    constexpr uint32_t width = 8;
    constexpr uint32_t height = 8;
    const std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * 4, 0x40);

    // TransferDst only: uploading level 0 is fine, generating mips is not.
    vulcao::Image no_src = vulcao::Image::create_2d(
        context.allocator(), vk::Extent2D{width, height}, vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
        4 /* mip_levels */);
    CHECK_NOTHROW(context.upload(no_src, pixels));
    CHECK_THROWS_AS(context.upload(no_src, pixels, vk::ImageLayout::eShaderReadOnlyOptimal, true),
                    std::runtime_error);

    // With TransferSrc the mip chain is generated and the layout is tracked.
    vulcao::Image with_src = vulcao::Image::create_2d(
        context.allocator(), vk::Extent2D{width, height}, vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc |
            vk::ImageUsageFlagBits::eTransferDst,
        4 /* mip_levels */);
    CHECK_NOTHROW(
        context.upload(with_src, pixels, vk::ImageLayout::eShaderReadOnlyOptimal, true));
    CHECK(with_src.layout() == vk::ImageLayout::eShaderReadOnlyOptimal);

    // Mip 0 must survive the chain generation unchanged.
    std::vector<uint8_t> readback(pixels.size());
    CHECK_NOTHROW(context.download(with_src, readback));
    CHECK(readback == pixels);
}

TEST_CASE("Buffer::create_with_data folds creation and upload into one call") {
    VULCAO_REQUIRE_DEVICE();

    vulcao::test::ErrorCapture capture;

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    const std::vector<uint32_t> data{7, 11, 13, 17, 19, 23};

    // The container overload adds TransferDst by itself.
    vulcao::Buffer buffer = vulcao::Buffer::create_with_data(
        context, data, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc);
    REQUIRE(buffer.valid());
    CHECK(buffer.size() == data.size() * sizeof(uint32_t));
    CHECK(static_cast<bool>(buffer.usage() & vk::BufferUsageFlagBits::eTransferDst));
    CHECK(static_cast<bool>(buffer.usage() & vk::BufferUsageFlagBits::eStorageBuffer));

    std::vector<uint32_t> readback(data.size());
    context.download(buffer, readback);
    CHECK(readback == data);

    // The raw pointer overload takes any byte range.
    const std::vector<uint8_t> bytes{1, 2, 3, 4, 5, 6, 7, 8};
    vulcao::Buffer raw = vulcao::Buffer::create_with_data(
        context, bytes.data(), static_cast<vk::DeviceSize>(bytes.size()),
        vk::BufferUsageFlagBits::eTransferSrc);
    std::vector<uint8_t> raw_readback(bytes.size());
    context.download(raw, raw_readback);
    CHECK(raw_readback == bytes);

    CHECK(capture.errors.empty());
}

namespace {

/// @brief Context parameters for transfer upload tests: dedicated transfer
///        queue when the GPU offers one, timeline semaphores always.
///
/// separate_transfer_queue is a hard requirement in the device selector, so
/// asking for it unconditionally would leave the suite unable to create a device
/// on a software rasterizer (lavapipe has one graphics|compute|transfer family).
/// The cases below guard their dedicated-queue-only assertions on
/// has_transfer_queue(), so both paths are meaningful.
vulcao::ContextInfo transfer_context_info() {
    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;
    info.separate_transfer_queue = vulcao::test::has_dedicated_transfer_queue();
    info.device_features.timeline_semaphore = true;
    return info;
}

}

TEST_CASE("async upload round trips a buffer through the transfer queue") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::test::ErrorCapture capture;
    vulcao::Context context{transfer_context_info()};
    context.initialize();

    const std::vector<uint32_t> data{42, 43, 44, 45, 46};

    // With a dedicated transfer queue, an exclusive destination is rejected.
    if (context.has_transfer_queue()) {
        vulcao::Buffer exclusive = vulcao::Buffer::create(
            context.allocator(), data.size() * sizeof(uint32_t),
            vk::BufferUsageFlagBits::eTransferDst);
        CHECK_THROWS_AS(context.upload_async(exclusive, data), std::runtime_error);
    }

    vulcao::Buffer buffer = vulcao::Buffer::create(
        context.allocator(), data.size() * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
            vk::BufferUsageFlagBits::eTransferSrc,
        VMA_MEMORY_USAGE_AUTO, 0, context.transfer_sharing_families());

    const vulcao::Context::AsyncUpload upload = context.upload_async(buffer, data);
    CHECK(upload.value > 0);
    CHECK(context.transfer_timeline().valid());

    context.wait_upload(upload);

    // Graphics side: no ownership ceremony, just order the readback after the
    // upload through the timeline.
    vulcao::Buffer readback_stage = vulcao::Buffer::create(
        context.allocator(), buffer.size(), vk::BufferUsageFlagBits::eTransferDst,
        VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

    // A second upload of the same buffer works without extra barriers.
    const vulcao::Context::AsyncUpload second = context.upload_async(buffer, data);
    vulcao::CommandBuffer cmd =
        vulcao::CommandBuffer::allocate(context.device(), context.command_pool());
    cmd.begin();
    cmd.copy_buffer(buffer.handle(), readback_stage.handle(), buffer.size());
    cmd.end();

    vulcao::Fence done = vulcao::Fence::create(context.device());
    context.submit(context.graphics_queue(), cmd.handle(), context.transfer_timeline(),
                   second.value, vk::PipelineStageFlagBits2::eTransfer, done.handle());
    done.wait();

    readback_stage.invalidate(0, buffer.size());
    std::vector<uint32_t> readback(data.size());
    std::memcpy(readback.data(), readback_stage.map(), buffer.size());
    CHECK(readback == data);

    for (const std::string& error : capture.errors)
        MESSAGE("logged error: ", error);
    CHECK(capture.errors.empty());
}

TEST_CASE("async upload moves an image to the graphics family with its final layout") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::test::ErrorCapture capture;
    vulcao::Context context{transfer_context_info()};
    context.initialize();

    constexpr uint32_t width = 4;
    constexpr uint32_t height = 4;
    std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * 4);
    for (size_t i = 0; i < pixels.size(); ++i)
        pixels[i] = static_cast<uint8_t>(i);

    vulcao::Image image = vulcao::Image::create_2d(
        context.allocator(), vk::Extent2D{width, height}, vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc |
            vk::ImageUsageFlagBits::eSampled,
        1, vk::SampleCountFlagBits::e1, context.transfer_sharing_families());

    const vulcao::Context::AsyncUpload upload = context.upload_async(image, pixels);
    CHECK(image.layout() == vk::ImageLayout::eShaderReadOnlyOptimal);

    vulcao::Buffer readback_stage = vulcao::Buffer::create(
        context.allocator(), pixels.size(), vk::BufferUsageFlagBits::eTransferDst,
        VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

    vulcao::CommandBuffer cmd =
        vulcao::CommandBuffer::allocate(context.device(), context.command_pool());
    cmd.begin();
    cmd.transition(image, vk::ImageLayout::eTransferSrcOptimal);
    cmd.copy_image_to_buffer(readback_stage.handle(), image);
    cmd.end();

    vulcao::Fence done = vulcao::Fence::create(context.device());
    context.submit(context.graphics_queue(), cmd.handle(), context.transfer_timeline(),
                   upload.value, vk::PipelineStageFlagBits2::eTransfer, done.handle());
    done.wait();

    readback_stage.invalidate(0, pixels.size());
    std::vector<uint8_t> readback(pixels.size());
    std::memcpy(readback.data(), readback_stage.map(), pixels.size());
    CHECK(readback == pixels);

    for (const std::string& error : capture.errors)
        MESSAGE("logged error: ", error);
    CHECK(capture.errors.empty());
}

TEST_CASE("upload_async requires the timeline semaphore feature") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    vulcao::Buffer buffer = vulcao::Buffer::create(
        context.allocator(), 16, vk::BufferUsageFlagBits::eTransferDst);
    const std::vector<uint32_t> data{1, 2, 3, 4};
    CHECK_THROWS_AS(context.upload_async(buffer, data), std::runtime_error);
}


TEST_CASE("async uploads can be in flight at the same time") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::test::ErrorCapture capture;
    vulcao::Context context{transfer_context_info()};
    context.initialize();

    const std::vector<uint32_t> first_data{1, 2, 3, 4, 5};
    const std::vector<uint32_t> second_data{6, 7, 8, 9, 10};

    vulcao::Buffer first = vulcao::Buffer::create(
        context.allocator(), first_data.size() * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
        VMA_MEMORY_USAGE_AUTO, 0, context.transfer_sharing_families());
    vulcao::Buffer second = vulcao::Buffer::create(
        context.allocator(), second_data.size() * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
        VMA_MEMORY_USAGE_AUTO, 0, context.transfer_sharing_families());

    // Both uploads are started before either receipt is waited on, which is what
    // an asynchronous upload exists for.
    const vulcao::Context::AsyncUpload first_upload = context.upload_async(first, first_data);
    const vulcao::Context::AsyncUpload second_upload = context.upload_async(second, second_data);
    CHECK(first_upload.value > 0);
    CHECK(second_upload.value > first_upload.value);

    vulcao::Buffer first_stage = vulcao::Buffer::create(
        context.allocator(), first.size(), vk::BufferUsageFlagBits::eTransferDst,
        VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
    vulcao::Buffer second_stage = vulcao::Buffer::create(
        context.allocator(), second.size(), vk::BufferUsageFlagBits::eTransferDst,
        VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

    vulcao::CommandBuffer cmd =
        vulcao::CommandBuffer::allocate(context.device(), context.command_pool());
    cmd.begin();
    cmd.copy_buffer(first.handle(), first_stage.handle(), first.size());
    cmd.copy_buffer(second.handle(), second_stage.handle(), second.size());
    cmd.end();

    vulcao::Fence done = vulcao::Fence::create(context.device());
    context.submit(context.graphics_queue(), cmd.handle(), context.transfer_timeline(),
                   second_upload.value, vk::PipelineStageFlagBits2::eTransfer, done.handle());
    done.wait();

    first_stage.invalidate(0, first.size());
    second_stage.invalidate(0, second.size());
    std::vector<uint32_t> first_readback(first_data.size());
    std::vector<uint32_t> second_readback(second_data.size());
    std::memcpy(first_readback.data(), first_stage.map(), first.size());
    std::memcpy(second_readback.data(), second_stage.map(), second.size());
    CHECK(first_readback == first_data);
    CHECK(second_readback == second_data);

    for (const std::string& error : capture.errors)
        MESSAGE("logged error: ", error);
    CHECK(capture.errors.empty());
}

TEST_CASE("async upload replaces an image that is already uploaded") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::test::ErrorCapture capture;
    vulcao::Context context{transfer_context_info()};
    context.initialize();

    constexpr uint32_t width = 4;
    constexpr uint32_t height = 4;
    const std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * 4, 0x7f);

    vulcao::Image image = vulcao::Image::create_2d(
        context.allocator(), vk::Extent2D{width, height}, vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc |
            vk::ImageUsageFlagBits::eSampled,
        1, vk::SampleCountFlagBits::e1, context.transfer_sharing_families());

    // The first upload leaves the image shader read only, so the second one
    // transitions out of a layout that a transfer only queue has no stage for.
    const vulcao::Context::AsyncUpload first = context.upload_async(image, pixels);
    context.wait_upload(first);
    CHECK(image.layout() == vk::ImageLayout::eShaderReadOnlyOptimal);

    const vulcao::Context::AsyncUpload second = context.upload_async(image, pixels);
    context.wait_upload(second);

    for (const std::string& error : capture.errors)
        MESSAGE("logged error: ", error);
    CHECK(capture.errors.empty());
}
