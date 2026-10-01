#include <stdexcept>

#include <doctest/doctest.h>

#include <vulcao/command_buffer.h>
#include <vulcao/context.h>
#include <vulcao/log.h>

#include "common.h"

TEST_CASE("immediate rejects reentrant recording and recovers afterwards") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    // Nesting would reset the command buffer that is currently being recorded.
    CHECK_THROWS_AS(
        context.immediate([&](vulcao::CommandBuffer&) {
            context.immediate([](vulcao::CommandBuffer&) {});
        }),
        std::runtime_error);

    // The guard must be released when the callable throws, otherwise the context
    // would stay locked for the rest of its lifetime.
    CHECK_THROWS_AS(
        context.immediate([](vulcao::CommandBuffer&) { throw std::runtime_error("boom"); }),
        std::runtime_error);

    CHECK_NOTHROW(context.immediate([](vulcao::CommandBuffer&) {}));
}

TEST_CASE("contexts with different validation settings coexist in one process") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    // vk-bootstrap caches its instance function pointers process wide on the first
    // instance it builds and never invalidates them, so a context created without
    // VK_EXT_debug_utils used to break the debug messenger of every later context
    // that asked for validation.
    {
        vulcao::ContextInfo quiet;
        quiet.headless = true;
        quiet.validation = false;

        vulcao::Context first{quiet};
        first.initialize();

        CHECK(first.initialized());
        CHECK_FALSE(first.debug_utils_enabled());
    }

    {
        vulcao::ContextInfo loud;
        loud.headless = true;
        loud.validation = true;

        vulcao::Context second{loud};
        second.initialize();

        CHECK(second.initialized());
        CHECK(second.debug_utils_enabled());
    }
}

TEST_CASE("submit_pooled recycles signaled fences") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    // A trivial command buffer submitted repeatedly. Not one-time submit: it is
    // submitted more than once without re-recording.
    vulcao::CommandBuffer cmd =
        vulcao::CommandBuffer::allocate(context.device(), context.command_pool());
    REQUIRE(cmd.valid());
    cmd.begin(vk::CommandBufferUsageFlags{}).end();

    // The first submission creates a fence in the pool.
    const vk::Fence pooled = context.submit_pooled(cmd.handle());
    CHECK(pooled != VK_NULL_HANDLE);
    CHECK(context.device().waitForFences(pooled, VK_TRUE, UINT64_MAX) == vk::Result::eSuccess);

    // Once signaled, the next pooled submission recycles the same fence instead
    // of creating a new one.
    const vk::Fence recycled = context.submit_pooled(cmd.handle());
    CHECK(recycled == pooled);
    CHECK(context.device().waitForFences(recycled, VK_TRUE, UINT64_MAX) == vk::Result::eSuccess);

    context.wait_idle();
}

TEST_CASE("context reports the selected device and its properties") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    const vulcao::PhysicalDeviceInfo device = context.physical_device_info();
    CHECK_FALSE(device.name.empty());
    CHECK(device.api_version != 0);
    CHECK(context.timestamp_period() > 0.0f);
    CHECK(context.memory_properties().memoryHeapCount > 0);
    CHECK(context.info().headless);
}

TEST_CASE("optional Vulkan 1.1 and 1.2 device features can be requested") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;
    info.device_features.scalar_block_layout = true;
    info.device_features.shader_float16 = true;
    info.device_features.shader_int8 = true;
    info.device_features.storage_buffer_8bit = true;
    info.device_features.storage_buffer_16bit = true;
    info.device_features.shader_subgroup_extended_types = true;

    // These are optional, so a device without them skips rather than fails.
    try {
        vulcao::Context context{info};
        context.initialize();
        CHECK(context.initialized());
    } catch (const std::exception& error) {
        MESSAGE("skipping: the device does not offer the requested features: ", error.what());
    }
}
