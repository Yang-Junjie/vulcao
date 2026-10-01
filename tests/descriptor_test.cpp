#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <doctest/doctest.h>

#include <vulcao/buffer.h>
#include <vulcao/context.h>
#include <vulcao/descriptor_set.h>
#include <vulcao/image.h>
#include <vulcao/image_view.h>
#include <vulcao/log.h>
#include <vulcao/sampler.h>

#include "common.h"

namespace {

using vk::DescriptorSetLayoutBinding;
using vk::DescriptorType;
using vk::ShaderStageFlagBits;

constexpr uint32_t uniform_binding = 0;
constexpr uint32_t storage_binding = 1;
constexpr uint32_t image_binding = 2;
constexpr uint32_t storage_image_binding = 3;

const std::array<DescriptorSetLayoutBinding, 4> all_bindings{{
    DescriptorSetLayoutBinding{.binding = uniform_binding,
                               .descriptorType = DescriptorType::eUniformBuffer,
                               .descriptorCount = 1,
                               .stageFlags = ShaderStageFlagBits::eVertex},
    DescriptorSetLayoutBinding{.binding = storage_binding,
                               .descriptorType = DescriptorType::eStorageBuffer,
                               .descriptorCount = 1,
                               .stageFlags = ShaderStageFlagBits::eCompute},
    DescriptorSetLayoutBinding{.binding = image_binding,
                               .descriptorType = DescriptorType::eCombinedImageSampler,
                               .descriptorCount = 1,
                               .stageFlags = ShaderStageFlagBits::eFragment},
    DescriptorSetLayoutBinding{.binding = storage_image_binding,
                               .descriptorType = DescriptorType::eStorageImage,
                               .descriptorCount = 1,
                               .stageFlags = ShaderStageFlagBits::eCompute},
}};

const std::array<vk::DescriptorPoolSize, 4> all_sizes{{
    vk::DescriptorPoolSize{DescriptorType::eUniformBuffer, 8},
    vk::DescriptorPoolSize{DescriptorType::eStorageBuffer, 8},
    vk::DescriptorPoolSize{DescriptorType::eCombinedImageSampler, 8},
    vk::DescriptorPoolSize{DescriptorType::eStorageImage, 8},
}};

}

TEST_CASE("descriptor pool allocates up to max_sets and resets") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    const vulcao::DescriptorSetLayout layout =
        vulcao::DescriptorSetLayout::create(context.device(), all_bindings);
    CHECK(layout.valid());

    constexpr uint32_t max_sets = 4;
    vulcao::DescriptorPool pool = vulcao::DescriptorPool::create(context.device(), all_sizes, max_sets);
    CHECK(pool.valid());

    for (uint32_t i = 0; i < max_sets; ++i) {
        const vulcao::DescriptorSet set = pool.allocate(layout);
        CHECK(set.valid());
        CHECK(set.handle() != VK_NULL_HANDLE);
    }

    // The pool is exhausted, so the next allocation has to fail rather than
    // hand out an invalid handle.
    CHECK_THROWS_AS(pool.allocate(layout), std::runtime_error);

    pool.reset();
    CHECK(pool.allocate(layout).valid());
}

TEST_CASE("descriptor pool rejects an empty layout") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    vulcao::DescriptorPool pool = vulcao::DescriptorPool::create(context.device(), all_sizes, 1);
    CHECK_THROWS_AS(pool.allocate(vulcao::DescriptorSetLayout{}), std::runtime_error);
}

TEST_CASE("descriptor set writes match the declared binding types") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::test::ErrorCapture capture;

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    const vulcao::DescriptorSetLayout layout =
        vulcao::DescriptorSetLayout::create(context.device(), all_bindings);
    vulcao::DescriptorPool pool = vulcao::DescriptorPool::create(context.device(), all_sizes, 2);

    vulcao::Buffer uniform = vulcao::Buffer::create(context.allocator(), 64,
                                                   vk::BufferUsageFlagBits::eUniformBuffer);
    vulcao::Buffer storage = vulcao::Buffer::create(context.allocator(), 256,
                                                   vk::BufferUsageFlagBits::eStorageBuffer);
    const vulcao::Sampler sampler = vulcao::Sampler::linear(context.device());
    vulcao::Image image = vulcao::Image::create_2d(
        context.allocator(), vk::Extent2D{4, 4}, vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage);
    const vulcao::ImageView image_view = vulcao::ImageView::create(context.device(), image);

    // The write helpers are const: they update the Vulkan set, not this handle.
    const vulcao::DescriptorSet set = pool.allocate(layout);

    CHECK_NOTHROW(set.write_uniform_buffer(uniform_binding, uniform));
    CHECK_NOTHROW(set.write_storage_buffer(storage_binding, storage));
    CHECK_NOTHROW(set.write_image(image_binding, image_view, sampler));
    CHECK_NOTHROW(set.write_storage_image(storage_image_binding, image_view));

    CHECK_NOTHROW(set.write_buffer(uniform_binding, uniform, DescriptorType::eUniformBuffer, 0, 64));

    // The batching writer must produce the same result through one update call.
    vulcao::DescriptorSetWriter writer{set};
    writer.write_uniform_buffer(uniform_binding, uniform)
        .write_storage_buffer(storage_binding, storage)
        .write_image(image_binding, image_view, sampler)
        .write_storage_image(storage_image_binding, image_view);
    CHECK_NOTHROW(writer.flush());

    // Flushing twice is a no-op, and clear() drops pending writes.
    CHECK_NOTHROW(writer.flush());
    writer.write_uniform_buffer(uniform_binding, uniform);
    writer.clear();
    CHECK_NOTHROW(writer.flush());

    CHECK(capture.errors.empty());
}

TEST_CASE("descriptor set layout cache reuses layouts by binding set") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    vulcao::DescriptorSetLayoutCache cache{context.device()};

    const vk::DescriptorSetLayout first = cache.get(all_bindings);
    CHECK(first != VK_NULL_HANDLE);
    CHECK(cache.get(all_bindings) == first);

    // The cache key normalises binding order, so a permuted list hits the same entry.
    const std::array<DescriptorSetLayoutBinding, 4> permuted{{all_bindings[3], all_bindings[1],
                                                              all_bindings[0], all_bindings[2]}};
    CHECK(cache.get(permuted) == first);

    const std::array<DescriptorSetLayoutBinding, 1> single{{all_bindings[0]}};
    CHECK(cache.get(single) != first);

    // Clearing drops the cached layouts; the cache stays usable afterwards.
    cache.clear();
    CHECK(cache.get(all_bindings) != VK_NULL_HANDLE);
}

TEST_CASE("descriptor set layout cache keys include immutable samplers") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    vulcao::DescriptorSetLayoutCache cache{context.device()};

    const vulcao::Sampler linear = vulcao::Sampler::linear(context.device());
    const vulcao::Sampler nearest = vulcao::Sampler::nearest(context.device());

    const vk::Sampler linear_handle = linear.handle();
    const vk::Sampler nearest_handle = nearest.handle();

    const auto make_binding = [](const vk::Sampler* samplers) {
        return vk::DescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment,
            .pImmutableSamplers = samplers,
        };
    };

    const std::array<vk::DescriptorSetLayoutBinding, 1> with_linear{make_binding(&linear_handle)};
    const std::array<vk::DescriptorSetLayoutBinding, 1> with_nearest{make_binding(&nearest_handle)};
    const std::array<vk::DescriptorSetLayoutBinding, 1> without_samplers{make_binding(nullptr)};

    // Same bindings with different baked-in samplers must not collide.
    const vk::DescriptorSetLayout linear_layout = cache.get(with_linear);
    CHECK(cache.get(with_linear) == linear_layout);
    CHECK(cache.get(with_nearest) != linear_layout);
    CHECK(cache.get(without_samplers) != linear_layout);
    CHECK(cache.get(without_samplers) == cache.get(without_samplers));
}

TEST_CASE("descriptor indexing layouts support binding flags and variable counts") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::test::ErrorCapture capture;

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;
    info.device_features.descriptor_indexing = true;

    vulcao::Context context{info};
    context.initialize();

    using B = vk::DescriptorBindingFlagBits;

    // A bindless-style layout: big partially bound arrays, updated after bind.
    const std::array<DescriptorSetLayoutBinding, 2> indexed_bindings{{
        DescriptorSetLayoutBinding{.binding = 0,
                                   .descriptorType = DescriptorType::eSampledImage,
                                   .descriptorCount = 1024,
                                   .stageFlags = ShaderStageFlagBits::eFragment},
        // The variable descriptor count binding must come last.
        DescriptorSetLayoutBinding{.binding = 1,
                                   .descriptorType = DescriptorType::eStorageBuffer,
                                   .descriptorCount = 64,
                                   .stageFlags = ShaderStageFlagBits::eFragment},
    }};
    const std::array<vk::DescriptorBindingFlags, 2> indexed_flags{{
        B::ePartiallyBound | B::eUpdateAfterBind,
        B::ePartiallyBound | B::eUpdateAfterBind | B::eVariableDescriptorCount,
    }};

    const vulcao::DescriptorSetLayout layout = vulcao::DescriptorSetLayout::create(
        context.device(), indexed_bindings, vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool,
        indexed_flags);
    REQUIRE(layout.valid());

    const std::array<vk::DescriptorPoolSize, 2> sizes{{
        vk::DescriptorPoolSize{DescriptorType::eSampledImage, 2048},
        vk::DescriptorPoolSize{DescriptorType::eStorageBuffer, 128},
    }};
    vulcao::DescriptorPool pool = vulcao::DescriptorPool::create(
        context.device(), sizes, 2, vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind);

    // Declared count allocation and a smaller variable count allocation.
    CHECK(pool.allocate(layout).valid());
    CHECK(pool.allocate(layout, 32).valid());

    // Zero counts (a reflected runtime array) are rejected with a clear error.
    std::array<DescriptorSetLayoutBinding, 1> runtime_array{indexed_bindings[0]};
    runtime_array[0].descriptorCount = 0;
    CHECK_THROWS_AS(vulcao::DescriptorSetLayout::create(context.device(), runtime_array),
                    std::runtime_error);

    // binding_flags must be empty or one entry per binding.
    const std::array<vk::DescriptorBindingFlags, 1> short_flags{{B::ePartiallyBound}};
    CHECK_THROWS_AS(
        vulcao::DescriptorSetLayout::create(context.device(), indexed_bindings, {}, short_flags),
        std::runtime_error);

    // The cache key differentiates binding flags and creation flags.
    const std::array<vk::DescriptorBindingFlags, 2> partial_flags{{
        B::ePartiallyBound,
        B::ePartiallyBound,
    }};
    vulcao::DescriptorSetLayoutCache cache{context.device()};
    const vk::DescriptorSetLayout plain = cache.get(indexed_bindings);
    CHECK(plain != VK_NULL_HANDLE);
    CHECK(cache.get(indexed_bindings) == plain);
    CHECK(cache.get(indexed_bindings, {}, partial_flags) != plain);
    CHECK(cache.get(indexed_bindings, {}, partial_flags) ==
          cache.get(indexed_bindings, {}, partial_flags));
    CHECK(cache.get(indexed_bindings, vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool,
                    indexed_flags) != plain);

    for (const std::string& error : capture.errors)
        MESSAGE("logged error: ", error);
    CHECK(capture.errors.empty());
}

TEST_CASE("descriptor pool create_for_bindings sizes the pool from the bindings") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    // One set uses 2 uniform buffers and 1 combined image sampler; the pool
    // must hold 3 sets of that shape without any hand-counted sizes.
    const std::array<DescriptorSetLayoutBinding, 3> bindings{{
        DescriptorSetLayoutBinding{.binding = 0,
                                   .descriptorType = DescriptorType::eUniformBuffer,
                                   .descriptorCount = 1,
                                   .stageFlags = ShaderStageFlagBits::eVertex},
        DescriptorSetLayoutBinding{.binding = 1,
                                   .descriptorType = DescriptorType::eUniformBuffer,
                                   .descriptorCount = 1,
                                   .stageFlags = ShaderStageFlagBits::eVertex},
        DescriptorSetLayoutBinding{.binding = 2,
                                   .descriptorType = DescriptorType::eCombinedImageSampler,
                                   .descriptorCount = 1,
                                   .stageFlags = ShaderStageFlagBits::eFragment},
    }};

    CHECK_THROWS_AS(vulcao::DescriptorPool::create_for_bindings(context.device(), bindings, 0),
                    std::runtime_error);

    vulcao::DescriptorPool pool =
        vulcao::DescriptorPool::create_for_bindings(context.device(), bindings, 3);
    const vulcao::DescriptorSetLayout layout =
        vulcao::DescriptorSetLayout::create(context.device(), bindings);

    // Exactly three sets fit: the fourth allocation proves the sizes were
    // multiplied by the set count, not more.
    for (uint32_t i = 0; i < 3; ++i)
        CHECK(pool.allocate(layout).valid());
    CHECK_THROWS(pool.allocate(layout));

    // An empty set still gets a usable (sizeless) pool.
    const std::array<DescriptorSetLayoutBinding, 0> no_bindings{};
    vulcao::DescriptorPool empty_pool =
        vulcao::DescriptorPool::create_for_bindings(context.device(), no_bindings, 2);
    const vulcao::DescriptorSetLayout empty_layout =
        vulcao::DescriptorSetLayout::create(context.device(), no_bindings);
    CHECK(empty_pool.allocate(empty_layout).valid());
}

TEST_CASE("texel buffer views and separate image descriptors write cleanly") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::test::ErrorCapture capture;

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    // A texel buffer view needs texel usage on the viewed buffer.
    vulcao::Buffer texels = vulcao::Buffer::create(
        context.allocator(), 16 * sizeof(float),
        vk::BufferUsageFlagBits::eStorageTexelBuffer |
            vk::BufferUsageFlagBits::eUniformTexelBuffer);
    const vulcao::BufferView view =
        vulcao::BufferView::create(context.device(), texels, vk::Format::eR32Sfloat);
    REQUIRE(view.valid());

    vulcao::Buffer not_texels = vulcao::Buffer::create(
        context.allocator(), 64, vk::BufferUsageFlagBits::eVertexBuffer);
    CHECK_THROWS_AS(vulcao::BufferView::create(context.device(), not_texels,
                                               vk::Format::eR32Sfloat),
                    std::runtime_error);
    CHECK_THROWS_AS(vulcao::BufferView::create(context.device(), vulcao::Buffer{},
                                               vk::Format::eR32Sfloat),
                    std::runtime_error);

    constexpr uint32_t sampler_binding = 0;
    constexpr uint32_t sampled_binding = 1;
    constexpr uint32_t input_binding = 2;
    constexpr uint32_t uniform_texel_binding = 3;
    constexpr uint32_t storage_texel_binding = 4;

    const std::array<DescriptorSetLayoutBinding, 5> bindings{{
        DescriptorSetLayoutBinding{.binding = sampler_binding,
                                   .descriptorType = DescriptorType::eSampler,
                                   .descriptorCount = 1,
                                   .stageFlags = ShaderStageFlagBits::eFragment},
        DescriptorSetLayoutBinding{.binding = sampled_binding,
                                   .descriptorType = DescriptorType::eSampledImage,
                                   .descriptorCount = 1,
                                   .stageFlags = ShaderStageFlagBits::eFragment},
        DescriptorSetLayoutBinding{.binding = input_binding,
                                   .descriptorType = DescriptorType::eInputAttachment,
                                   .descriptorCount = 1,
                                   .stageFlags = ShaderStageFlagBits::eFragment},
        DescriptorSetLayoutBinding{.binding = uniform_texel_binding,
                                   .descriptorType = DescriptorType::eUniformTexelBuffer,
                                   .descriptorCount = 1,
                                   .stageFlags = ShaderStageFlagBits::eFragment},
        DescriptorSetLayoutBinding{.binding = storage_texel_binding,
                                   .descriptorType = DescriptorType::eStorageTexelBuffer,
                                   .descriptorCount = 1,
                                   .stageFlags = ShaderStageFlagBits::eCompute},
    }};

    const vulcao::DescriptorSetLayout layout =
        vulcao::DescriptorSetLayout::create(context.device(), bindings);
    vulcao::DescriptorPool pool =
        vulcao::DescriptorPool::create_for_bindings(context.device(), bindings, 1);

    const vulcao::Sampler sampler = vulcao::Sampler::linear(context.device());
    vulcao::Image image = vulcao::Image::create_2d(
        context.allocator(), vk::Extent2D{4, 4}, vk::Format::eR8G8B8A8Unorm,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eInputAttachment);
    const vulcao::ImageView image_view = vulcao::ImageView::create(context.device(), image);

    const vulcao::DescriptorSet set = pool.allocate(layout);
    CHECK_NOTHROW(set.write_sampler(sampler_binding, sampler));
    CHECK_NOTHROW(set.write_sampled_image(sampled_binding, image_view));
    CHECK_NOTHROW(set.write_input_attachment(input_binding, image_view));
    CHECK_NOTHROW(set.write_uniform_texel_buffer(uniform_texel_binding, view));
    CHECK_NOTHROW(set.write_storage_texel_buffer(storage_texel_binding, view));

    // The batching writer covers the same descriptor kinds in one update.
    vulcao::DescriptorSetWriter writer{set};
    writer.write_sampler(sampler_binding, sampler)
        .write_sampled_image(sampled_binding, image_view)
        .write_input_attachment(input_binding, image_view)
        .write_uniform_texel_buffer(uniform_texel_binding, view)
        .write_storage_texel_buffer(storage_texel_binding, view);
    CHECK_NOTHROW(writer.flush());

    for (const std::string& error : capture.errors)
        MESSAGE("logged error: ", error);
    CHECK(capture.errors.empty());
}
