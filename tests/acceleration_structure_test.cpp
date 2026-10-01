#include <array>
#include <stdexcept>

#include <doctest/doctest.h>

#include <vulcao/acceleration_structure.h>
#include <vulcao/buffer.h>
#include <vulcao/command_buffer.h>
#include <vulcao/context.h>
#include <vulcao/descriptor_set.h>
#include <vulcao/log.h>

#include "common.h"

namespace {

struct Vertex {
    float position[3];
};

const std::array<Vertex, 3> kTriangle{{
    {{0.0f, 1.0f, 0.0f}},
    {{1.0f, -1.0f, 0.0f}},
    {{-1.0f, -1.0f, 0.0f}},
}};

vk::BufferUsageFlags build_input_usage() {
    return vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
           vk::BufferUsageFlagBits::eShaderDeviceAddress |
           vk::BufferUsageFlagBits::eStorageBuffer;
}

}

TEST_CASE("acceleration structures build from triangle and instance geometry") {
    VULCAO_REQUIRE_DEVICE();
    VULCAO_REQUIRE_RAY_QUERY();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::test::ErrorCapture capture;

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;
    info.device_features.ray_query = true;

    vulcao::Context context{info};
    context.initialize();

    REQUIRE(context.has_acceleration_structure());
    CHECK(context.has_ray_query());

    const vulcao::Buffer vertex_buffer =
        vulcao::Buffer::create_with_data(context, kTriangle, build_input_usage());
    REQUIRE(vertex_buffer.valid());
    CHECK(vertex_buffer.device_address() != 0);

    const vulcao::TrianglesGeometry geometry{
        .vertex_data = vertex_buffer.device_address(),
        .index_data = 0,
        .vertex_count = static_cast<uint32_t>(kTriangle.size()),
        .index_count = 0,
        .vertex_stride = sizeof(Vertex),
        .vertex_format = vk::Format::eR32G32B32Sfloat,
        .flags = vk::GeometryFlagBitsKHR::eOpaque,
    };
    const std::array<vulcao::TrianglesGeometry, 1> geometries{geometry};

    const vulcao::AccelerationStructureSizes sizes =
        vulcao::blas_build_sizes(context.device(), geometries);
    CHECK(sizes.acceleration_structure_size > 0);
    CHECK(sizes.build_scratch_size > 0);

    vulcao::AccelerationStructure blas = vulcao::AccelerationStructure::create_blas(
        context, geometries, vk::BuildAccelerationStructureFlagsKHR{}, true);
    REQUIRE(blas.valid());
    CHECK(blas.type() == vk::AccelerationStructureTypeKHR::eBottomLevel);
    CHECK(blas.device_address() != 0);
    REQUIRE(blas.scratch().valid());

    context.immediate(
        [&](vulcao::CommandBuffer& cmd) { cmd.build_acceleration_structure(blas, geometries); });

    vulcao::AccelerationStructureInstance instance;
    instance.acceleration_structure = blas.handle();
    const std::array<vulcao::AccelerationStructureInstance, 1> instances{instance};

    const vulcao::Buffer instance_buffer = vulcao::make_instance_buffer(context, instances);
    REQUIRE(instance_buffer.valid());

    vulcao::AccelerationStructure tlas =
        vulcao::AccelerationStructure::create_tlas(context, 1, {}, true);
    REQUIRE(tlas.valid());
    CHECK(tlas.type() == vk::AccelerationStructureTypeKHR::eTopLevel);
    CHECK(tlas.device_address() != 0);
    CHECK(tlas.element_count() == 1);

    context.immediate([&](vulcao::CommandBuffer& cmd) {
        cmd.build_acceleration_structure(tlas, instance_buffer);
    });

    const std::array<vk::DescriptorSetLayoutBinding, 1> bindings{{
        {.binding = 0,
         .descriptorType = vk::DescriptorType::eAccelerationStructureKHR,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute},
    }};
    const vulcao::DescriptorSetLayout layout =
        vulcao::DescriptorSetLayout::create(context.device(), bindings);
    vulcao::DescriptorPool pool =
        vulcao::DescriptorPool::create_for_bindings(context.device(), bindings, 1);
    const vulcao::DescriptorSet set = pool.allocate(layout);
    CHECK_NOTHROW(vulcao::DescriptorSetWriter{set}.write_acceleration_structure(0, tlas).flush());

    context.wait_idle();
    CHECK(capture.errors.empty());
}

TEST_CASE("acceleration structure helpers reject invalid input") {
    VULCAO_REQUIRE_DEVICE();
    VULCAO_REQUIRE_RAY_QUERY();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;
    info.device_features.ray_query = true;

    vulcao::Context context{info};
    context.initialize();

    const std::array<vulcao::TrianglesGeometry, 0> no_geometry{};
    CHECK_THROWS_AS(vulcao::blas_build_sizes(context.device(), no_geometry), std::runtime_error);
    CHECK_THROWS_AS(vulcao::tlas_build_sizes(context.device(), 0), std::runtime_error);

    vulcao::Allocator invalid_allocator;
    CHECK_THROWS_AS(
        vulcao::AccelerationStructure::create(invalid_allocator,
                                              vk::AccelerationStructureTypeKHR::eBottomLevel, 16),
        std::runtime_error);

    // A plain storage buffer has no device address.
    const vulcao::Buffer plain =
        vulcao::Buffer::create(context.allocator(), 16, vk::BufferUsageFlagBits::eStorageBuffer);
    CHECK_THROWS_AS(plain.device_address(), std::runtime_error);
}
