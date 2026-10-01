#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <stdexcept>
#include <vector>

#include <doctest/doctest.h>

#include <vulcao/context.h>
#include <vulcao/descriptor_set.h>
#include <vulcao/log.h>
#include <vulcao/pipeline.h>
#include <vulcao/pipeline_cache.h>
#include <vulcao/pipeline_layout.h>
#include <vulcao/shader_module.h>
#include <vulcao/vertex_layout.h>

#include "common.h"

#ifdef VULCAO_HAVE_TEST_SHADERS

namespace {

struct Vertex {
    float position[3];
    float color[3];
};

/// Reads a SPIR-V file into words, returning an empty vector when it cannot be read.
std::vector<uint32_t> read_spirv(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        return {};

    const std::streamsize size = file.tellg();
    if (size <= 0 || size % 4 != 0)
        return {};

    std::vector<uint32_t> words(static_cast<size_t>(size) / sizeof(uint32_t));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(words.data()), size))
        return {};
    return words;
}

const vk::DescriptorSetLayoutBinding* find_binding(const vulcao::ShaderReflection& reflection,
                                                   uint32_t set,
                                                   uint32_t binding) {
    for (const vk::DescriptorSetLayoutBinding& candidate : reflection.bindings_for_set(set))
        if (candidate.binding == binding)
            return &candidate;
    return nullptr;
}

}

TEST_CASE("shader reflection reports what each stage uses") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::test::ErrorCapture capture;

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    const std::filesystem::path dir = vulcao::test::shader_dir();
    const vulcao::ShaderModule vertex = vulcao::ShaderModule::create_from_file(
        context.device(), vk::ShaderStageFlagBits::eVertex, dir / "reflection.vert.spv");
    const vulcao::ShaderModule fragment = vulcao::ShaderModule::create_from_file(
        context.device(), vk::ShaderStageFlagBits::eFragment, dir / "reflection.frag.spv");
    const vulcao::ShaderModule compute = vulcao::ShaderModule::create_from_file(
        context.device(), vk::ShaderStageFlagBits::eCompute, dir / "reflection.comp.spv");

    REQUIRE(vertex.valid());
    REQUIRE(fragment.valid());
    REQUIRE(compute.valid());
    CHECK(vertex.stage() == vk::ShaderStageFlagBits::eVertex);

    // The vertex stage reads one uniform buffer and two vertex inputs.
    const vulcao::ShaderReflection& vertex_reflection = vertex.reflection();
    CHECK(vertex_reflection.stage == vk::ShaderStageFlagBits::eVertex);
    CHECK(vertex_reflection.bindings_for_set(0).size() == 1);
    const vk::DescriptorSetLayoutBinding* vertex_uniform =
        find_binding(vertex_reflection, 0, 0);
    REQUIRE(vertex_uniform != nullptr);
    CHECK(vertex_uniform->descriptorType == vk::DescriptorType::eUniformBuffer);
    CHECK(vertex_uniform->stageFlags == vk::ShaderStageFlagBits::eVertex);
    CHECK(vertex_reflection.push_constants.empty());

    REQUIRE(vertex_reflection.vertex_attributes.size() == 2);
    const auto& attributes = vertex_reflection.vertex_attributes;
    CHECK(attributes[0].location == 0);
    CHECK(attributes[0].format == vk::Format::eR32G32B32Sfloat);
    CHECK(attributes[1].location == 1);
    CHECK(attributes[1].format == vk::Format::eR32G32B32Sfloat);

    // The fragment stage uses no descriptors at all, only push constants.
    const vulcao::ShaderReflection& fragment_reflection = fragment.reflection();
    CHECK(fragment_reflection.stage == vk::ShaderStageFlagBits::eFragment);
    CHECK(fragment_reflection.sets.empty());
    REQUIRE(fragment_reflection.push_constants.size() == 1);
    CHECK(fragment_reflection.push_constants.front().offset == 0);
    CHECK(fragment_reflection.push_constants.front().stageFlags == vk::ShaderStageFlagBits::eFragment);
    // float4 tint plus a uint, so at least 20 bytes and far below the guaranteed minimum.
    CHECK(fragment_reflection.push_constants.front().size >= 20);
    CHECK(fragment_reflection.push_constants.front().size <= 128);

    // The compute stage reads a uniform buffer and a storage buffer.
    const vulcao::ShaderReflection& compute_reflection = compute.reflection();
    CHECK(compute_reflection.stage == vk::ShaderStageFlagBits::eCompute);
    REQUIRE(compute_reflection.sets.size() == 1);
    CHECK(compute_reflection.sets.front().set == 0);
    CHECK(compute_reflection.bindings_for_set(0).size() == 2);

    const vk::DescriptorSetLayoutBinding* compute_uniform = find_binding(compute_reflection, 0, 0);
    const vk::DescriptorSetLayoutBinding* compute_storage = find_binding(compute_reflection, 0, 1);
    REQUIRE(compute_uniform != nullptr);
    REQUIRE(compute_storage != nullptr);
    CHECK(compute_uniform->descriptorType == vk::DescriptorType::eUniformBuffer);
    CHECK(compute_storage->descriptorType == vk::DescriptorType::eStorageBuffer);
    CHECK(compute_storage->stageFlags == vk::ShaderStageFlagBits::eCompute);
    REQUIRE(compute_reflection.push_constants.size() == 1);
    CHECK(compute_reflection.push_constants.front().offset == 0);

    CHECK(capture.errors.empty());
}

TEST_CASE("shader module builds from memory and rejects bad input") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    const std::filesystem::path path = vulcao::test::shader_dir() / "reflection.comp.spv";
    const std::vector<uint32_t> spirv = read_spirv(path);
    REQUIRE(!spirv.empty());

    const vulcao::ShaderModule from_memory = vulcao::ShaderModule::create(
        context.device(), vk::ShaderStageFlagBits::eCompute, spirv);
    const vulcao::ShaderModule from_file = vulcao::ShaderModule::create_from_file(
        context.device(), vk::ShaderStageFlagBits::eCompute, path);

    REQUIRE(from_memory.valid());
    REQUIRE(from_file.valid());
    CHECK(from_memory.reflection().sets.size() == from_file.reflection().sets.size());
    CHECK(from_memory.reflection().push_constants.size() ==
          from_file.reflection().push_constants.size());

    CHECK_THROWS_AS(
        vulcao::ShaderModule::create(context.device(), vk::ShaderStageFlagBits::eCompute,
                                     std::span<const uint32_t>{}),
        std::runtime_error);
    CHECK_THROWS_AS(vulcao::ShaderModule::create_from_file(
                        context.device(), vk::ShaderStageFlagBits::eCompute,
                        std::filesystem::path{"vulcao_no_such_shader.spv"}),
                    std::runtime_error);
}

TEST_CASE("pipeline layout merges reflections and tracks owned set layouts") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    const std::filesystem::path dir = vulcao::test::shader_dir();
    const vulcao::ShaderModule vertex = vulcao::ShaderModule::create_from_file(
        context.device(), vk::ShaderStageFlagBits::eVertex, dir / "reflection.vert.spv");
    const vulcao::ShaderModule fragment = vulcao::ShaderModule::create_from_file(
        context.device(), vk::ShaderStageFlagBits::eFragment, dir / "reflection.frag.spv");
    REQUIRE(vertex.valid());
    REQUIRE(fragment.valid());

    const std::array<vulcao::ShaderReflection, 2> reflections{vertex.reflection(),
                                                              fragment.reflection()};

    // Owning variant: the layout keeps the descriptor set layouts alive.
    const vulcao::PipelineLayout owning =
        vulcao::PipelineLayout::create_from_reflection(context.device(), reflections);
    CHECK(owning.valid());
    CHECK(owning.set_count() == 1);
    CHECK(owning.set_layout(0) != VK_NULL_HANDLE);
    CHECK(owning.set_layouts().size() == 1);

    // Cached variant: the cache owns the set layouts, so the layout owns none.
    vulcao::DescriptorSetLayoutCache cache{context.device()};
    const vulcao::PipelineLayout cached =
        vulcao::PipelineLayout::create_from_reflection(context.device(), reflections, &cache);
    CHECK(cached.valid());
    CHECK(cached.set_count() == 1);
    CHECK(cached.set_layouts().empty());

    // Manual variant from raw handles.
    const std::array<vk::DescriptorSetLayout, 1> raw_layouts{owning.set_layout(0)};
    const std::array<vk::PushConstantRange, 1> ranges{vk::PushConstantRange{
        .stageFlags = vk::ShaderStageFlagBits::eFragment, .offset = 0, .size = 20}};
    const vulcao::PipelineLayout manual =
        vulcao::PipelineLayout::create(context.device(), raw_layouts, ranges);
    CHECK(manual.valid());
    CHECK(manual.set_count() == 1);
    CHECK(manual.set_layouts().empty());

    CHECK_THROWS_AS(owning.set_layout(4), std::out_of_range);
}

TEST_CASE("pipeline layout from reflection fills gaps between used sets") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    // A stage that only uses set 2: set 0 and set 1 must be filled with empty
    // layouts so vk::PipelineLayoutCreateInfo indexes layouts by set number.
    vulcao::ShaderReflection reflection;
    reflection.stage = vk::ShaderStageFlagBits::eCompute;
    reflection.sets.push_back(vulcao::DescriptorSetLayoutInfo{
        .set = 2,
        .bindings = {vk::DescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        }},
    });

    const vulcao::PipelineLayout layout = vulcao::PipelineLayout::create_from_reflection(
        context.device(), std::span(&reflection, 1));
    REQUIRE(layout.valid());
    CHECK(layout.set_count() == 3);
    CHECK(layout.set_layouts().size() == 3);
    for (uint32_t set = 0; set < 3; ++set)
        CHECK(layout.set_layout(set) != VK_NULL_HANDLE);
    CHECK_THROWS_AS(layout.set_layout(3), std::out_of_range);

    // The gap layouts are empty and therefore interchangeable: allocating a
    // set from the shader's set 2 layout must carry the storage buffer binding.
    vulcao::DescriptorPool pool = vulcao::DescriptorPool::create(
        context.device(),
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 1},
        1);
    const vulcao::DescriptorSet descriptor_set = pool.allocate(layout.set_layout(2));
    CHECK(descriptor_set.valid());

    // Cached variant goes through the same gap filling.
    vulcao::DescriptorSetLayoutCache cache{context.device()};
    const vulcao::PipelineLayout cached =
        vulcao::PipelineLayout::create_from_reflection(context.device(), std::span(&reflection, 1),
                                                       &cache);
    REQUIRE(cached.valid());
    CHECK(cached.set_count() == 3);
    CHECK(cached.set_layout(2) != VK_NULL_HANDLE);
}

TEST_CASE("compute pipelines build from reflected layouts, with and without a cache") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::test::ErrorCapture capture;

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    const vulcao::ShaderModule compute = vulcao::ShaderModule::create_from_file(
        context.device(), vk::ShaderStageFlagBits::eCompute,
        vulcao::test::shader_dir() / "reflection.comp.spv");
    REQUIRE(compute.valid());

    const vulcao::PipelineLayout layout = vulcao::PipelineLayout::create_from_reflection(
        context.device(), std::span(&compute.reflection(), 1));
    REQUIRE(layout.valid());

    const vulcao::Pipeline pipeline =
        vulcao::Pipeline::create_compute(context.device(), layout, compute, "computeMain");
    CHECK(pipeline.valid());
    CHECK(pipeline.bind_point() == vk::PipelineBindPoint::eCompute);

    vulcao::PipelineCache cache = vulcao::PipelineCache::create(context.device());
    REQUIRE(cache.valid());
    const vulcao::Pipeline cached = vulcao::Pipeline::create_compute(
        context.device(), layout, compute, "computeMain", &cache);
    CHECK(cached.valid());

    // The serialized cache round trips into a new cache object.
    const std::vector<uint8_t> blob = cache.data();
    CHECK_NOTHROW(vulcao::PipelineCache::create(context.device(), blob.data(), blob.size()));

    CHECK_THROWS_AS(
        vulcao::Pipeline::create_compute(context.device(), vulcao::PipelineLayout{}, compute,
                                         "computeMain"),
        std::runtime_error);

    // bind_pipeline(const Pipeline&) picks the pipeline's own bind point.
    CHECK_NOTHROW(context.immediate(
        [&](vulcao::CommandBuffer& cmd) { cmd.bind_pipeline(pipeline); }));

    CHECK(capture.errors.empty());
}

TEST_CASE("graphics pipelines build from a reflected vertex layout") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::test::ErrorCapture capture;

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    const std::filesystem::path dir = vulcao::test::shader_dir();
    const vulcao::ShaderModule vertex = vulcao::ShaderModule::create_from_file(
        context.device(), vk::ShaderStageFlagBits::eVertex, dir / "reflection.vert.spv");
    const vulcao::ShaderModule fragment = vulcao::ShaderModule::create_from_file(
        context.device(), vk::ShaderStageFlagBits::eFragment, dir / "reflection.frag.spv");
    REQUIRE(vertex.valid());
    REQUIRE(fragment.valid());

    const vulcao::VertexLayout vertex_layout = vulcao::make_vertex_layout<Vertex>(
        vertex.reflection(), {offsetof(Vertex, position), offsetof(Vertex, color)});
    REQUIRE(vertex_layout.attributes.size() == 2);
    CHECK(vertex_layout.bindings.size() == 1);
    CHECK(vertex_layout.bindings.front().stride == sizeof(Vertex));
    CHECK(vertex_layout.bindings.front().inputRate == vk::VertexInputRate::eVertex);
    CHECK(vertex_layout.attributes[0].offset == offsetof(Vertex, position));
    CHECK(vertex_layout.attributes[1].offset == offsetof(Vertex, color));

    const std::array<vulcao::ShaderReflection, 2> reflections{vertex.reflection(),
                                                              fragment.reflection()};
    const vulcao::PipelineLayout layout =
        vulcao::PipelineLayout::create_from_reflection(context.device(), reflections);
    REQUIRE(layout.valid());

    const vulcao::GraphicsPipelineInfo pipeline_info{
        .vertex_shader = vertex.handle(),
        .fragment_shader = fragment.handle(),
        .vertex_entry = "vertMain",
        .fragment_entry = "fragMain",
        .vertex_bindings = vertex_layout.bindings,
        .vertex_attributes = vertex_layout.attributes,
        .color_formats = {vk::Format::eB8G8R8A8Unorm},
    };

    const vulcao::Pipeline pipeline =
        vulcao::Pipeline::create_graphics(context.device(), layout, pipeline_info);
    CHECK(pipeline.valid());
    CHECK(pipeline.bind_point() == vk::PipelineBindPoint::eGraphics);

    // Documented preconditions of create_graphics.
    CHECK_THROWS_AS(
        vulcao::Pipeline::create_graphics(context.device(), vulcao::PipelineLayout{}, pipeline_info),
        std::runtime_error);

    vulcao::GraphicsPipelineInfo missing_fragment = pipeline_info;
    missing_fragment.fragment_shader = vk::ShaderModule{};
    CHECK_THROWS_AS(
        vulcao::Pipeline::create_graphics(context.device(), layout, missing_fragment),
        std::runtime_error);

    // make_vertex_layout rejects a member offset list that does not match.
    CHECK_THROWS_AS(vulcao::make_vertex_layout<Vertex>(vertex.reflection(), {0}), std::runtime_error);

    CHECK(capture.errors.empty());
}

TEST_CASE("shader draw parameters can be requested for a vertex index shader") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::test::ErrorCapture capture;

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;
    info.device_features.shader_draw_parameters = true;

    vulcao::Context context{info};
    context.initialize();

    // This entry point reads SV_VertexID, so its SPIR-V declares the
    // DrawParameters capability and creating the module without
    // shader_draw_parameters would raise a validation error.
    const std::filesystem::path dir = vulcao::test::shader_dir();
    const vulcao::ShaderModule vertex = vulcao::ShaderModule::create_from_file(
        context.device(), vk::ShaderStageFlagBits::eVertex, dir / "reflection.index.vert.spv");
    const vulcao::ShaderModule fragment = vulcao::ShaderModule::create_from_file(
        context.device(), vk::ShaderStageFlagBits::eFragment, dir / "reflection.frag.spv");
    REQUIRE(vertex.valid());
    REQUIRE(fragment.valid());

    const std::array<vulcao::ShaderReflection, 2> reflections{vertex.reflection(),
                                                              fragment.reflection()};
    const vulcao::PipelineLayout layout =
        vulcao::PipelineLayout::create_from_reflection(context.device(), reflections);
    REQUIRE(layout.valid());

    const vulcao::Pipeline pipeline = vulcao::Pipeline::create_graphics(
        context.device(), layout,
        vulcao::GraphicsPipelineInfo{
            .vertex_shader = vertex.handle(),
            .fragment_shader = fragment.handle(),
            .vertex_entry = "vertexIndexMain",
            .fragment_entry = "fragMain",
            .color_formats = {vk::Format::eB8G8R8A8Unorm},
        });
    CHECK(pipeline.valid());

    CHECK(capture.errors.empty());
}

TEST_CASE("graphics pipelines accept explicit blend state and raw descriptions") {
    VULCAO_REQUIRE_DEVICE();

    const vulcao::test::LogLevelGuard log_level_guard;
    vulcao::set_log_level(vulcao::LogLevel::warning);

    vulcao::test::ErrorCapture capture;

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;

    vulcao::Context context{info};
    context.initialize();

    const std::filesystem::path dir = vulcao::test::shader_dir();
    const vulcao::ShaderModule vertex = vulcao::ShaderModule::create_from_file(
        context.device(), vk::ShaderStageFlagBits::eVertex, dir / "reflection.vert.spv");
    const vulcao::ShaderModule fragment = vulcao::ShaderModule::create_from_file(
        context.device(), vk::ShaderStageFlagBits::eFragment, dir / "reflection.frag.spv");
    REQUIRE(vertex.valid());
    REQUIRE(fragment.valid());

    const std::array<vulcao::ShaderReflection, 2> reflections{vertex.reflection(),
                                                              fragment.reflection()};
    const vulcao::PipelineLayout layout =
        vulcao::PipelineLayout::create_from_reflection(context.device(), reflections);
    REQUIRE(layout.valid());

    const vulcao::VertexLayout vertex_layout = vulcao::make_vertex_layout<Vertex>(
        vertex.reflection(), {offsetof(Vertex, position), offsetof(Vertex, color)});

    // Explicit per-attachment blend must win over the default derived from blend.
    const vulcao::GraphicsPipelineInfo pipeline_info{
        .vertex_shader = vertex.handle(),
        .fragment_shader = fragment.handle(),
        .vertex_entry = "vertMain",
        .fragment_entry = "fragMain",
        .topology = vk::PrimitiveTopology::eTriangleStrip,
        .vertex_bindings = vertex_layout.bindings,
        .vertex_attributes = vertex_layout.attributes,
        .color_formats = {vk::Format::eB8G8R8A8Unorm},
        .primitive_restart = true,
        .color_blend_attachments = {vk::PipelineColorBlendAttachmentState{
            .blendEnable = VK_TRUE,
            .srcColorBlendFactor = vk::BlendFactor::eOne,
            .dstColorBlendFactor = vk::BlendFactor::eZero,
            .colorBlendOp = vk::BlendOp::eAdd,
            .colorWriteMask = vk::ColorComponentFlagBits::eR,
        }},
        .blend_constants = {0.1f, 0.2f, 0.3f, 0.4f},
    };

    const vulcao::Pipeline pipeline =
        vulcao::Pipeline::create_graphics(context.device(), layout, pipeline_info);
    CHECK(pipeline.valid());

    // The raw overload takes a full Vulkan description and borrows the layout.
    const std::array<vk::PipelineShaderStageCreateInfo, 2> stages{{
        {.stage = vk::ShaderStageFlagBits::eVertex,
         .module = vertex.handle(),
         .pName = "vertMain"},
        {.stage = vk::ShaderStageFlagBits::eFragment,
         .module = fragment.handle(),
         .pName = "fragMain"},
    }};
    const vk::PipelineVertexInputStateCreateInfo vertex_input{
        .vertexBindingDescriptionCount = static_cast<uint32_t>(vertex_layout.bindings.size()),
        .pVertexBindingDescriptions = vertex_layout.bindings.data(),
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(vertex_layout.attributes.size()),
        .pVertexAttributeDescriptions = vertex_layout.attributes.data(),
    };
    const vk::PipelineInputAssemblyStateCreateInfo input_assembly{
        .topology = vk::PrimitiveTopology::eTriangleList,
    };
    const vk::PipelineViewportStateCreateInfo viewport_state{.viewportCount = 1, .scissorCount = 1};
    const vk::PipelineRasterizationStateCreateInfo rasterization{
        .polygonMode = vk::PolygonMode::eFill,
        .frontFace = vk::FrontFace::eCounterClockwise,
        .lineWidth = 1.0f,
    };
    const vk::PipelineMultisampleStateCreateInfo multisample{
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
    };
    const std::array<vk::PipelineColorBlendAttachmentState, 1> blend_attachments{{
        vk::PipelineColorBlendAttachmentState{
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA},
    }};
    const vk::PipelineColorBlendStateCreateInfo color_blend{
        .attachmentCount = 1,
        .pAttachments = blend_attachments.data(),
    };
    const std::array<vk::DynamicState, 2> dynamic_states{vk::DynamicState::eViewport,
                                                         vk::DynamicState::eScissor};
    const vk::PipelineDynamicStateCreateInfo dynamic_state{
        .dynamicStateCount = static_cast<uint32_t>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data(),
    };
    const std::array<vk::Format, 1> color_formats{vk::Format::eB8G8R8A8Unorm};
    const vk::PipelineRenderingCreateInfo rendering{
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = color_formats.data(),
    };
    const vk::GraphicsPipelineCreateInfo raw{
        .pNext = &rendering,
        .stageCount = static_cast<uint32_t>(stages.size()),
        .pStages = stages.data(),
        .pVertexInputState = &vertex_input,
        .pInputAssemblyState = &input_assembly,
        .pViewportState = &viewport_state,
        .pRasterizationState = &rasterization,
        .pMultisampleState = &multisample,
        .pColorBlendState = &color_blend,
        .pDynamicState = &dynamic_state,
    };

    const vulcao::Pipeline raw_pipeline =
        vulcao::Pipeline::create_graphics(context.device(), layout, raw);
    CHECK(raw_pipeline.valid());

    CHECK_THROWS_AS(
        vulcao::Pipeline::create_graphics(context.device(), vulcao::PipelineLayout{}, raw),
        std::runtime_error);

    for (const std::string& error : capture.errors)
        MESSAGE("logged error: ", error);
    CHECK(capture.errors.empty());
}

#endif
