#include "vulcao/pipeline.h"

#include "vulcao/detail/check.h"
#include "vulcao/pipeline_cache.h"
#include "vulcao/pipeline_layout.h"
#include "vulcao/shader_module.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vulcao {

Pipeline::~Pipeline() {
    destroy();
}

Pipeline::Pipeline(Pipeline&& other) noexcept
    : device_(std::exchange(other.device_, vk::Device{})),
      pipeline_(std::exchange(other.pipeline_, vk::Pipeline{})),
      bind_point_(std::exchange(other.bind_point_, vk::PipelineBindPoint::eGraphics)) {}

Pipeline& Pipeline::operator=(Pipeline&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = std::exchange(other.device_, vk::Device{});
        pipeline_ = std::exchange(other.pipeline_, vk::Pipeline{});
        bind_point_ = std::exchange(other.bind_point_, vk::PipelineBindPoint::eGraphics);
    }
    return *this;
}

Pipeline Pipeline::create_graphics(vk::Device device,
                                   const PipelineLayout& layout,
                                   const GraphicsPipelineInfo& info,
                                   const PipelineCache* cache) {
    return create_graphics_impl(device, cache ? cache->handle() : vk::PipelineCache{}, layout, info);
}

Pipeline Pipeline::create_graphics_impl(vk::Device device,
                                        vk::PipelineCache cache,
                                        const PipelineLayout& layout,
                                        const GraphicsPipelineInfo& info) {
    if (!layout.valid())
        throw std::runtime_error("Pipeline::create_graphics: invalid pipeline layout");
    if (!info.vertex_shader || !info.fragment_shader)
        throw std::runtime_error("Pipeline::create_graphics: missing shader module");

    const vk::SpecializationInfo vertex_specialization =
        info.vertex_specialization ? info.vertex_specialization->get() : vk::SpecializationInfo{};
    const vk::SpecializationInfo fragment_specialization =
        info.fragment_specialization ? info.fragment_specialization->get() : vk::SpecializationInfo{};
    const vk::SpecializationInfo tessellation_control_specialization =
        info.tessellation_control_specialization
            ? info.tessellation_control_specialization->get()
            : vk::SpecializationInfo{};
    const vk::SpecializationInfo tessellation_evaluation_specialization =
        info.tessellation_evaluation_specialization
            ? info.tessellation_evaluation_specialization->get()
            : vk::SpecializationInfo{};
    const vk::SpecializationInfo geometry_specialization =
        info.geometry_specialization ? info.geometry_specialization->get()
                                     : vk::SpecializationInfo{};

    std::vector<vk::PipelineShaderStageCreateInfo> stages;
    stages.push_back(vk::PipelineShaderStageCreateInfo{
        .stage = vk::ShaderStageFlagBits::eVertex,
        .module = info.vertex_shader,
        .pName = info.vertex_entry,
        .pSpecializationInfo = info.vertex_specialization ? &vertex_specialization : nullptr,
    });
    if (info.tessellation_control_shader)
        stages.push_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eTessellationControl,
            .module = info.tessellation_control_shader,
            .pName = info.tessellation_control_entry,
            .pSpecializationInfo = info.tessellation_control_specialization
                                        ? &tessellation_control_specialization
                                        : nullptr,
        });
    if (info.tessellation_evaluation_shader)
        stages.push_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eTessellationEvaluation,
            .module = info.tessellation_evaluation_shader,
            .pName = info.tessellation_evaluation_entry,
            .pSpecializationInfo = info.tessellation_evaluation_specialization
                                        ? &tessellation_evaluation_specialization
                                        : nullptr,
        });
    if (info.geometry_shader)
        stages.push_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eGeometry,
            .module = info.geometry_shader,
            .pName = info.geometry_entry,
            .pSpecializationInfo = info.geometry_specialization ? &geometry_specialization : nullptr,
        });
    stages.push_back(vk::PipelineShaderStageCreateInfo{
        .stage = vk::ShaderStageFlagBits::eFragment,
        .module = info.fragment_shader,
        .pName = info.fragment_entry,
        .pSpecializationInfo = info.fragment_specialization ? &fragment_specialization : nullptr,
    });

    const bool has_tessellation =
        info.tessellation_control_shader || info.tessellation_evaluation_shader;

    const vk::PipelineVertexInputStateCreateInfo vertex_input{
        .vertexBindingDescriptionCount = static_cast<uint32_t>(info.vertex_bindings.size()),
        .pVertexBindingDescriptions = info.vertex_bindings.data(),
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(info.vertex_attributes.size()),
        .pVertexAttributeDescriptions = info.vertex_attributes.data(),
    };

    const vk::PipelineInputAssemblyStateCreateInfo input_assembly{
        .topology = info.topology,
        .primitiveRestartEnable = info.primitive_restart ? VK_TRUE : VK_FALSE,
    };

    const vk::PipelineTessellationStateCreateInfo tessellation{
        .patchControlPoints = info.patch_control_points,
    };

    const std::vector<vk::DynamicState> dynamic_states =
        info.dynamic_states.empty()
            ? std::vector<vk::DynamicState>{vk::DynamicState::eViewportWithCount,
                                            vk::DynamicState::eScissorWithCount}
            : info.dynamic_states;
    const auto has_dynamic_state = [&](vk::DynamicState state) {
        return std::find(dynamic_states.begin(), dynamic_states.end(), state) != dynamic_states.end();
    };

    // The counted dynamic states take over the viewport and scissor, so the
    // static counts must be zero for them.
    const vk::PipelineViewportStateCreateInfo viewport_state{
        .viewportCount =
            has_dynamic_state(vk::DynamicState::eViewportWithCount) ? 0u : info.viewport_count,
        .scissorCount =
            has_dynamic_state(vk::DynamicState::eScissorWithCount) ? 0u : info.viewport_count,
    };

    const vk::PipelineRasterizationStateCreateInfo rasterization{
        .depthClampEnable = info.depth_clamp_enable ? VK_TRUE : VK_FALSE,
        .rasterizerDiscardEnable = info.rasterizer_discard ? VK_TRUE : VK_FALSE,
        .polygonMode = info.polygon_mode,
        .cullMode = info.cull_mode,
        .frontFace = info.front_face,
        .depthBiasEnable = info.depth_bias_enable ? VK_TRUE : VK_FALSE,
        .lineWidth = info.line_width,
    };

    const vk::PipelineMultisampleStateCreateInfo multisample{
        .rasterizationSamples = info.samples,
        .sampleShadingEnable = info.sample_shading ? VK_TRUE : VK_FALSE,
        .minSampleShading = info.min_sample_shading,
        .pSampleMask = info.sample_mask != 0 ? &info.sample_mask : nullptr,
        .alphaToCoverageEnable = info.alpha_to_coverage ? VK_TRUE : VK_FALSE,
    };

    const bool depth_enabled = info.depth_test && info.depth_format != vk::Format::eUndefined;
    const vk::PipelineDepthStencilStateCreateInfo depth_stencil{
        .depthTestEnable = depth_enabled ? VK_TRUE : VK_FALSE,
        .depthWriteEnable = depth_enabled && info.depth_write ? VK_TRUE : VK_FALSE,
        .depthCompareOp = info.depth_compare,
        .depthBoundsTestEnable = info.depth_bounds_test ? VK_TRUE : VK_FALSE,
        .stencilTestEnable = info.stencil_test ? VK_TRUE : VK_FALSE,
        .front = info.front_stencil,
        .back = info.back_stencil,
        .minDepthBounds = info.min_depth_bounds,
        .maxDepthBounds = info.max_depth_bounds,
    };

    std::vector<vk::PipelineColorBlendAttachmentState> blend_attachments;
    if (!info.color_blend_attachments.empty()) {
        blend_attachments = info.color_blend_attachments;
    } else {
        blend_attachments.reserve(info.color_formats.size());
        for (size_t i = 0; i < info.color_formats.size(); ++i) {
            blend_attachments.push_back(vk::PipelineColorBlendAttachmentState{
                .blendEnable = info.blend ? VK_TRUE : VK_FALSE,
                .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
                .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
                .colorBlendOp = vk::BlendOp::eAdd,
                .srcAlphaBlendFactor = vk::BlendFactor::eOne,
                .dstAlphaBlendFactor = vk::BlendFactor::eZero,
                .alphaBlendOp = vk::BlendOp::eAdd,
                .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                                  vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
            });
        }
    }

    const vk::PipelineColorBlendStateCreateInfo color_blend{
        .logicOpEnable = info.logic_op_enable ? VK_TRUE : VK_FALSE,
        .logicOp = info.logic_op,
        .attachmentCount = static_cast<uint32_t>(blend_attachments.size()),
        .pAttachments = blend_attachments.data(),
        .blendConstants = info.blend_constants,
    };

    const vk::PipelineDynamicStateCreateInfo dynamic_state{
        .dynamicStateCount = static_cast<uint32_t>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data(),
    };

    const vk::PipelineRenderingCreateInfo rendering{
        .colorAttachmentCount = static_cast<uint32_t>(info.color_formats.size()),
        .pColorAttachmentFormats = info.color_formats.data(),
        .depthAttachmentFormat = info.depth_format,
    };

    const vk::GraphicsPipelineCreateInfo create_info{
        .pNext = &rendering,
        .stageCount = static_cast<uint32_t>(stages.size()),
        .pStages = stages.data(),
        .pVertexInputState = &vertex_input,
        .pInputAssemblyState = &input_assembly,
        .pTessellationState = has_tessellation ? &tessellation : nullptr,
        .pViewportState = &viewport_state,
        .pRasterizationState = &rasterization,
        .pMultisampleState = &multisample,
        .pDepthStencilState = &depth_stencil,
        .pColorBlendState = &color_blend,
        .pDynamicState = &dynamic_state,
        .layout = layout.handle(),
    };

    Pipeline pipeline;
    pipeline.device_ = device;
    pipeline.bind_point_ = vk::PipelineBindPoint::eGraphics;
    const vk::ResultValue<vk::Pipeline> result =
        device.createGraphicsPipeline(cache, create_info);
    detail::check(result.result, "create graphics pipeline");
    pipeline.pipeline_ = result.value;
    return pipeline;
}

Pipeline Pipeline::create_graphics(vk::Device device,
                                   const PipelineLayout& layout,
                                   const vk::GraphicsPipelineCreateInfo& create_info,
                                   const PipelineCache* cache) {
    return create_graphics_raw(device, cache ? cache->handle() : vk::PipelineCache{}, layout,
                               create_info);
}

Pipeline Pipeline::create_graphics_raw(vk::Device device,
                                       vk::PipelineCache cache,
                                       const PipelineLayout& layout,
                                       const vk::GraphicsPipelineCreateInfo& create_info) {
    if (!layout.valid())
        throw std::runtime_error("Pipeline::create_graphics: invalid pipeline layout");

    vk::GraphicsPipelineCreateInfo info = create_info;
    info.layout = layout.handle();

    Pipeline pipeline;
    pipeline.device_ = device;
    pipeline.bind_point_ = vk::PipelineBindPoint::eGraphics;
    const vk::ResultValue<vk::Pipeline> result = device.createGraphicsPipeline(cache, info);
    detail::check(result.result, "create graphics pipeline");
    pipeline.pipeline_ = result.value;
    return pipeline;
}

Pipeline Pipeline::create_compute(vk::Device device,
                                  const PipelineLayout& layout,
                                  const ShaderModule& shader,
                                  const char* entry,
                                  const PipelineCache* cache,
                                  const SpecializationInfo* specialization) {
    return create_compute_impl(device, cache ? cache->handle() : vk::PipelineCache{}, layout, shader,
                               entry, specialization);
}

Pipeline Pipeline::create_compute_impl(vk::Device device,
                                       vk::PipelineCache cache,
                                       const PipelineLayout& layout,
                                       const ShaderModule& shader,
                                       const char* entry,
                                       const SpecializationInfo* specialization) {
    if (!layout.valid())
        throw std::runtime_error("Pipeline::create_compute: invalid pipeline layout");
    if (!shader.valid())
        throw std::runtime_error("Pipeline::create_compute: invalid shader module");

    const vk::SpecializationInfo spec_info =
        specialization ? specialization->get() : vk::SpecializationInfo{};

    const vk::ComputePipelineCreateInfo create_info{
        .stage = vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eCompute,
            .module = shader.handle(),
            .pName = entry,
            .pSpecializationInfo = specialization ? &spec_info : nullptr,
        },
        .layout = layout.handle(),
    };

    Pipeline pipeline;
    pipeline.device_ = device;
    pipeline.bind_point_ = vk::PipelineBindPoint::eCompute;
    const vk::ResultValue<vk::Pipeline> result = device.createComputePipeline(cache, create_info);
    detail::check(result.result, "create compute pipeline");
    pipeline.pipeline_ = result.value;
    return pipeline;
}

void Pipeline::destroy() {
    if (pipeline_)
        device_.destroyPipeline(pipeline_);

    device_ = nullptr;
    pipeline_ = nullptr;
    bind_point_ = vk::PipelineBindPoint::eGraphics;
}

}
