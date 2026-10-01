#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include <vulkan/vulkan.hpp>

namespace vulcao {

/// @brief A descriptor set and its bindings.
struct DescriptorSetLayoutInfo {
    uint32_t set = 0;                                  ///< Descriptor set index.
    std::vector<vk::DescriptorSetLayoutBinding> bindings; ///< Bindings of the set.
};

/// @brief A specialization constant declared by a shader.
struct SpecializationConstantInfo {
    uint32_t constant_id = 0;   ///< Constant id declared in the shader.
    uint32_t default_value = 0; ///< Default value declared in the shader.
};

/// @brief Reflection data of a single shader stage.
struct ShaderReflection {
    vk::ShaderStageFlags stage;                                       ///< Stage the module was reflected for.
    std::vector<DescriptorSetLayoutInfo> sets;                        ///< Descriptor sets used by the stage.
    std::vector<vk::PushConstantRange> push_constants;                ///< Push constant ranges of the stage.
    std::vector<vk::VertexInputAttributeDescription> vertex_attributes; ///< Vertex inputs, for vertex stages.
    std::vector<SpecializationConstantInfo> specialization_constants; ///< Specialization constants.

    /// @brief Returns the bindings of a descriptor set, or an empty range if the set is unused.
    /// @param set Descriptor set index.
    /// @return Bindings of the set.
    const std::vector<vk::DescriptorSetLayoutBinding>& bindings_for_set(uint32_t set) const;
};

/// @brief Merged reflection data of several shader stages.
struct PipelineReflection {
    std::vector<DescriptorSetLayoutInfo> sets;   ///< Merged descriptor sets.
    std::vector<vk::PushConstantRange> push_constants; ///< Merged push constant ranges.
};

/// @brief Reflects a SPIR-V module.
/// @param spirv SPIR-V code.
/// @return Reflection data of the module.
/// @throws std::runtime_error if the SPIR-V cannot be reflected.
ShaderReflection reflect_spirv(std::span<const uint32_t> spirv);

/// @brief Sets the descriptor count of one binding in a reflected set.
///
/// Reflected runtime arrays (unsized descriptor arrays, the building block of
/// bindless) report descriptorCount 0, which vk::DescriptorSetLayoutCreateInfo
/// rejects. Use this to give such a binding a concrete upper bound before
/// creating a layout, typically together with the ePartiallyBound or
/// eVariableDescriptorCount binding flags.
/// @param set Reflected set to modify.
/// @param binding Binding number to change.
/// @param count New descriptor count, must be non-zero.
/// @return True if the binding exists in the set.
bool set_binding_count(DescriptorSetLayoutInfo& set, uint32_t binding, uint32_t count);

}
