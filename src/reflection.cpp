#include "vulcao/reflection.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#include <spirv_reflect.h>

namespace vulcao {
namespace {

const char* result_name(SpvReflectResult result) {
    switch (result) {
        case SPV_REFLECT_RESULT_SUCCESS: return "success";
        case SPV_REFLECT_RESULT_NOT_READY: return "not ready";
        case SPV_REFLECT_RESULT_ERROR_PARSE_FAILED: return "parse failed";
        case SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED: return "allocation failed";
        case SPV_REFLECT_RESULT_ERROR_RANGE_EXCEEDED: return "range exceeded";
        case SPV_REFLECT_RESULT_ERROR_NULL_POINTER: return "null pointer";
        case SPV_REFLECT_RESULT_ERROR_INTERNAL_ERROR: return "internal error";
        case SPV_REFLECT_RESULT_ERROR_COUNT_MISMATCH: return "count mismatch";
        case SPV_REFLECT_RESULT_ERROR_ELEMENT_NOT_FOUND: return "element not found";
        case SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_CODE_SIZE: return "invalid code size";
        case SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_MAGIC_NUMBER: return "invalid magic number";
        case SPV_REFLECT_RESULT_ERROR_SPIRV_UNEXPECTED_EOF: return "unexpected end of file";
        case SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE: return "invalid id reference";
        case SPV_REFLECT_RESULT_ERROR_SPIRV_SET_NUMBER_OVERFLOW: return "set number overflow";
        case SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_STORAGE_CLASS: return "invalid storage class";
        case SPV_REFLECT_RESULT_ERROR_SPIRV_RECURSION: return "recursion";
        case SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_INSTRUCTION: return "invalid instruction";
        case SPV_REFLECT_RESULT_ERROR_SPIRV_UNEXPECTED_BLOCK_DATA: return "unexpected block data";
        case SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_BLOCK_MEMBER_REFERENCE: return "invalid block member reference";
        case SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ENTRY_POINT: return "invalid entry point";
        case SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_EXECUTION_MODE: return "invalid execution mode";
        case SPV_REFLECT_RESULT_ERROR_SPIRV_MAX_RECURSIVE_EXCEEDED: return "max recursive exceeded";
    }
    return "unknown error";
}

void check_reflect(SpvReflectResult result, const char* what) {
    if (result != SPV_REFLECT_RESULT_SUCCESS)
        throw std::runtime_error(std::string(what) + " failed: " + result_name(result));
}

class ModuleGuard {
public:
    ~ModuleGuard() {
        if (created_)
            spvReflectDestroyShaderModule(&module_);
    }

    SpvReflectShaderModule& get() { return module_; }
    void mark_created() { created_ = true; }

private:
    SpvReflectShaderModule module_{};
    bool created_ = false;
};

template <typename T, typename Fn>
std::vector<T*> enumerate(SpvReflectShaderModule& module, Fn fn, const char* what) {
    uint32_t count = 0;
    check_reflect(fn(&module, &count, nullptr), what);
    if (count == 0)
        return {};

    std::vector<T*> items(count);
    check_reflect(fn(&module, &count, items.data()), what);
    items.resize(count);
    return items;
}

}

const std::vector<vk::DescriptorSetLayoutBinding>& ShaderReflection::bindings_for_set(
    uint32_t set) const {
    static const std::vector<vk::DescriptorSetLayoutBinding> empty;
    for (const DescriptorSetLayoutInfo& info : sets)
        if (info.set == set)
            return info.bindings;
    return empty;
}

ShaderReflection reflect_spirv(std::span<const uint32_t> spirv) {
    if (spirv.empty())
        throw std::runtime_error("reflect_spirv: empty SPIR-V");

    ModuleGuard guard;
    check_reflect(spvReflectCreateShaderModule(spirv.size() * sizeof(uint32_t), spirv.data(),
                                               &guard.get()),
                  "reflect spirv");
    guard.mark_created();

    SpvReflectShaderModule& module = guard.get();

    ShaderReflection reflection;
    reflection.stage = static_cast<vk::ShaderStageFlags>(module.shader_stage);

    for (SpvReflectDescriptorSet* set :
         enumerate<SpvReflectDescriptorSet>(module, spvReflectEnumerateDescriptorSets,
                                            "enumerate descriptor sets")) {
        DescriptorSetLayoutInfo info;
        info.set = set->set;
        info.bindings.reserve(set->binding_count);
        for (uint32_t i = 0; i < set->binding_count; ++i) {
            const SpvReflectDescriptorBinding* binding = set->bindings[i];
            info.bindings.push_back(vk::DescriptorSetLayoutBinding{
                .binding = binding->binding,
                .descriptorType = static_cast<vk::DescriptorType>(binding->descriptor_type),
                .descriptorCount = binding->count,
                .stageFlags = reflection.stage,
            });
        }
        reflection.sets.push_back(std::move(info));
    }

    for (SpvReflectBlockVariable* block :
         enumerate<SpvReflectBlockVariable>(module, spvReflectEnumeratePushConstantBlocks,
                                            "enumerate push constant blocks")) {
        reflection.push_constants.push_back(vk::PushConstantRange{
            .stageFlags = reflection.stage,
            .offset = block->offset,
            .size = block->size,
        });
    }

    if (reflection.stage & vk::ShaderStageFlagBits::eVertex) {
        for (SpvReflectInterfaceVariable* input :
             enumerate<SpvReflectInterfaceVariable>(module, spvReflectEnumerateInputVariables,
                                                    "enumerate input variables")) {
            if (input->decoration_flags & SPV_REFLECT_DECORATION_BUILT_IN)
                continue;

            reflection.vertex_attributes.push_back(vk::VertexInputAttributeDescription{
                .location = input->location,
                .binding = 0,
                .format = static_cast<vk::Format>(input->format),
                .offset = 0,
            });
        }
    }

    for (SpvReflectSpecializationConstant* constant : enumerate<SpvReflectSpecializationConstant>(
             module, spvReflectEnumerateSpecializationConstants,
             "enumerate specialization constants")) {
        uint32_t default_value = 0;
        if (constant->default_value != nullptr && constant->default_value_size >= sizeof(uint32_t))
            std::memcpy(&default_value, constant->default_value, sizeof(uint32_t));

        reflection.specialization_constants.push_back(SpecializationConstantInfo{
            .constant_id = constant->constant_id,
            .default_value = default_value,
        });
    }

    return reflection;
}

namespace detail {

PipelineReflection merge_reflections(std::span<const ShaderReflection> reflections) {
    PipelineReflection merged;

    for (const ShaderReflection& reflection : reflections) {
        for (const DescriptorSetLayoutInfo& set : reflection.sets) {
            auto set_it = std::find_if(merged.sets.begin(), merged.sets.end(),
                                       [&](const DescriptorSetLayoutInfo& s) { return s.set == set.set; });
            if (set_it == merged.sets.end()) {
                merged.sets.push_back(set);
                continue;
            }

            for (const vk::DescriptorSetLayoutBinding& binding : set.bindings) {
                auto binding_it = std::find_if(
                    set_it->bindings.begin(), set_it->bindings.end(),
                    [&](const vk::DescriptorSetLayoutBinding& b) { return b.binding == binding.binding; });
                if (binding_it == set_it->bindings.end())
                    set_it->bindings.push_back(binding);
                else
                    binding_it->stageFlags |= binding.stageFlags;
            }
        }

        for (const vk::PushConstantRange& range : reflection.push_constants) {
            // Stages may declare the same push constant block with different
            // ranges (e.g. the vertex stage reads the first 64 bytes, the
            // fragment stage all 128). Overlapping ranges are illegal in a
            // pipeline layout, so fold them into the union with OR'd stages.
            vk::PushConstantRange incoming = range;
            for (auto it = merged.push_constants.begin(); it != merged.push_constants.end();) {
                const uint32_t existing_end = it->offset + it->size;
                const uint32_t incoming_end = incoming.offset + incoming.size;
                if (it->offset < incoming_end && incoming.offset < existing_end) {
                    incoming.offset = std::min(it->offset, incoming.offset);
                    incoming.size = std::max(existing_end, incoming_end) - incoming.offset;
                    incoming.stageFlags |= it->stageFlags;
                    it = merged.push_constants.erase(it);
                } else {
                    ++it;
                }
            }
            merged.push_constants.push_back(incoming);
        }
    }

    std::sort(merged.sets.begin(), merged.sets.end(),
              [](const DescriptorSetLayoutInfo& a, const DescriptorSetLayoutInfo& b) { return a.set < b.set; });
    for (DescriptorSetLayoutInfo& set : merged.sets)
        std::sort(set.bindings.begin(), set.bindings.end(),
                  [](const vk::DescriptorSetLayoutBinding& a, const vk::DescriptorSetLayoutBinding& b) {
                      return a.binding < b.binding;
                  });
    std::sort(merged.push_constants.begin(), merged.push_constants.end(),
              [](const vk::PushConstantRange& a, const vk::PushConstantRange& b) {
                  return a.offset < b.offset;
              });

    return merged;
}

}

bool set_binding_count(DescriptorSetLayoutInfo& set, uint32_t binding, uint32_t count) {
    if (count == 0)
        throw std::runtime_error("set_binding_count: count must be non-zero");

    for (vk::DescriptorSetLayoutBinding& b : set.bindings) {
        if (b.binding == binding) {
            b.descriptorCount = count;
            return true;
        }
    }
    return false;
}

}
