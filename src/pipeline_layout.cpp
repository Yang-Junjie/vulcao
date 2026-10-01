#include "vulcao/pipeline_layout.h"

#include "vulcao/detail/reflection.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace vulcao {
namespace {

/// @brief Returns one bindings vector per set number, with empty vectors for
///        the sets the shaders do not use.
///
/// vk::PipelineLayoutCreateInfo::pSetLayouts is indexed by set number, so a
/// pipeline whose shaders skip a set still needs a (empty) layout at that
/// position.
std::vector<const std::vector<vk::DescriptorSetLayoutBinding>*> bindings_by_set(
    const PipelineReflection& merged) {
    uint32_t max_set = 0;
    for (const DescriptorSetLayoutInfo& set : merged.sets)
        max_set = std::max(max_set, set.set);

    static const std::vector<vk::DescriptorSetLayoutBinding> empty;
    std::vector<const std::vector<vk::DescriptorSetLayoutBinding>*> result;
    if (merged.sets.empty())
        return result;

    result.assign(max_set + 1, &empty);
    for (const DescriptorSetLayoutInfo& set : merged.sets)
        result[set.set] = &set.bindings;
    return result;
}

}

PipelineLayout::~PipelineLayout() {
    destroy();
}

PipelineLayout::PipelineLayout(PipelineLayout&& other) noexcept
    : device_(std::exchange(other.device_, vk::Device{})),
      layout_(std::exchange(other.layout_, vk::PipelineLayout{})),
      owned_set_layouts_(std::move(other.owned_set_layouts_)),
      set_layout_handles_(std::move(other.set_layout_handles_)) {}

PipelineLayout& PipelineLayout::operator=(PipelineLayout&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = std::exchange(other.device_, vk::Device{});
        layout_ = std::exchange(other.layout_, vk::PipelineLayout{});
        owned_set_layouts_ = std::move(other.owned_set_layouts_);
        set_layout_handles_ = std::move(other.set_layout_handles_);
    }
    return *this;
}

PipelineLayout PipelineLayout::create(vk::Device device,
                                      vk::ArrayProxy<const vk::DescriptorSetLayout> set_layouts,
                                      vk::ArrayProxy<const vk::PushConstantRange> push_constants) {
    PipelineLayout layout;
    layout.device_ = device;
    layout.layout_ = device.createPipelineLayout(vk::PipelineLayoutCreateInfo{
        .setLayoutCount = static_cast<uint32_t>(set_layouts.size()),
        .pSetLayouts = set_layouts.data(),
        .pushConstantRangeCount = static_cast<uint32_t>(push_constants.size()),
        .pPushConstantRanges = push_constants.data(),
    });
    layout.set_layout_handles_.assign(set_layouts.begin(), set_layouts.end());
    return layout;
}

PipelineLayout PipelineLayout::create_from_reflection(
    vk::Device device,
    std::span<const ShaderReflection> reflections,
    DescriptorSetLayoutCache* cache) {
    const PipelineReflection merged = detail::merge_reflections(reflections);
    const std::vector<const std::vector<vk::DescriptorSetLayoutBinding>*> by_set =
        bindings_by_set(merged);

    std::vector<DescriptorSetLayout> set_layouts;
    std::vector<vk::DescriptorSetLayout> raw_layouts;
    raw_layouts.reserve(by_set.size());

    if (cache != nullptr) {
        for (const std::vector<vk::DescriptorSetLayoutBinding>* bindings : by_set)
            raw_layouts.push_back(cache->get(*bindings));
        return create(device, raw_layouts, merged.push_constants);
    }

    set_layouts.reserve(by_set.size());
    for (const std::vector<vk::DescriptorSetLayoutBinding>* bindings : by_set)
        set_layouts.push_back(DescriptorSetLayout::create(device, *bindings));

    for (const DescriptorSetLayout& layout : set_layouts)
        raw_layouts.push_back(layout.handle());

    PipelineLayout layout = create(device, raw_layouts, merged.push_constants);
    layout.owned_set_layouts_ = std::move(set_layouts);
    return layout;
}

void PipelineLayout::destroy() {
    if (layout_)
        device_.destroyPipelineLayout(layout_);

    owned_set_layouts_.clear();
    set_layout_handles_.clear();
    device_ = nullptr;
    layout_ = nullptr;
}

}
