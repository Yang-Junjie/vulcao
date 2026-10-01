#pragma once

#include <vulkan/vulkan.hpp>

namespace vulcao::detail {

/// @brief Returns the pipeline stages that access an image in a layout.
/// @param layout Image layout.
/// @return Pipeline stages that read or write the layout.
vk::PipelineStageFlags2 stage_for_layout(vk::ImageLayout layout);

/// @brief Returns the access flags that match an image layout.
/// @param layout Image layout.
/// @return Access flags of the layout.
vk::AccessFlags2 access_for_layout(vk::ImageLayout layout);

}
