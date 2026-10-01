#pragma once

#include <span>

#include "vulcao/reflection.h"

namespace vulcao::detail {

/// @brief Merges the reflection data of several shader stages.
///
/// Internal helper behind PipelineLayout::create_from_reflection.
/// @param reflections Reflection data to merge.
/// @return Merged descriptor sets and push constant ranges.
PipelineReflection merge_reflections(std::span<const ShaderReflection> reflections);

}
