#include <array>

#include <doctest/doctest.h>

#include <vulcao/detail/reflection.h>
#include <vulcao/reflection.h>

namespace {

vulcao::ShaderReflection make_stage(vk::ShaderStageFlagBits stage,
                                    uint32_t binding,
                                    uint32_t offset,
                                    uint32_t size) {
    vulcao::ShaderReflection reflection;
    reflection.stage = stage;
    reflection.sets.push_back(vulcao::DescriptorSetLayoutInfo{
        .set = 0,
        .bindings = {vk::DescriptorSetLayoutBinding{
            .binding = binding,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1,
            .stageFlags = stage,
        }},
    });
    reflection.push_constants.push_back(vk::PushConstantRange{
        .stageFlags = stage,
        .offset = offset,
        .size = size,
    });
    return reflection;
}

}

TEST_CASE("bindings_for_set returns bindings of a known set") {
    const vulcao::ShaderReflection reflection =
        make_stage(vk::ShaderStageFlagBits::eVertex, 3, 0, 16);

    CHECK(reflection.bindings_for_set(0).size() == 1);
    CHECK(reflection.bindings_for_set(0).front().binding == 3);
    CHECK(reflection.bindings_for_set(1).empty());
}

TEST_CASE("merge_reflections merges binding and push constant stage flags") {
    const std::array<vulcao::ShaderReflection, 2> stages{
        make_stage(vk::ShaderStageFlagBits::eVertex, 0, 0, 64),
        make_stage(vk::ShaderStageFlagBits::eFragment, 0, 0, 64),
    };

    const vulcao::PipelineReflection merged = vulcao::detail::merge_reflections(stages);

    REQUIRE(merged.sets.size() == 1);
    REQUIRE(merged.sets.front().bindings.size() == 1);
    CHECK(merged.sets.front().bindings.front().binding == 0);
    CHECK(merged.sets.front().bindings.front().stageFlags ==
          (vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment));

    REQUIRE(merged.push_constants.size() == 1);
    CHECK(merged.push_constants.front().stageFlags ==
          (vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment));
}

TEST_CASE("merge_reflections keeps distinct bindings and ranges") {
    vulcao::ShaderReflection vertex = make_stage(vk::ShaderStageFlagBits::eVertex, 0, 0, 64);
    vulcao::ShaderReflection fragment = make_stage(vk::ShaderStageFlagBits::eFragment, 1, 64, 16);

    const std::array<vulcao::ShaderReflection, 2> stages{vertex, fragment};
    const vulcao::PipelineReflection merged = vulcao::detail::merge_reflections(stages);

    REQUIRE(merged.sets.size() == 1);
    REQUIRE(merged.sets.front().bindings.size() == 2);
    CHECK(merged.sets.front().bindings[0].binding == 0);
    CHECK(merged.sets.front().bindings[1].binding == 1);
    CHECK(merged.push_constants.size() == 2);
}

TEST_CASE("merge_reflections folds overlapping push constant ranges into their union") {
    // The vertex stage reads the first 64 bytes, the fragment stage a window
    // that overlaps it: overlapping ranges are illegal in a pipeline layout.
    const std::array<vulcao::ShaderReflection, 2> stages{
        make_stage(vk::ShaderStageFlagBits::eVertex, 0, 0, 64),
        make_stage(vk::ShaderStageFlagBits::eFragment, 1, 32, 128),
    };

    const vulcao::PipelineReflection merged = vulcao::detail::merge_reflections(stages);

    REQUIRE(merged.push_constants.size() == 1);
    CHECK(merged.push_constants.front().offset == 0);
    CHECK(merged.push_constants.front().size == 160);
    CHECK(merged.push_constants.front().stageFlags ==
          (vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment));
}

TEST_CASE("merge_reflections folds chains of overlapping push constant ranges") {
    // [0,100) + [90,150) + [140,300) overlap transitively and must collapse
    // into a single [0,300) range.
    const std::array<vulcao::ShaderReflection, 3> stages{
        make_stage(vk::ShaderStageFlagBits::eVertex, 0, 0, 100),
        make_stage(vk::ShaderStageFlagBits::eFragment, 1, 90, 60),
        make_stage(vk::ShaderStageFlagBits::eCompute, 2, 140, 160),
    };

    const vulcao::PipelineReflection merged = vulcao::detail::merge_reflections(stages);

    REQUIRE(merged.push_constants.size() == 1);
    CHECK(merged.push_constants.front().offset == 0);
    CHECK(merged.push_constants.front().size == 300);
    CHECK(merged.push_constants.front().stageFlags ==
          (vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment |
           vk::ShaderStageFlagBits::eCompute));
}

TEST_CASE("merge_reflections keeps adjacent push constant ranges separate") {
    // [0,64) and [64,16) touch but do not overlap; both are legal and stay.
    const std::array<vulcao::ShaderReflection, 2> stages{
        make_stage(vk::ShaderStageFlagBits::eVertex, 0, 64, 16),
        make_stage(vk::ShaderStageFlagBits::eFragment, 1, 0, 64),
    };

    const vulcao::PipelineReflection merged = vulcao::detail::merge_reflections(stages);

    // Sorted by offset.
    REQUIRE(merged.push_constants.size() == 2);
    CHECK(merged.push_constants[0].offset == 0);
    CHECK(merged.push_constants[0].size == 64);
    CHECK(merged.push_constants[1].offset == 64);
    CHECK(merged.push_constants[1].size == 16);
}

TEST_CASE("set_binding_count gives a reflected runtime array a concrete bound") {
    // Runtime arrays reflect with descriptorCount 0, which no layout accepts.
    vulcao::DescriptorSetLayoutInfo set{
        .set = 0,
        .bindings = {vk::DescriptorSetLayoutBinding{
                         .binding = 0,
                         .descriptorType = vk::DescriptorType::eSampledImage,
                         .descriptorCount = 0,
                         .stageFlags = vk::ShaderStageFlagBits::eFragment,
                     },
                     vk::DescriptorSetLayoutBinding{
                         .binding = 3,
                         .descriptorType = vk::DescriptorType::eUniformBuffer,
                         .descriptorCount = 1,
                         .stageFlags = vk::ShaderStageFlagBits::eFragment,
                     }},
    };

    CHECK(vulcao::set_binding_count(set, 0, 1024));
    CHECK(set.bindings[0].descriptorCount == 1024);
    CHECK(set.bindings[1].descriptorCount == 1);

    CHECK_FALSE(vulcao::set_binding_count(set, 7, 4));
    CHECK_THROWS_AS(vulcao::set_binding_count(set, 0, 0), std::runtime_error);
}

TEST_CASE("merge_reflections sorts sets and bindings") {
    vulcao::ShaderReflection reflection;
    reflection.stage = vk::ShaderStageFlagBits::eFragment;
    reflection.sets.push_back(vulcao::DescriptorSetLayoutInfo{
        .set = 1,
        .bindings = {vk::DescriptorSetLayoutBinding{.binding = 2}},
    });
    reflection.sets.push_back(vulcao::DescriptorSetLayoutInfo{
        .set = 0,
        .bindings = {vk::DescriptorSetLayoutBinding{.binding = 5},
                     vk::DescriptorSetLayoutBinding{.binding = 1}},
    });

    const std::array<vulcao::ShaderReflection, 1> stages{reflection};
    const vulcao::PipelineReflection merged = vulcao::detail::merge_reflections(stages);

    REQUIRE(merged.sets.size() == 2);
    CHECK(merged.sets[0].set == 0);
    CHECK(merged.sets[1].set == 1);
    REQUIRE(merged.sets[0].bindings.size() == 2);
    CHECK(merged.sets[0].bindings[0].binding == 1);
    CHECK(merged.sets[0].bindings[1].binding == 5);
}
