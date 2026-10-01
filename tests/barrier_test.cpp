#include <doctest/doctest.h>

#include <vulcao/detail/barrier.h>

namespace {

constexpr vk::PipelineStageFlags2 stages(vk::PipelineStageFlagBits2 bits) {
    return vk::PipelineStageFlags2{bits};
}

constexpr vk::AccessFlags2 access(vk::AccessFlagBits2 bits) {
    return vk::AccessFlags2{bits};
}

}

TEST_CASE("stage_for_layout maps common layouts") {
    CHECK(vulcao::detail::stage_for_layout(vk::ImageLayout::eUndefined) ==
          stages(vk::PipelineStageFlagBits2::eTopOfPipe));
    CHECK(vulcao::detail::stage_for_layout(vk::ImageLayout::eTransferSrcOptimal) ==
          stages(vk::PipelineStageFlagBits2::eTransfer));
    CHECK(vulcao::detail::stage_for_layout(vk::ImageLayout::eTransferDstOptimal) ==
          stages(vk::PipelineStageFlagBits2::eTransfer));
    CHECK(vulcao::detail::stage_for_layout(vk::ImageLayout::eColorAttachmentOptimal) ==
          stages(vk::PipelineStageFlagBits2::eColorAttachmentOutput));
    CHECK(vulcao::detail::stage_for_layout(vk::ImageLayout::eDepthStencilAttachmentOptimal) ==
          (stages(vk::PipelineStageFlagBits2::eEarlyFragmentTests) |
           stages(vk::PipelineStageFlagBits2::eLateFragmentTests)));
    CHECK(vulcao::detail::stage_for_layout(vk::ImageLayout::eShaderReadOnlyOptimal) ==
          (stages(vk::PipelineStageFlagBits2::eFragmentShader) |
           stages(vk::PipelineStageFlagBits2::eComputeShader)));
    CHECK(vulcao::detail::stage_for_layout(vk::ImageLayout::ePresentSrcKHR) ==
          stages(vk::PipelineStageFlagBits2::eBottomOfPipe));
}

TEST_CASE("stage_for_layout falls back to all commands") {
    CHECK(vulcao::detail::stage_for_layout(vk::ImageLayout::eGeneral) ==
          stages(vk::PipelineStageFlagBits2::eAllCommands));
}

TEST_CASE("access_for_layout maps common layouts") {
    CHECK(vulcao::detail::access_for_layout(vk::ImageLayout::eUndefined) == vk::AccessFlags2{});
    CHECK(vulcao::detail::access_for_layout(vk::ImageLayout::ePresentSrcKHR) == vk::AccessFlags2{});
    CHECK(vulcao::detail::access_for_layout(vk::ImageLayout::eTransferSrcOptimal) ==
          access(vk::AccessFlagBits2::eTransferRead));
    CHECK(vulcao::detail::access_for_layout(vk::ImageLayout::eTransferDstOptimal) ==
          access(vk::AccessFlagBits2::eTransferWrite));
    CHECK(vulcao::detail::access_for_layout(vk::ImageLayout::eColorAttachmentOptimal) ==
          access(vk::AccessFlagBits2::eColorAttachmentWrite));
    CHECK(vulcao::detail::access_for_layout(vk::ImageLayout::eDepthStencilAttachmentOptimal) ==
          access(vk::AccessFlagBits2::eDepthStencilAttachmentWrite));
    CHECK(vulcao::detail::access_for_layout(vk::ImageLayout::eShaderReadOnlyOptimal) ==
          access(vk::AccessFlagBits2::eShaderRead));
}

TEST_CASE("access_for_layout maps read only depth layouts") {
    CHECK(vulcao::detail::access_for_layout(vk::ImageLayout::eDepthStencilReadOnlyOptimal) ==
          (access(vk::AccessFlagBits2::eDepthStencilAttachmentRead) |
           access(vk::AccessFlagBits2::eShaderRead)));
    CHECK(vulcao::detail::access_for_layout(vk::ImageLayout::eDepthReadOnlyOptimal) ==
          access(vk::AccessFlagBits2::eDepthStencilAttachmentRead));
    CHECK(vulcao::detail::access_for_layout(vk::ImageLayout::eStencilReadOnlyOptimal) ==
          access(vk::AccessFlagBits2::eDepthStencilAttachmentRead));
}

TEST_CASE("access_for_layout includes writes for general layouts") {
    // eGeneral admits writes; without the write bit a transition into or out of
    // it would not order write-after-write hazards.
    CHECK(vulcao::detail::access_for_layout(vk::ImageLayout::eGeneral) ==
          (access(vk::AccessFlagBits2::eMemoryRead) |
           access(vk::AccessFlagBits2::eMemoryWrite)));
    CHECK(vulcao::detail::access_for_layout(vk::ImageLayout::eAttachmentOptimal) ==
          (access(vk::AccessFlagBits2::eMemoryRead) |
           access(vk::AccessFlagBits2::eMemoryWrite)));
}

TEST_CASE("stage_for_layout maps read only depth layouts") {
    const vk::PipelineStageFlags2 depth_read_stages =
        stages(vk::PipelineStageFlagBits2::eEarlyFragmentTests) |
        stages(vk::PipelineStageFlagBits2::eLateFragmentTests) |
        stages(vk::PipelineStageFlagBits2::eFragmentShader);
    CHECK(vulcao::detail::stage_for_layout(vk::ImageLayout::eDepthStencilReadOnlyOptimal) ==
          depth_read_stages);
    CHECK(vulcao::detail::stage_for_layout(vk::ImageLayout::eDepthReadOnlyOptimal) == depth_read_stages);
    CHECK(vulcao::detail::stage_for_layout(vk::ImageLayout::eStencilReadOnlyOptimal) == depth_read_stages);
}
