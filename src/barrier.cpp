#include "vulcao/detail/barrier.h"

namespace vulcao {

vk::PipelineStageFlags2 detail::stage_for_layout(vk::ImageLayout layout) {
    switch (layout) {
        case vk::ImageLayout::eUndefined:
            return vk::PipelineStageFlagBits2::eTopOfPipe;
        case vk::ImageLayout::eTransferSrcOptimal:
        case vk::ImageLayout::eTransferDstOptimal:
            return vk::PipelineStageFlagBits2::eTransfer;
        case vk::ImageLayout::eColorAttachmentOptimal:
            return vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        case vk::ImageLayout::eDepthStencilAttachmentOptimal:
        case vk::ImageLayout::eDepthAttachmentOptimal:
        case vk::ImageLayout::eStencilAttachmentOptimal:
            return vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                   vk::PipelineStageFlagBits2::eLateFragmentTests;
        case vk::ImageLayout::eDepthStencilReadOnlyOptimal:
        case vk::ImageLayout::eDepthReadOnlyOptimal:
        case vk::ImageLayout::eStencilReadOnlyOptimal:
            return vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                   vk::PipelineStageFlagBits2::eLateFragmentTests |
                   vk::PipelineStageFlagBits2::eFragmentShader;
        case vk::ImageLayout::eShaderReadOnlyOptimal:
            return vk::PipelineStageFlagBits2::eFragmentShader | vk::PipelineStageFlagBits2::eComputeShader;
        case vk::ImageLayout::ePresentSrcKHR:
            return vk::PipelineStageFlagBits2::eBottomOfPipe;
        default:
            // eGeneral, eAttachmentOptimal, eReadOnlyOptimal and anything unknown:
            // the layout admits accesses from any stage.
            return vk::PipelineStageFlagBits2::eAllCommands;
    }
}

vk::AccessFlags2 detail::access_for_layout(vk::ImageLayout layout) {
    switch (layout) {
        case vk::ImageLayout::eUndefined:
        case vk::ImageLayout::ePresentSrcKHR:
            return {};
        case vk::ImageLayout::eTransferSrcOptimal:
            return vk::AccessFlagBits2::eTransferRead;
        case vk::ImageLayout::eTransferDstOptimal:
            return vk::AccessFlagBits2::eTransferWrite;
        case vk::ImageLayout::eColorAttachmentOptimal:
            return vk::AccessFlagBits2::eColorAttachmentWrite;
        case vk::ImageLayout::eDepthStencilAttachmentOptimal:
        case vk::ImageLayout::eDepthAttachmentOptimal:
        case vk::ImageLayout::eStencilAttachmentOptimal:
            return vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
        case vk::ImageLayout::eDepthStencilReadOnlyOptimal:
            return vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eShaderRead;
        case vk::ImageLayout::eDepthReadOnlyOptimal:
        case vk::ImageLayout::eStencilReadOnlyOptimal:
            return vk::AccessFlagBits2::eDepthStencilAttachmentRead;
        case vk::ImageLayout::eShaderReadOnlyOptimal:
            return vk::AccessFlagBits2::eShaderRead;
        case vk::ImageLayout::eGeneral:
            // eGeneral admits reads and writes (storage image, shared
            // compute/graphics use), so the destination side of a barrier must
            // include write access or write-after-write hazards go unsynchronized.
            return vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
        default:
            // Unknown layouts may be written in, so order both reads and writes.
            return vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
    }
}

}
