#include "vulcao/command_buffer.h"

#include "vulcao/barrier.h"
#include "vulcao/buffer.h"
#include "vulcao/image.h"
#include "vulcao/pipeline.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace vulcao {
namespace {

vk::ImageSubresourceLayers single_layer(const vk::ImageSubresourceRange& range) {
    return vk::ImageSubresourceLayers{
        .aspectMask = range.aspectMask,
        .mipLevel = 0,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };
}

}

CommandBuffer::~CommandBuffer() {
    destroy();
}

CommandBuffer::CommandBuffer(CommandBuffer&& other) noexcept
    : device_(std::exchange(other.device_, vk::Device{})),
      pool_(std::exchange(other.pool_, vk::CommandPool{})),
      cmd_(std::exchange(other.cmd_, vk::CommandBuffer{})),
      level_(std::exchange(other.level_, vk::CommandBufferLevel::ePrimary)),
      begin_label_ext_(std::exchange(other.begin_label_ext_, nullptr)),
      end_label_ext_(std::exchange(other.end_label_ext_, nullptr)),
      insert_label_ext_(std::exchange(other.insert_label_ext_, nullptr)) {}

CommandBuffer& CommandBuffer::operator=(CommandBuffer&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = std::exchange(other.device_, vk::Device{});
        pool_ = std::exchange(other.pool_, vk::CommandPool{});
        cmd_ = std::exchange(other.cmd_, vk::CommandBuffer{});
        level_ = std::exchange(other.level_, vk::CommandBufferLevel::ePrimary);
        begin_label_ext_ = std::exchange(other.begin_label_ext_, nullptr);
        end_label_ext_ = std::exchange(other.end_label_ext_, nullptr);
        insert_label_ext_ = std::exchange(other.insert_label_ext_, nullptr);
    }
    return *this;
}

CommandBuffer CommandBuffer::allocate(vk::Device device,
                                      vk::CommandPool pool,
                                      vk::CommandBufferLevel level,
                                      bool debug_utils) {
    CommandBuffer command_buffer;
    command_buffer.device_ = device;
    command_buffer.pool_ = pool;
    command_buffer.level_ = level;
    if (debug_utils) {
        command_buffer.begin_label_ext_ = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
            device.getProcAddr("vkCmdBeginDebugUtilsLabelEXT"));
        command_buffer.end_label_ext_ = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
            device.getProcAddr("vkCmdEndDebugUtilsLabelEXT"));
        command_buffer.insert_label_ext_ = reinterpret_cast<PFN_vkCmdInsertDebugUtilsLabelEXT>(
            device.getProcAddr("vkCmdInsertDebugUtilsLabelEXT"));
    }
    command_buffer.cmd_ = device
                              .allocateCommandBuffers(vk::CommandBufferAllocateInfo{
                                  .commandPool = pool,
                                  .level = level,
                                  .commandBufferCount = 1,
                              })
                              .front();
    return command_buffer;
}

void CommandBuffer::destroy() {
    if (cmd_)
        device_.freeCommandBuffers(pool_, cmd_);

    device_ = nullptr;
    pool_ = nullptr;
    cmd_ = nullptr;
    level_ = vk::CommandBufferLevel::ePrimary;
    begin_label_ext_ = nullptr;
    end_label_ext_ = nullptr;
    insert_label_ext_ = nullptr;
}

CommandBuffer& CommandBuffer::reset(vk::CommandBufferResetFlags flags) {
    cmd_.reset(flags);
    return *this;
}

CommandBuffer& CommandBuffer::begin(vk::CommandBufferUsageFlags flags) {
    if (level_ == vk::CommandBufferLevel::eSecondary) {
        const vk::CommandBufferInheritanceInfo inheritance{};
        cmd_.begin(vk::CommandBufferBeginInfo{
            .flags = flags,
            .pInheritanceInfo = &inheritance,
        });
        return *this;
    }

    cmd_.begin(vk::CommandBufferBeginInfo{.flags = flags});
    return *this;
}

CommandBuffer& CommandBuffer::begin(const vk::CommandBufferInheritanceInfo& inheritance,
                                    vk::CommandBufferUsageFlags flags) {
    cmd_.begin(vk::CommandBufferBeginInfo{
        .flags = flags,
        .pInheritanceInfo = &inheritance,
    });
    return *this;
}

CommandBuffer& CommandBuffer::end() {
    cmd_.end();
    return *this;
}

CommandBuffer& CommandBuffer::begin_debug_label(const char* name, const std::array<float, 4>& color) {
    if (!begin_label_ext_)
        return *this;

    VkDebugUtilsLabelEXT label{};
    label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    label.pLabelName = name;
    for (size_t i = 0; i < color.size(); ++i)
        label.color[i] = color[i];

    begin_label_ext_(cmd_, &label);
    return *this;
}

CommandBuffer& CommandBuffer::end_debug_label() {
    if (!end_label_ext_)
        return *this;

    end_label_ext_(cmd_);
    return *this;
}

CommandBuffer& CommandBuffer::insert_debug_label(const char* name, const std::array<float, 4>& color) {
    if (!insert_label_ext_)
        return *this;

    VkDebugUtilsLabelEXT label{};
    label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    label.pLabelName = name;
    for (size_t i = 0; i < color.size(); ++i)
        label.color[i] = color[i];

    insert_label_ext_(cmd_, &label);
    return *this;
}

CommandBuffer& CommandBuffer::barrier(vk::PipelineStageFlags2 src_stage,
                                      vk::AccessFlags2 src_access,
                                      vk::PipelineStageFlags2 dst_stage,
                                      vk::AccessFlags2 dst_access) {
    const vk::MemoryBarrier2 memory_barrier{
        .srcStageMask = src_stage,
        .srcAccessMask = src_access,
        .dstStageMask = dst_stage,
        .dstAccessMask = dst_access,
    };

    cmd_.pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &memory_barrier,
    });
    return *this;
}

CommandBuffer& CommandBuffer::buffer_barrier(vk::Buffer buffer,
                                             vk::PipelineStageFlags2 src_stage,
                                             vk::AccessFlags2 src_access,
                                             vk::PipelineStageFlags2 dst_stage,
                                             vk::AccessFlags2 dst_access,
                                             vk::DeviceSize offset,
                                             vk::DeviceSize size,
                                             uint32_t src_queue_family,
                                             uint32_t dst_queue_family) {
    const vk::BufferMemoryBarrier2 buffer_memory_barrier{
        .srcStageMask = src_stage,
        .srcAccessMask = src_access,
        .dstStageMask = dst_stage,
        .dstAccessMask = dst_access,
        .srcQueueFamilyIndex = src_queue_family,
        .dstQueueFamilyIndex = dst_queue_family,
        .buffer = buffer,
        .offset = offset,
        .size = size,
    };

    cmd_.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &buffer_memory_barrier,
    });
    return *this;
}

CommandBuffer& CommandBuffer::release_buffer(vk::Buffer buffer,
                                             uint32_t producer_queue_family,
                                             uint32_t consumer_queue_family,
                                             vk::PipelineStageFlags2 src_stage,
                                             vk::AccessFlags2 src_access) {
    return buffer_barrier(buffer, src_stage, src_access, vk::PipelineStageFlagBits2::eNone,
                          vk::AccessFlagBits2::eNone, 0, VK_WHOLE_SIZE, producer_queue_family,
                          consumer_queue_family);
}

CommandBuffer& CommandBuffer::acquire_buffer(vk::Buffer buffer,
                                             uint32_t producer_queue_family,
                                             uint32_t consumer_queue_family,
                                             vk::PipelineStageFlags2 dst_stage,
                                             vk::AccessFlags2 dst_access) {
    return buffer_barrier(buffer, vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone,
                          dst_stage, dst_access, 0, VK_WHOLE_SIZE, producer_queue_family,
                          consumer_queue_family);
}

CommandBuffer& CommandBuffer::transition(vk::Image image,
                                         vk::ImageLayout old_layout,
                                         vk::ImageLayout new_layout,
                                         const vk::ImageSubresourceRange& range,
                                         vk::PipelineStageFlags2 src_stage,
                                         vk::AccessFlags2 src_access,
                                         vk::PipelineStageFlags2 dst_stage,
                                         vk::AccessFlags2 dst_access,
                                         uint32_t src_queue_family,
                                         uint32_t dst_queue_family) {
    if (!src_stage)
        src_stage = stage_for_layout(old_layout);
    if (!src_access)
        src_access = access_for_layout(old_layout);
    if (!dst_stage)
        dst_stage = stage_for_layout(new_layout);
    if (!dst_access)
        dst_access = access_for_layout(new_layout);

    const vk::ImageMemoryBarrier2 image_barrier{
        .srcStageMask = src_stage,
        .srcAccessMask = src_access,
        .dstStageMask = dst_stage,
        .dstAccessMask = dst_access,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = src_queue_family,
        .dstQueueFamilyIndex = dst_queue_family,
        .image = image,
        .subresourceRange = range,
    };

    cmd_.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &image_barrier,
    });
    return *this;
}

CommandBuffer& CommandBuffer::transition(Image& image,
                                         vk::ImageLayout new_layout,
                                         vk::PipelineStageFlags2 src_stage,
                                         vk::AccessFlags2 src_access,
                                         vk::PipelineStageFlags2 dst_stage,
                                         vk::AccessFlags2 dst_access) {
    const vk::ImageLayout old_layout = image.layout();
    transition(image.handle(), old_layout, new_layout, image.subresource_range(), src_stage, src_access,
               dst_stage, dst_access);
    image.set_layout(new_layout);
    return *this;
}

CommandBuffer& CommandBuffer::transition(Image& image,
                                         vk::ImageLayout new_layout,
                                         uint32_t src_queue_family,
                                         uint32_t dst_queue_family,
                                         vk::PipelineStageFlags2 src_stage,
                                         vk::AccessFlags2 src_access,
                                         vk::PipelineStageFlags2 dst_stage,
                                         vk::AccessFlags2 dst_access) {
    transition(image.handle(), image.layout(), new_layout, image.subresource_range(), src_stage,
               src_access, dst_stage, dst_access, src_queue_family, dst_queue_family);
    image.set_layout(new_layout);
    return *this;
}

CommandBuffer& CommandBuffer::release_image(Image& image,
                                            uint32_t producer_queue_family,
                                            uint32_t consumer_queue_family,
                                            vk::PipelineStageFlags2 src_stage,
                                            vk::AccessFlags2 src_access) {
    const vk::ImageLayout layout = image.layout();
    if (!src_stage)
        src_stage = stage_for_layout(layout);
    if (!src_access)
        src_access = access_for_layout(layout);

    const vk::ImageMemoryBarrier2 image_barrier{
        .srcStageMask = src_stage,
        .srcAccessMask = src_access,
        .dstStageMask = vk::PipelineStageFlagBits2::eNone,
        .dstAccessMask = vk::AccessFlagBits2::eNone,
        .oldLayout = layout,
        .newLayout = layout,
        .srcQueueFamilyIndex = producer_queue_family,
        .dstQueueFamilyIndex = consumer_queue_family,
        .image = image.handle(),
        .subresourceRange = image.subresource_range(),
    };

    cmd_.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &image_barrier,
    });
    return *this;
}

CommandBuffer& CommandBuffer::acquire_image(Image& image,
                                            uint32_t producer_queue_family,
                                            uint32_t consumer_queue_family,
                                            vk::PipelineStageFlags2 dst_stage,
                                            vk::AccessFlags2 dst_access) {
    const vk::ImageLayout layout = image.layout();
    if (!dst_stage)
        dst_stage = stage_for_layout(layout);
    if (!dst_access)
        dst_access = access_for_layout(layout);

    const vk::ImageMemoryBarrier2 image_barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eNone,
        .srcAccessMask = vk::AccessFlagBits2::eNone,
        .dstStageMask = dst_stage,
        .dstAccessMask = dst_access,
        .oldLayout = layout,
        .newLayout = layout,
        .srcQueueFamilyIndex = producer_queue_family,
        .dstQueueFamilyIndex = consumer_queue_family,
        .image = image.handle(),
        .subresourceRange = image.subresource_range(),
    };

    cmd_.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &image_barrier,
    });
    return *this;
}

CommandBuffer& CommandBuffer::copy_buffer(vk::Buffer src,
                                          vk::Buffer dst,
                                          vk::DeviceSize size,
                                          vk::DeviceSize src_offset,
                                          vk::DeviceSize dst_offset) {
    const vk::BufferCopy region{
        .srcOffset = src_offset,
        .dstOffset = dst_offset,
        .size = size,
    };
    cmd_.copyBuffer(src, dst, region);
    return *this;
}

CommandBuffer& CommandBuffer::copy_buffer(vk::Buffer src,
                                          vk::Buffer dst,
                                          std::span<const vk::BufferCopy> regions) {
    cmd_.copyBuffer(src, dst, static_cast<uint32_t>(regions.size()), regions.data());
    return *this;
}

CommandBuffer& CommandBuffer::copy_buffer_to_image(vk::Buffer src, vk::Image dst,
                                                   const vk::BufferImageCopy& region) {
    cmd_.copyBufferToImage(src, dst, vk::ImageLayout::eTransferDstOptimal, region);
    return *this;
}

CommandBuffer& CommandBuffer::copy_buffer_to_image(vk::Buffer src,
                                                   vk::Image dst,
                                                   std::span<const vk::BufferImageCopy> regions) {
    cmd_.copyBufferToImage(src, dst, vk::ImageLayout::eTransferDstOptimal,
                           static_cast<uint32_t>(regions.size()), regions.data());
    return *this;
}

CommandBuffer& CommandBuffer::copy_buffer_to_image(vk::Buffer src,
                                                   vk::Image dst,
                                                   vk::Extent3D extent,
                                                   const vk::ImageSubresourceLayers& layers,
                                                   vk::Offset3D offset) {
    return copy_buffer_to_image(src, dst, vk::BufferImageCopy{
                                             .bufferOffset = 0,
                                             .bufferRowLength = 0,
                                             .bufferImageHeight = 0,
                                             .imageSubresource = layers,
                                             .imageOffset = offset,
                                             .imageExtent = extent,
                                         });
}

CommandBuffer& CommandBuffer::copy_buffer_to_image(vk::Buffer src,
                                                   const Image& dst,
                                                   vk::Offset3D offset) {
    return copy_buffer_to_image(src, dst.handle(), dst.extent(),
                                single_layer(dst.subresource_range()), offset);
}

CommandBuffer& CommandBuffer::copy_image_to_buffer(vk::Buffer dst, vk::Image src,
                                                   const vk::BufferImageCopy& region) {
    cmd_.copyImageToBuffer(src, vk::ImageLayout::eTransferSrcOptimal, dst, region);
    return *this;
}

CommandBuffer& CommandBuffer::copy_image_to_buffer(vk::Buffer dst,
                                                   vk::Image src,
                                                   std::span<const vk::BufferImageCopy> regions) {
    cmd_.copyImageToBuffer(src, vk::ImageLayout::eTransferSrcOptimal, dst,
                           static_cast<uint32_t>(regions.size()), regions.data());
    return *this;
}

CommandBuffer& CommandBuffer::copy_image_to_buffer(vk::Buffer dst,
                                                   vk::Image src,
                                                   vk::Extent3D extent,
                                                   const vk::ImageSubresourceLayers& layers,
                                                   vk::Offset3D offset) {
    return copy_image_to_buffer(dst, src, vk::BufferImageCopy{
                                             .bufferOffset = 0,
                                             .bufferRowLength = 0,
                                             .bufferImageHeight = 0,
                                             .imageSubresource = layers,
                                             .imageOffset = offset,
                                             .imageExtent = extent,
                                         });
}

CommandBuffer& CommandBuffer::copy_image_to_buffer(vk::Buffer dst,
                                                   const Image& src,
                                                   vk::Offset3D offset) {
    return copy_image_to_buffer(dst, src.handle(), src.extent(),
                                single_layer(src.subresource_range()), offset);
}

CommandBuffer& CommandBuffer::generate_mipmaps(Image& image, vk::ImageLayout final_layout,
                                               vk::Filter filter) {
    const uint32_t level_count = image.mip_levels();
    if (level_count <= 1) {
        transition(image, final_layout);
        return *this;
    }

    const vk::ImageSubresourceRange full_range = image.subresource_range();
    const vk::ImageAspectFlags aspect = full_range.aspectMask;
    const uint32_t base_layer = full_range.baseArrayLayer;
    const uint32_t layer_count = full_range.layerCount;
    const vk::Extent3D extent = image.extent();
    const vk::Image handle = image.handle();

    const auto level_range = [&](uint32_t level) {
        return vk::ImageSubresourceRange{aspect, level, 1, base_layer, layer_count};
    };

    // Level 0 starts in TransferDst and becomes the first blit source. Each
    // level stays in TransferDst until it has been blitted into, then becomes
    // the source of the next level and is only moved to the final layout once
    // it is no longer needed: no layout is ever entered twice.
    transition(handle, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eTransferSrcOptimal,
               level_range(0));

    for (uint32_t level = 1; level < level_count; ++level) {
        const uint32_t src_width = std::max(1u, extent.width >> (level - 1));
        const uint32_t src_height = std::max(1u, extent.height >> (level - 1));
        const uint32_t dst_width = std::max(1u, extent.width >> level);
        const uint32_t dst_height = std::max(1u, extent.height >> level);

        for (uint32_t layer = 0; layer < layer_count; ++layer) {
            vk::ImageBlit region{};
            region.srcSubresource =
                vk::ImageSubresourceLayers{aspect, level - 1, base_layer + layer, 1};
            region.srcOffsets[0] = vk::Offset3D{0, 0, 0};
            region.srcOffsets[1] = vk::Offset3D{static_cast<int32_t>(src_width),
                                                static_cast<int32_t>(src_height), 1};
            region.dstSubresource = vk::ImageSubresourceLayers{aspect, level, base_layer + layer, 1};
            region.dstOffsets[0] = vk::Offset3D{0, 0, 0};
            region.dstOffsets[1] = vk::Offset3D{static_cast<int32_t>(dst_width),
                                                static_cast<int32_t>(dst_height), 1};

            blit_image(handle, vk::ImageLayout::eTransferSrcOptimal, handle,
                       vk::ImageLayout::eTransferDstOptimal, region, filter);
        }

        transition(handle, vk::ImageLayout::eTransferSrcOptimal, final_layout, level_range(level - 1));

        if (level + 1 < level_count)
            transition(handle, vk::ImageLayout::eTransferDstOptimal,
                       vk::ImageLayout::eTransferSrcOptimal, level_range(level));
    }

    transition(handle, vk::ImageLayout::eTransferDstOptimal, final_layout,
               level_range(level_count - 1));

    image.set_layout(final_layout);
    return *this;
}

CommandBuffer& CommandBuffer::copy_image(vk::Image src,
                                         vk::ImageLayout src_layout,
                                         vk::Image dst,
                                         vk::ImageLayout dst_layout,
                                         const vk::ImageCopy& region) {
    cmd_.copyImage(src, src_layout, dst, dst_layout, region);
    return *this;
}

CommandBuffer& CommandBuffer::copy_image(vk::Image src,
                                         vk::ImageLayout src_layout,
                                         vk::Image dst,
                                         vk::ImageLayout dst_layout,
                                         std::span<const vk::ImageCopy> regions) {
    cmd_.copyImage(src, src_layout, dst, dst_layout, static_cast<uint32_t>(regions.size()),
                   regions.data());
    return *this;
}

CommandBuffer& CommandBuffer::blit_image(vk::Image src,
                                         vk::ImageLayout src_layout,
                                         vk::Image dst,
                                         vk::ImageLayout dst_layout,
                                         const vk::ImageBlit& region,
                                         vk::Filter filter) {
    cmd_.blitImage(src, src_layout, dst, dst_layout, region, filter);
    return *this;
}

CommandBuffer& CommandBuffer::blit_image(vk::Image src,
                                         vk::ImageLayout src_layout,
                                         vk::Image dst,
                                         vk::ImageLayout dst_layout,
                                         std::span<const vk::ImageBlit> regions,
                                         vk::Filter filter) {
    cmd_.blitImage(src, src_layout, dst, dst_layout, static_cast<uint32_t>(regions.size()),
                   regions.data(), filter);
    return *this;
}

CommandBuffer& CommandBuffer::clear_color_image(vk::Image image,
                                                vk::ImageLayout layout,
                                                const vk::ClearColorValue& color,
                                                const vk::ImageSubresourceRange& range) {
    cmd_.clearColorImage(image, layout, color, range);
    return *this;
}

CommandBuffer& CommandBuffer::clear_color_image(const Image& image, const vk::ClearColorValue& color) {
    return clear_color_image(image.handle(), image.layout(), color, image.subresource_range());
}

CommandBuffer& CommandBuffer::clear_depth_stencil_image(vk::Image image,
                                                        vk::ImageLayout layout,
                                                        const vk::ClearDepthStencilValue& depth_stencil,
                                                        const vk::ImageSubresourceRange& range) {
    cmd_.clearDepthStencilImage(image, layout, depth_stencil, range);
    return *this;
}

CommandBuffer& CommandBuffer::clear_depth_stencil_image(const Image& image,
                                                        const vk::ClearDepthStencilValue& depth_stencil) {
    return clear_depth_stencil_image(image.handle(), image.layout(), depth_stencil,
                                     image.subresource_range());
}

CommandBuffer& CommandBuffer::transition_to_render(vk::Image image) {
    const vk::ImageSubresourceRange range{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
    // The source stage matches FrameManager::acquire_wait_stage so the barrier
    // of a frame begun through a FrameManager is ordered after the acquire.
    return transition(image, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
                      range, vk::PipelineStageFlagBits2::eColorAttachmentOutput);
}

CommandBuffer& CommandBuffer::transition_to_present(vk::Image image) {
    const vk::ImageSubresourceRange range{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
    return transition(image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR,
                      range);
}

CommandBuffer& CommandBuffer::execute_commands(vk::CommandBuffer cmd) {
    cmd_.executeCommands(cmd);
    return *this;
}

CommandBuffer& CommandBuffer::fill_buffer(vk::Buffer dst,
                                          vk::DeviceSize offset,
                                          vk::DeviceSize size,
                                          uint32_t data) {
    if (size == 0 || offset % 4 != 0 || size % 4 != 0)
        throw std::runtime_error("CommandBuffer::fill_buffer: size must be non-zero and offset and size must be multiples of 4");

    cmd_.fillBuffer(dst, offset, size, data);
    return *this;
}

CommandBuffer& CommandBuffer::update_buffer(vk::Buffer dst,
                                            vk::DeviceSize offset,
                                            const void* data,
                                            vk::DeviceSize size) {
    if (size == 0 || size % 4 != 0 || size > 65536)
        throw std::runtime_error("CommandBuffer::update_buffer: size must be a non-zero multiple of 4 and at most 65536 bytes");

    cmd_.updateBuffer(dst, offset, size, data);
    return *this;
}

CommandBuffer& CommandBuffer::bind_pipeline(vk::PipelineBindPoint bind_point, vk::Pipeline pipeline) {
    cmd_.bindPipeline(bind_point, pipeline);
    return *this;
}

CommandBuffer& CommandBuffer::bind_pipeline(const Pipeline& pipeline) {
    return bind_pipeline(pipeline.bind_point(), pipeline.handle());
}

CommandBuffer& CommandBuffer::bind_vertex_buffer(uint32_t binding,
                                                 const Buffer& buffer,
                                                 vk::DeviceSize offset) {
    const vk::Buffer raw_buffer = buffer.handle();
    cmd_.bindVertexBuffers(binding, raw_buffer, offset);
    return *this;
}

CommandBuffer& CommandBuffer::bind_vertex_buffers(uint32_t first_binding,
                                                  vk::ArrayProxy<const vk::Buffer> buffers,
                                                  vk::ArrayProxy<const vk::DeviceSize> offsets) {
    cmd_.bindVertexBuffers(first_binding, buffers, offsets);
    return *this;
}

CommandBuffer& CommandBuffer::bind_index_buffer(const Buffer& buffer,
                                                vk::DeviceSize offset,
                                                vk::IndexType index_type) {
    cmd_.bindIndexBuffer(buffer.handle(), offset, index_type);
    return *this;
}

CommandBuffer& CommandBuffer::set_viewport(const vk::Viewport& viewport) {
    cmd_.setViewport(0, viewport);
    return *this;
}

CommandBuffer& CommandBuffer::set_viewport(vk::Extent2D extent) {
    return set_viewport(vk::Viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(extent.width),
        .height = static_cast<float>(extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    });
}

CommandBuffer& CommandBuffer::set_scissor(const vk::Rect2D& scissor) {
    cmd_.setScissor(0, scissor);
    return *this;
}

CommandBuffer& CommandBuffer::set_scissor(vk::Extent2D extent) {
    return set_scissor(vk::Rect2D{.offset = vk::Offset2D{0, 0}, .extent = extent});
}

CommandBuffer& CommandBuffer::set_cull_mode(vk::CullModeFlags cull_mode) {
    cmd_.setCullMode(cull_mode);
    return *this;
}

CommandBuffer& CommandBuffer::set_front_face(vk::FrontFace front_face) {
    cmd_.setFrontFace(front_face);
    return *this;
}

CommandBuffer& CommandBuffer::set_depth_bias(float constant_factor, float clamp, float slope_factor) {
    cmd_.setDepthBias(constant_factor, clamp, slope_factor);
    return *this;
}

CommandBuffer& CommandBuffer::set_blend_constants(const std::array<float, 4>& constants) {
    cmd_.setBlendConstants(constants.data());
    return *this;
}

CommandBuffer& CommandBuffer::set_stencil_reference(vk::StencilFaceFlags face_mask,
                                                    uint32_t reference) {
    cmd_.setStencilReference(face_mask, reference);
    return *this;
}

CommandBuffer& CommandBuffer::set_depth_bounds(float min_depth_bounds, float max_depth_bounds) {
    cmd_.setDepthBounds(min_depth_bounds, max_depth_bounds);
    return *this;
}

CommandBuffer& CommandBuffer::set_line_width(float line_width) {
    cmd_.setLineWidth(line_width);
    return *this;
}

CommandBuffer& CommandBuffer::set_primitive_topology(vk::PrimitiveTopology topology) {
    cmd_.setPrimitiveTopology(topology);
    return *this;
}

CommandBuffer& CommandBuffer::set_depth_test_enable(bool enable) {
    cmd_.setDepthTestEnable(enable ? VK_TRUE : VK_FALSE);
    return *this;
}

CommandBuffer& CommandBuffer::set_depth_write_enable(bool enable) {
    cmd_.setDepthWriteEnable(enable ? VK_TRUE : VK_FALSE);
    return *this;
}

CommandBuffer& CommandBuffer::set_depth_compare_op(vk::CompareOp compare_op) {
    cmd_.setDepthCompareOp(compare_op);
    return *this;
}

CommandBuffer& CommandBuffer::push_constants(vk::PipelineLayout layout,
                                             vk::ShaderStageFlags stages,
                                             uint32_t offset,
                                             const void* data,
                                             uint32_t size) {
    cmd_.pushConstants(layout, stages, offset, size, data);
    return *this;
}

CommandBuffer& CommandBuffer::draw(uint32_t vertex_count,
                                   uint32_t instance_count,
                                   uint32_t first_vertex,
                                   uint32_t first_instance) {
    cmd_.draw(vertex_count, instance_count, first_vertex, first_instance);
    return *this;
}

CommandBuffer& CommandBuffer::draw_indexed(uint32_t index_count,
                                           uint32_t instance_count,
                                           uint32_t first_index,
                                           int32_t vertex_offset,
                                           uint32_t first_instance) {
    cmd_.drawIndexed(index_count, instance_count, first_index, vertex_offset, first_instance);
    return *this;
}

CommandBuffer& CommandBuffer::draw_indirect(vk::Buffer buffer,
                                            vk::DeviceSize offset,
                                            uint32_t draw_count,
                                            uint32_t stride) {
    cmd_.drawIndirect(buffer, offset, draw_count, stride);
    return *this;
}

CommandBuffer& CommandBuffer::draw_indexed_indirect(vk::Buffer buffer,
                                                    vk::DeviceSize offset,
                                                    uint32_t draw_count,
                                                    uint32_t stride) {
    cmd_.drawIndexedIndirect(buffer, offset, draw_count, stride);
    return *this;
}

CommandBuffer& CommandBuffer::begin_rendering(const vk::RenderingInfo& info) {
    cmd_.beginRendering(info);
    return *this;
}

CommandBuffer& CommandBuffer::begin_rendering(vk::Extent2D extent,
                                              std::span<const vk::RenderingAttachmentInfo> colors,
                                              const vk::RenderingAttachmentInfo* depth) {
    return begin_rendering(vk::RenderingInfo{
        .renderArea = vk::Rect2D{.offset = vk::Offset2D{0, 0}, .extent = extent},
        .layerCount = 1,
        .colorAttachmentCount = static_cast<uint32_t>(colors.size()),
        .pColorAttachments = colors.data(),
        .pDepthAttachment = depth,
    });
}

CommandBuffer& CommandBuffer::begin_rendering(vk::Extent2D extent,
                                              const vk::RenderingAttachmentInfo& color,
                                              const vk::RenderingAttachmentInfo* depth) {
    return begin_rendering(extent, std::span(&color, 1), depth);
}

CommandBuffer& CommandBuffer::end_rendering() {
    cmd_.endRendering();
    return *this;
}

CommandBuffer& CommandBuffer::bind_descriptor_sets(vk::PipelineBindPoint bind_point,
                                                   vk::PipelineLayout layout,
                                                   vk::ArrayProxy<const vk::DescriptorSet> descriptor_sets,
                                                   vk::ArrayProxy<const uint32_t> dynamic_offsets) {
    cmd_.bindDescriptorSets(bind_point, layout, 0, descriptor_sets, dynamic_offsets);
    return *this;
}

CommandBuffer& CommandBuffer::dispatch(uint32_t group_count_x,
                                       uint32_t group_count_y,
                                       uint32_t group_count_z) {
    cmd_.dispatch(group_count_x, group_count_y, group_count_z);
    return *this;
}

CommandBuffer& CommandBuffer::dispatch_indirect(vk::Buffer buffer, vk::DeviceSize offset) {
    cmd_.dispatchIndirect(buffer, offset);
    return *this;
}

CommandBuffer& CommandBuffer::resolve_image(vk::Image src,
                                            vk::ImageLayout src_layout,
                                            vk::Image dst,
                                            vk::ImageLayout dst_layout,
                                            std::span<const vk::ImageResolve> regions) {
    cmd_.resolveImage(src, src_layout, dst, dst_layout, static_cast<uint32_t>(regions.size()),
                      regions.data());
    return *this;
}

CommandBuffer& CommandBuffer::resolve_image(vk::Image src,
                                            vk::ImageLayout src_layout,
                                            vk::Image dst,
                                            vk::ImageLayout dst_layout,
                                            const vk::ImageResolve& region) {
    return resolve_image(src, src_layout, dst, dst_layout, std::span(&region, 1));
}

CommandBuffer& CommandBuffer::clear_attachments(std::span<const vk::ClearAttachment> attachments,
                                                std::span<const vk::ClearRect> rects) {
    cmd_.clearAttachments(static_cast<uint32_t>(attachments.size()), attachments.data(),
                          static_cast<uint32_t>(rects.size()), rects.data());
    return *this;
}

CommandBuffer& CommandBuffer::draw_indirect_count(vk::Buffer buffer,
                                                  vk::DeviceSize offset,
                                                  vk::Buffer count_buffer,
                                                  vk::DeviceSize count_offset,
                                                  uint32_t max_draw_count,
                                                  uint32_t stride) {
    cmd_.drawIndirectCount(buffer, offset, count_buffer, count_offset, max_draw_count, stride);
    return *this;
}

CommandBuffer& CommandBuffer::draw_indexed_indirect_count(vk::Buffer buffer,
                                                          vk::DeviceSize offset,
                                                          vk::Buffer count_buffer,
                                                          vk::DeviceSize count_offset,
                                                          uint32_t max_draw_count,
                                                          uint32_t stride) {
    cmd_.drawIndexedIndirectCount(buffer, offset, count_buffer, count_offset, max_draw_count, stride);
    return *this;
}

CommandBuffer& CommandBuffer::set_viewport_with_count(std::span<const vk::Viewport> viewports) {
    cmd_.setViewportWithCount(static_cast<uint32_t>(viewports.size()), viewports.data());
    return *this;
}

CommandBuffer& CommandBuffer::set_scissor_with_count(std::span<const vk::Rect2D> scissors) {
    cmd_.setScissorWithCount(static_cast<uint32_t>(scissors.size()), scissors.data());
    return *this;
}

CommandBuffer& CommandBuffer::set_event(vk::Event event, const vk::DependencyInfo& dependency) {
    cmd_.setEvent2(event, dependency);
    return *this;
}

CommandBuffer& CommandBuffer::reset_event(vk::Event event, vk::PipelineStageFlags2 stage) {
    cmd_.resetEvent2(event, stage);
    return *this;
}

CommandBuffer& CommandBuffer::wait_event(const vk::Event& event,
                                         vk::PipelineStageFlags2 src_stage,
                                         vk::PipelineStageFlags2 dst_stage) {
    const vk::MemoryBarrier2 barrier{
        .srcStageMask = src_stage,
        .dstStageMask = dst_stage,
    };
    const vk::DependencyInfo dependency{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier,
    };
    cmd_.waitEvents2(event, dependency);
    return *this;
}

CommandBuffer& CommandBuffer::reset_query_pool(vk::QueryPool pool,
                                               uint32_t first_query,
                                               uint32_t query_count) {
    cmd_.resetQueryPool(pool, first_query, query_count);
    return *this;
}

CommandBuffer& CommandBuffer::write_timestamp(vk::QueryPool pool,
                                              vk::PipelineStageFlags2 stage,
                                              uint32_t query) {
    cmd_.writeTimestamp2(stage, pool, query);
    return *this;
}

CommandBuffer& CommandBuffer::begin_query(vk::QueryPool pool, uint32_t query, vk::QueryControlFlags flags) {
    cmd_.beginQuery(pool, query, flags);
    return *this;
}

CommandBuffer& CommandBuffer::end_query(vk::QueryPool pool, uint32_t query) {
    cmd_.endQuery(pool, query);
    return *this;
}

CommandBuffer& CommandBuffer::copy_query_pool_results(vk::QueryPool pool,
                                                      uint32_t first_query,
                                                      uint32_t query_count,
                                                      vk::Buffer dst,
                                                      vk::DeviceSize dst_offset,
                                                      vk::DeviceSize stride,
                                                      vk::QueryResultFlags flags) {
    cmd_.copyQueryPoolResults(pool, first_query, query_count, dst, dst_offset, stride, flags);
    return *this;
}

}
