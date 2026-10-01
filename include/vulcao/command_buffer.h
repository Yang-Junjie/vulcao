#pragma once

#include <array>
#include <cstdint>
#include <ranges>
#include <span>
#include <type_traits>

#include <vulkan/vulkan.hpp>

#include "vulcao/acceleration_structure.h"

namespace vulcao {

class Buffer;
class Image;
class Pipeline;

/// @brief RAII wrapper around a Vulkan command buffer with recording helpers.
class CommandBuffer {
public:
    /// @brief Creates an empty command buffer.
    CommandBuffer() = default;

    /// @brief Frees the command buffer from its pool.
    ~CommandBuffer();

    /// @brief Not copyable.
    CommandBuffer(const CommandBuffer&) = delete;
    CommandBuffer& operator=(const CommandBuffer&) = delete;

    /// @brief Moves the command buffer, leaving the source empty.
    CommandBuffer(CommandBuffer&& other) noexcept;

    /// @brief Move assignment. Destroys the current command buffer first.
    CommandBuffer& operator=(CommandBuffer&& other) noexcept;

    /// @brief Allocates one command buffer from a pool.
    /// @param device Device that owns the pool.
    /// @param pool Pool to allocate from.
    /// @param level Command buffer level.
    /// @param debug_utils True to enable debug labels on this command buffer.
    /// @return The allocated command buffer.
    static CommandBuffer allocate(vk::Device device,
                                  vk::CommandPool pool,
                                  vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary,
                                  bool debug_utils = false);

    /// @brief Returns true if the command buffer holds a valid handle.
    bool valid() const { return static_cast<bool>(cmd_); }

    /// @brief Returns true if the command buffer holds a valid handle.
    explicit operator bool() const { return valid(); }

    /// @brief Returns the raw Vulkan command buffer handle.
    vk::CommandBuffer handle() const { return cmd_; }

    /// @brief Returns the level the command buffer was allocated with.
    vk::CommandBufferLevel level() const { return level_; }

    /// @brief Frees the command buffer and resets the wrapper.
    void destroy();

    /// @brief Resets the command buffer to the initial recording state.
    /// @param flags Reset flags.
    /// @return This command buffer.
    CommandBuffer& reset(vk::CommandBufferResetFlags flags = {});

    /// @brief Starts recording. Secondary command buffers get default inheritance info.
    /// @param flags Usage flags for the recording.
    /// @return This command buffer.
    CommandBuffer& begin(vk::CommandBufferUsageFlags flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    /// @brief Starts recording with explicit inheritance info.
    /// @param inheritance Inheritance info for the recording.
    /// @param flags Usage flags for the recording.
    /// @return This command buffer.
    CommandBuffer& begin(const vk::CommandBufferInheritanceInfo& inheritance,
                         vk::CommandBufferUsageFlags flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    /// @brief Stops recording.
    /// @return This command buffer.
    CommandBuffer& end();

    /// @brief Begins a debug label region. No-op unless debug labels were enabled.
    /// @param name Label name.
    /// @param color Label color.
    /// @return This command buffer.
    CommandBuffer& begin_debug_label(const char* name, const std::array<float, 4>& color = {});

    /// @brief Ends the current debug label region.
    /// @return This command buffer.
    CommandBuffer& end_debug_label();

    /// @brief Inserts a single debug label. No-op unless debug labels were enabled.
    /// @param name Label name.
    /// @param color Label color.
    /// @return This command buffer.
    CommandBuffer& insert_debug_label(const char* name, const std::array<float, 4>& color = {});

    /// @brief Inserts a global memory barrier.
    /// @param src_stage Source pipeline stages.
    /// @param src_access Source access flags.
    /// @param dst_stage Destination pipeline stages.
    /// @param dst_access Destination access flags.
    /// @return This command buffer.
    CommandBuffer& barrier(vk::PipelineStageFlags2 src_stage,
                           vk::AccessFlags2 src_access,
                           vk::PipelineStageFlags2 dst_stage,
                           vk::AccessFlags2 dst_access);

    /// @brief Inserts a buffer memory barrier.
    /// @param buffer Buffer to synchronize.
    /// @param src_stage Source pipeline stages.
    /// @param src_access Source access flags.
    /// @param dst_stage Destination pipeline stages.
    /// @param dst_access Destination access flags.
    /// @param offset Byte offset of the range.
    /// @param size Size of the range, or VK_WHOLE_SIZE.
    /// @param src_queue_family Queue family that currently owns the buffer, or VK_QUEUE_FAMILY_IGNORED.
    /// @param dst_queue_family Queue family that receives ownership, or VK_QUEUE_FAMILY_IGNORED.
    /// @return This command buffer.
    CommandBuffer& buffer_barrier(vk::Buffer buffer,
                                  vk::PipelineStageFlags2 src_stage,
                                  vk::AccessFlags2 src_access,
                                  vk::PipelineStageFlags2 dst_stage,
                                  vk::AccessFlags2 dst_access,
                                  vk::DeviceSize offset = 0,
                                  vk::DeviceSize size = VK_WHOLE_SIZE,
                                  uint32_t src_queue_family = VK_QUEUE_FAMILY_IGNORED,
                                  uint32_t dst_queue_family = VK_QUEUE_FAMILY_IGNORED);

    /// @brief Releases ownership of a buffer to another queue family.
    /// @param buffer Buffer to release.
    /// @param producer_queue_family Queue family that currently owns the buffer.
    /// @param consumer_queue_family Queue family that receives ownership.
    /// @param src_stage Stages that produced the writes.
    /// @param src_access Writes to make visible.
    /// @return This command buffer.
    CommandBuffer& release_buffer(vk::Buffer buffer,
                                  uint32_t producer_queue_family,
                                  uint32_t consumer_queue_family,
                                  vk::PipelineStageFlags2 src_stage,
                                  vk::AccessFlags2 src_access);

    /// @brief Acquires ownership of a buffer from another queue family.
    /// @param buffer Buffer to acquire.
    /// @param producer_queue_family Queue family that released the buffer.
    /// @param consumer_queue_family Queue family that receives ownership.
    /// @param dst_stage Stages that will access the buffer.
    /// @param dst_access Accesses to make available.
    /// @return This command buffer.
    CommandBuffer& acquire_buffer(vk::Buffer buffer,
                                  uint32_t producer_queue_family,
                                  uint32_t consumer_queue_family,
                                  vk::PipelineStageFlags2 dst_stage,
                                  vk::AccessFlags2 dst_access);

    /// @brief Transitions an image layout using raw handles.
    ///
    /// The tracked layout is a CPU-side convention that vulcao maintains on the
    /// Image wrapper; this overload takes a bare handle, so it cannot read or
    /// update that state. Prefer the Image& overload unless the image is not
    /// owned by a vulcao::Image — practical for swapchain images, which the
    /// Context owns rather than an Image wrapper, so nothing is being tracked
    /// and there is no state to desynchronize. Using this on a tracked image
    /// leaves Image::layout() stale, and later barriers then record the wrong
    /// oldLayout; the resulting hazard is silent outside the validation layers.
    /// @param image Image to transition.
    /// @param old_layout Current layout.
    /// @param new_layout Target layout.
    /// @param range Subresource range to transition.
    /// @param src_stage Source stages. Derived from old_layout when zero.
    /// @param src_access Source access. Derived from old_layout when zero.
    /// @param dst_stage Destination stages. Derived from new_layout when zero.
    /// @param dst_access Destination access. Derived from new_layout when zero.
    /// @param src_queue_family Queue family that currently owns the image, or VK_QUEUE_FAMILY_IGNORED.
    /// @param dst_queue_family Queue family that receives ownership, or VK_QUEUE_FAMILY_IGNORED.
    /// @return This command buffer.
    CommandBuffer& transition(vk::Image image,
                              vk::ImageLayout old_layout,
                              vk::ImageLayout new_layout,
                              const vk::ImageSubresourceRange& range,
                              vk::PipelineStageFlags2 src_stage = {},
                              vk::AccessFlags2 src_access = {},
                              vk::PipelineStageFlags2 dst_stage = {},
                              vk::AccessFlags2 dst_access = {},
                              uint32_t src_queue_family = VK_QUEUE_FAMILY_IGNORED,
                              uint32_t dst_queue_family = VK_QUEUE_FAMILY_IGNORED);

    /// @brief Transitions an image layout and updates its tracked layout.
    ///
    /// The old layout comes from Image::layout(), so the tracked state must be
    /// accurate when this is called. Start from what the image was created with
    /// and use this overload for each transition; mixing in the raw-handle
    /// overload without updating the wrapper desynchronizes the two.
    /// @param image Image to transition.
    /// @param new_layout Target layout.
    /// @param src_stage Source stages. Derived from the tracked layout when zero.
    /// @param src_access Source access. Derived from the tracked layout when zero.
    /// @param dst_stage Destination stages. Derived from new_layout when zero.
    /// @param dst_access Destination access. Derived from new_layout when zero.
    /// @return This command buffer.
    CommandBuffer& transition(Image& image,
                              vk::ImageLayout new_layout,
                              vk::PipelineStageFlags2 src_stage = {},
                              vk::AccessFlags2 src_access = {},
                              vk::PipelineStageFlags2 dst_stage = {},
                              vk::AccessFlags2 dst_access = {});

    /// @brief Transitions an image layout while changing its queue family
    ///        ownership, and updates its tracked layout.
    ///
    /// Complements release_image/acquire_image (which keep the layout) for the
    /// common case where the releasing barrier also moves the layout to its
    /// final state, e.g. a transfer queue upload ending in ShaderReadOnly.
    /// @param image Image to transition.
    /// @param new_layout Target layout.
    /// @param src_queue_family Queue family that currently owns the image.
    /// @param dst_queue_family Queue family that receives ownership.
    /// @param src_stage Source stages. Derived from the tracked layout when zero.
    /// @param src_access Source access. Derived from the tracked layout when zero.
    /// @param dst_stage Destination stages. Derived from new_layout when zero.
    /// @param dst_access Destination access. Derived from new_layout when zero.
    /// @return This command buffer.
    CommandBuffer& transition(Image& image,
                              vk::ImageLayout new_layout,
                              uint32_t src_queue_family,
                              uint32_t dst_queue_family,
                              vk::PipelineStageFlags2 src_stage = {},
                              vk::AccessFlags2 src_access = {},
                              vk::PipelineStageFlags2 dst_stage = {},
                              vk::AccessFlags2 dst_access = {});

    /// @brief Releases ownership of an image to another queue family.
    /// @param image Image to release. Its tracked layout is unchanged.
    /// @param producer_queue_family Queue family that currently owns the image.
    /// @param consumer_queue_family Queue family that receives ownership.
    /// @param src_stage Stages that produced the writes. Derived from the tracked layout when zero.
    /// @param src_access Writes to make visible. Derived from the tracked layout when zero.
    /// @return This command buffer.
    CommandBuffer& release_image(Image& image,
                                 uint32_t producer_queue_family,
                                 uint32_t consumer_queue_family,
                                 vk::PipelineStageFlags2 src_stage = {},
                                 vk::AccessFlags2 src_access = {});

    /// @brief Acquires ownership of an image from another queue family.
    /// @param image Image to acquire. Its tracked layout is unchanged.
    /// @param producer_queue_family Queue family that released the image.
    /// @param consumer_queue_family Queue family that receives ownership.
    /// @param dst_stage Stages that will access the image. Derived from the tracked layout when zero.
    /// @param dst_access Accesses to make available. Derived from the tracked layout when zero.
    /// @return This command buffer.
    CommandBuffer& acquire_image(Image& image,
                                 uint32_t producer_queue_family,
                                 uint32_t consumer_queue_family,
                                 vk::PipelineStageFlags2 dst_stage = {},
                                 vk::AccessFlags2 dst_access = {});

    /// @brief Copies data between buffers.
    /// @param src Source buffer.
    /// @param dst Destination buffer.
    /// @param size Number of bytes to copy.
    /// @param src_offset Byte offset in the source.
    /// @param dst_offset Byte offset in the destination.
    /// @return This command buffer.
    CommandBuffer& copy_buffer(vk::Buffer src,
                               vk::Buffer dst,
                               vk::DeviceSize size,
                               vk::DeviceSize src_offset = 0,
                               vk::DeviceSize dst_offset = 0);

    /// @brief Copies a buffer into an image using raw handles.
    /// @param src Source buffer.
    /// @param dst Destination image, must be in TransferDst layout.
    /// @param region Full copy region, including buffer offset, row length and extent.
    /// @return This command buffer.
    CommandBuffer& copy_buffer_to_image(vk::Buffer src, vk::Image dst,
                                        const vk::BufferImageCopy& region);

    /// @brief Copies a buffer into an image using raw handles.
    /// @param src Source buffer.
    /// @param dst Destination image, must be in TransferDst layout.
    /// @param extent Size of the copied region.
    /// @param layers Image subresource layers to copy into.
    /// @param offset Offset in the destination image.
    /// @return This command buffer.
    CommandBuffer& copy_buffer_to_image(vk::Buffer src,
                                        vk::Image dst,
                                        vk::Extent3D extent,
                                        const vk::ImageSubresourceLayers& layers,
                                        vk::Offset3D offset = vk::Offset3D{0, 0, 0});

    /// @brief Copies a buffer into mip 0, layer 0 of an image.
    /// @param src Source buffer.
    /// @param dst Destination image, must be in TransferDst layout.
    /// @param offset Offset in the destination image.
    /// @return This command buffer.
    CommandBuffer& copy_buffer_to_image(vk::Buffer src,
                                        const Image& dst,
                                        vk::Offset3D offset = vk::Offset3D{0, 0, 0});

    /// @brief Copies an image into a buffer using raw handles.
    /// @param dst Destination buffer.
    /// @param src Source image, must be in TransferSrc layout.
    /// @param region Full copy region, including buffer offset, row length and extent.
    /// @return This command buffer.
    CommandBuffer& copy_image_to_buffer(vk::Buffer dst, vk::Image src,
                                        const vk::BufferImageCopy& region);

    /// @brief Copies an image into a buffer using raw handles.
    /// @param dst Destination buffer.
    /// @param src Source image, must be in TransferSrc layout.
    /// @param extent Size of the copied region.
    /// @param layers Image subresource layers to copy from.
    /// @param offset Offset in the source image.
    /// @return This command buffer.
    CommandBuffer& copy_image_to_buffer(vk::Buffer dst,
                                        vk::Image src,
                                        vk::Extent3D extent,
                                        const vk::ImageSubresourceLayers& layers,
                                        vk::Offset3D offset = vk::Offset3D{0, 0, 0});

    /// @brief Copies mip 0, layer 0 of an image into a buffer.
    /// @param dst Destination buffer.
    /// @param src Source image, must be in TransferSrc layout.
    /// @param offset Offset in the source image.
    /// @return This command buffer.
    CommandBuffer& copy_image_to_buffer(vk::Buffer dst,
                                        const Image& src,
                                        vk::Offset3D offset = vk::Offset3D{0, 0, 0});

    /// @brief Generates the mip chain of an image by blitting level to level.
    ///
    /// Every layer of the image's subresource range is processed. The whole
    /// range must be in TransferDst layout with mip 0 of every layer filled,
    /// and the image must have been created with TransferSrc | TransferDst
    /// usage. The caller must make sure the format supports blitting with the
    /// chosen filter (VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT for
    /// eLinear).
    /// @param image Image to generate mipmaps for.
    /// @param final_layout Layout all levels are transitioned to at the end.
    /// @param filter Filter used when scaling down between levels.
    /// @return This command buffer.
    CommandBuffer& generate_mipmaps(Image& image,
                                    vk::ImageLayout final_layout = vk::ImageLayout::eShaderReadOnlyOptimal,
                                    vk::Filter filter = vk::Filter::eLinear);

    /// @brief Copies one image into another using raw handles.
    /// @param src Source image.
    /// @param src_layout Layout of the source image.
    /// @param dst Destination image.
    /// @param dst_layout Layout of the destination image.
    /// @param region Region to copy.
    /// @return This command buffer.
    CommandBuffer& copy_image(vk::Image src,
                              vk::ImageLayout src_layout,
                              vk::Image dst,
                              vk::ImageLayout dst_layout,
                              const vk::ImageCopy& region);

    /// @brief Blits one image into another using raw handles.
    /// @param src Source image.
    /// @param src_layout Layout of the source image.
    /// @param dst Destination image.
    /// @param dst_layout Layout of the destination image.
    /// @param region Region to blit.
    /// @param filter Filter used when scaling.
    /// @return This command buffer.
    CommandBuffer& blit_image(vk::Image src,
                              vk::ImageLayout src_layout,
                              vk::Image dst,
                              vk::ImageLayout dst_layout,
                              const vk::ImageBlit& region,
                              vk::Filter filter = vk::Filter::eLinear);

    /// @brief Clears a color image using raw handles.
    /// @param image Image to clear.
    /// @param layout Layout of the image.
    /// @param color Clear color.
    /// @param range Subresource range to clear.
    /// @return This command buffer.
    CommandBuffer& clear_color_image(vk::Image image,
                                     vk::ImageLayout layout,
                                     const vk::ClearColorValue& color,
                                     const vk::ImageSubresourceRange& range);

    /// @brief Clears a color image using its tracked layout and range.
    ///
    /// Takes the layout from Image::layout(), which must therefore be accurate;
    /// a clear does not change the layout, so the tracked value stays valid.
    /// @param image Image to clear.
    /// @param color Clear color.
    /// @return This command buffer.
    CommandBuffer& clear_color_image(const Image& image, const vk::ClearColorValue& color);

    /// @brief Clears a depth-stencil image using raw handles.
    /// @param image Image to clear.
    /// @param layout Layout of the image.
    /// @param depth_stencil Clear depth and stencil values.
    /// @param range Subresource range to clear.
    /// @return This command buffer.
    CommandBuffer& clear_depth_stencil_image(vk::Image image,
                                             vk::ImageLayout layout,
                                             const vk::ClearDepthStencilValue& depth_stencil,
                                             const vk::ImageSubresourceRange& range);

    /// @brief Clears a depth-stencil image using its tracked layout and range.
    ///
    /// Takes the layout from Image::layout(); a clear does not change it, so the
    /// tracked value stays valid afterwards.
    /// @param image Image to clear.
    /// @param depth_stencil Clear depth and stencil values.
    /// @return This command buffer.
    CommandBuffer& clear_depth_stencil_image(const Image& image,
                                             const vk::ClearDepthStencilValue& depth_stencil);

    /// @brief Transitions a swapchain image to the color attachment layout.
    ///
    /// The image is treated as undefined: its previous contents are discarded,
    /// which is the usual swapchain rendering pattern. The source stage matches
    /// FrameManager::acquire_wait_stage, so the barrier is ordered after the
    /// acquire of a frame begun through a FrameManager.
    /// @param image Acquired swapchain image.
    /// @return This command buffer.
    CommandBuffer& transition_to_render(vk::Image image);

    /// @brief Transitions a swapchain image from the color attachment layout to
    ///        the present layout.
    /// @param image Swapchain image that was rendered into.
    /// @return This command buffer.
    CommandBuffer& transition_to_present(vk::Image image);

    /// @brief Executes the recorded commands of a secondary command buffer.
    /// @param cmd Secondary command buffer to execute.
    /// @return This command buffer.
    CommandBuffer& execute_commands(vk::CommandBuffer cmd);

    /// @brief Fills a buffer range with a 32 bit value.
    /// @param dst Destination buffer.
    /// @param offset Byte offset of the range, must be a multiple of 4.
    /// @param size Size of the range in bytes, must be a multiple of 4.
    /// @param data Value written to the range.
    /// @return This command buffer.
    /// @throws std::runtime_error if size is zero or offset or size is not a multiple of 4.
    CommandBuffer& fill_buffer(vk::Buffer dst, vk::DeviceSize offset, vk::DeviceSize size, uint32_t data);

    /// @brief Copies raw bytes into a buffer from host memory.
    /// @param dst Destination buffer.
    /// @param offset Byte offset in the destination.
    /// @param data Source pointer.
    /// @param size Number of bytes to copy, at most 65536.
    /// @return This command buffer.
    /// @throws std::runtime_error if size is not a non-zero multiple of 4 or exceeds 65536.
    CommandBuffer& update_buffer(vk::Buffer dst, vk::DeviceSize offset, const void* data, vk::DeviceSize size);

    /// @brief Copies a contiguous range into a buffer from host memory.
    /// @tparam Container Contiguous range of trivially copyable values.
    /// @param dst Destination buffer.
    /// @param offset Byte offset in the destination.
    /// @param data Source range, at most 65536 bytes.
    /// @return This command buffer.
    /// @throws std::runtime_error if size is not a non-zero multiple of 4 or exceeds 65536.
    template <typename Container>
        requires std::ranges::contiguous_range<Container>
    CommandBuffer& update_buffer(vk::Buffer dst, vk::DeviceSize offset, const Container& data) {
        using T = std::ranges::range_value_t<Container>;
        static_assert(std::is_trivially_copyable_v<T>, "buffer data must be trivially copyable");
        return update_buffer(dst,
                             offset,
                             std::ranges::data(data),
                             static_cast<vk::DeviceSize>(std::ranges::size(data)) * sizeof(T));
    }

    /// @brief Binds a pipeline.
    /// @param bind_point Pipeline bind point.
    /// @param pipeline Pipeline to bind.
    /// @return This command buffer.
    CommandBuffer& bind_pipeline(vk::PipelineBindPoint bind_point, vk::Pipeline pipeline);

    /// @brief Binds a pipeline at its own bind point.
    /// @param pipeline Pipeline to bind.
    /// @return This command buffer.
    CommandBuffer& bind_pipeline(const Pipeline& pipeline);

    /// @brief Binds one vertex buffer.
    /// @param binding Vertex binding index.
    /// @param buffer Vertex buffer to bind.
    /// @param offset Byte offset in the buffer.
    /// @return This command buffer.
    CommandBuffer& bind_vertex_buffer(uint32_t binding, const Buffer& buffer, vk::DeviceSize offset = 0);

    /// @brief Binds several vertex buffers.
    /// @param first_binding First vertex binding index.
    /// @param buffers Vertex buffers to bind.
    /// @param offsets Byte offsets of each buffer.
    /// @return This command buffer.
    CommandBuffer& bind_vertex_buffers(uint32_t first_binding,
                                       vk::ArrayProxy<const vk::Buffer> buffers,
                                       vk::ArrayProxy<const vk::DeviceSize> offsets);

    /// @brief Binds the index buffer.
    /// @param buffer Index buffer to bind.
    /// @param offset Byte offset in the buffer.
    /// @param index_type Type of the indices.
    /// @return This command buffer.
    CommandBuffer& bind_index_buffer(const Buffer& buffer, vk::DeviceSize offset, vk::IndexType index_type);

    /// @brief Sets one viewport at index 0.
    /// @param viewport Viewport to set.
    /// @return This command buffer.
    CommandBuffer& set_viewport(const vk::Viewport& viewport);

    /// @brief Sets a full extent viewport at index 0.
    /// @param extent Width and height of the viewport.
    /// @return This command buffer.
    CommandBuffer& set_viewport(vk::Extent2D extent);

    /// @brief Sets one scissor at index 0.
    /// @param scissor Scissor rectangle to set.
    /// @return This command buffer.
    CommandBuffer& set_scissor(const vk::Rect2D& scissor);

    /// @brief Sets a full extent scissor at index 0.
    /// @param extent Width and height of the scissor.
    /// @return This command buffer.
    CommandBuffer& set_scissor(vk::Extent2D extent);

    /// @brief Sets the cull mode (requires the matching dynamic state).
    /// @param cull_mode Cull mode to set.
    /// @return This command buffer.
    CommandBuffer& set_cull_mode(vk::CullModeFlags cull_mode);

    /// @brief Sets the front face (requires the matching dynamic state).
    /// @param front_face Front face winding.
    /// @return This command buffer.
    CommandBuffer& set_front_face(vk::FrontFace front_face);

    /// @brief Sets the depth bias (requires the matching dynamic state).
    /// @param constant_factor Constant depth bias factor.
    /// @param clamp Maximum depth bias.
    /// @param slope_factor Slope depth bias factor.
    /// @return This command buffer.
    CommandBuffer& set_depth_bias(float constant_factor, float clamp, float slope_factor);

    /// @brief Sets the blend constants (requires the matching dynamic state).
    /// @param constants RGBA blend constants.
    /// @return This command buffer.
    CommandBuffer& set_blend_constants(const std::array<float, 4>& constants);

    /// @brief Sets the stencil reference (requires the matching dynamic state).
    /// @param face_mask Faces the reference applies to.
    /// @param reference Reference value.
    /// @return This command buffer.
    CommandBuffer& set_stencil_reference(vk::StencilFaceFlags face_mask, uint32_t reference);

    /// @brief Sets the depth bounds (requires the matching dynamic state).
    /// @param min_depth_bounds Minimum depth bound.
    /// @param max_depth_bounds Maximum depth bound.
    /// @return This command buffer.
    CommandBuffer& set_depth_bounds(float min_depth_bounds, float max_depth_bounds);

    /// @brief Sets the line width (requires the matching dynamic state).
    /// @param line_width Line width.
    /// @return This command buffer.
    CommandBuffer& set_line_width(float line_width);

    /// @brief Sets the primitive topology (requires the matching dynamic state).
    /// @param topology Topology to set.
    /// @return This command buffer.
    CommandBuffer& set_primitive_topology(vk::PrimitiveTopology topology);

    /// @brief Enables or disables depth testing (requires the matching dynamic state).
    /// @param enable True to enable depth testing.
    /// @return This command buffer.
    CommandBuffer& set_depth_test_enable(bool enable);

    /// @brief Enables or disables depth writes (requires the matching dynamic state).
    /// @param enable True to enable depth writes.
    /// @return This command buffer.
    CommandBuffer& set_depth_write_enable(bool enable);

    /// @brief Sets the depth compare operation (requires the matching dynamic state).
    /// @param compare_op Compare operation.
    /// @return This command buffer.
    CommandBuffer& set_depth_compare_op(vk::CompareOp compare_op);

    /// @brief Pushes raw constant data to the pipeline layout.
    /// @param layout Pipeline layout.
    /// @param stages Shader stages that read the constants.
    /// @param offset Byte offset in the push constant range.
    /// @param data Source pointer.
    /// @param size Number of bytes to push.
    /// @return This command buffer.
    CommandBuffer& push_constants(vk::PipelineLayout layout,
                                  vk::ShaderStageFlags stages,
                                  uint32_t offset,
                                  const void* data,
                                  uint32_t size);

    /// @brief Pushes one trivially copyable value to the pipeline layout.
    /// @tparam T Trivially copyable value type.
    /// @param layout Pipeline layout.
    /// @param stages Shader stages that read the constants.
    /// @param offset Byte offset in the push constant range.
    /// @param value Value to push.
    /// @return This command buffer.
    template <typename T>
    CommandBuffer& push_constants(vk::PipelineLayout layout,
                                  vk::ShaderStageFlags stages,
                                  uint32_t offset,
                                  const T& value) {
        static_assert(std::is_trivially_copyable_v<T>, "push constant data must be trivially copyable");
        return push_constants(layout, stages, offset, &value, static_cast<uint32_t>(sizeof(T)));
    }

    /// @brief Records a draw call.
    /// @param vertex_count Number of vertices.
    /// @param instance_count Number of instances.
    /// @param first_vertex Index of the first vertex.
    /// @param first_instance Index of the first instance.
    /// @return This command buffer.
    CommandBuffer& draw(uint32_t vertex_count,
                        uint32_t instance_count = 1,
                        uint32_t first_vertex = 0,
                        uint32_t first_instance = 0);

    /// @brief Records an indexed draw call.
    /// @param index_count Number of indices.
    /// @param instance_count Number of instances.
    /// @param first_index Index of the first index.
    /// @param vertex_offset Value added to each index.
    /// @param first_instance Index of the first instance.
    /// @return This command buffer.
    CommandBuffer& draw_indexed(uint32_t index_count,
                                uint32_t instance_count = 1,
                                uint32_t first_index = 0,
                                int32_t vertex_offset = 0,
                                uint32_t first_instance = 0);

    /// @brief Records an indirect draw call.
    /// @param buffer Buffer holding the draw parameters.
    /// @param offset Byte offset of the draw parameters.
    /// @param draw_count Number of draws.
    /// @param stride Byte stride between draw parameters.
    /// @return This command buffer.
    CommandBuffer& draw_indirect(vk::Buffer buffer,
                                 vk::DeviceSize offset,
                                 uint32_t draw_count,
                                 uint32_t stride);

    /// @brief Records an indirect indexed draw call.
    /// @param buffer Buffer holding the draw parameters.
    /// @param offset Byte offset of the draw parameters.
    /// @param draw_count Number of draws.
    /// @param stride Byte stride between draw parameters.
    /// @return This command buffer.
    CommandBuffer& draw_indexed_indirect(vk::Buffer buffer,
                                         vk::DeviceSize offset,
                                         uint32_t draw_count,
                                         uint32_t stride);

    /// @brief Begins dynamic rendering.
    /// @param info Rendering parameters.
    /// @return This command buffer.
    CommandBuffer& begin_rendering(const vk::RenderingInfo& info);

    /// @brief Begins dynamic rendering covering a whole extent with one layer.
    ///
    /// Convenience for the common case of rendering into full extent
    /// attachments; combine with the color_attachment/depth_attachment builders
    /// in vulcao/rendering.h. Use the vk::RenderingInfo overload for layered
    /// rendering, view masks or custom render areas.
    /// @param extent Render area, starting at (0, 0).
    /// @param colors Color attachments.
    /// @param depth Optional depth attachment.
    /// @return This command buffer.
    CommandBuffer& begin_rendering(vk::Extent2D extent,
                                   std::span<const vk::RenderingAttachmentInfo> colors,
                                   const vk::RenderingAttachmentInfo* depth = nullptr);

    /// @brief Begins dynamic rendering with a single color attachment.
    /// @param extent Render area, starting at (0, 0).
    /// @param color Color attachment.
    /// @param depth Optional depth attachment.
    /// @return This command buffer.
    CommandBuffer& begin_rendering(vk::Extent2D extent,
                                   const vk::RenderingAttachmentInfo& color,
                                   const vk::RenderingAttachmentInfo* depth = nullptr);

    /// @brief Ends dynamic rendering.
    /// @return This command buffer.
    CommandBuffer& end_rendering();

    /// @brief Binds descriptor sets.
    /// @param bind_point Pipeline bind point.
    /// @param layout Pipeline layout.
    /// @param descriptor_sets Descriptor sets to bind, starting at set 0.
    /// @param dynamic_offsets Dynamic offsets for the sets.
    /// @return This command buffer.
    CommandBuffer& bind_descriptor_sets(vk::PipelineBindPoint bind_point,
                                        vk::PipelineLayout layout,
                                        vk::ArrayProxy<const vk::DescriptorSet> descriptor_sets,
                                        vk::ArrayProxy<const uint32_t> dynamic_offsets = {});

    /// @brief Records a compute dispatch.
    /// @param group_count_x Number of groups in X.
    /// @param group_count_y Number of groups in Y.
    /// @param group_count_z Number of groups in Z.
    /// @return This command buffer.
    CommandBuffer& dispatch(uint32_t group_count_x, uint32_t group_count_y = 1, uint32_t group_count_z = 1);

    /// @brief Records an indirect compute dispatch.
    /// @param buffer Buffer holding the dispatch parameters.
    /// @param offset Byte offset of the dispatch parameters.
    /// @return This command buffer.
    CommandBuffer& dispatch_indirect(vk::Buffer buffer, vk::DeviceSize offset);

    /// @brief Builds a bottom level acceleration structure from triangle geometry.
    ///
    /// @p scratch must be a storage buffer created with eShaderDeviceAddress and
    /// hold at least the build or update scratch size of the structure.
    /// @param structure Bottom level structure to build into.
    /// @param geometries One entry per triangle geometry.
    /// @param scratch Scratch buffer.
    /// @param mode Build or update.
    /// @return This command buffer.
    /// @throws std::runtime_error if the structure is not a valid bottom level structure.
    CommandBuffer& build_acceleration_structure(
        const AccelerationStructure& structure,
        std::span<const TrianglesGeometry> geometries,
        const Buffer& scratch,
        vk::BuildAccelerationStructureModeKHR mode = vk::BuildAccelerationStructureModeKHR::eBuild);

    /// @brief Builds a bottom level acceleration structure using its own scratch buffer.
    /// @param structure Bottom level structure created with an internal scratch buffer.
    /// @param geometries One entry per triangle geometry.
    /// @param mode Build or update.
    /// @return This command buffer.
    /// @throws std::runtime_error if the structure owns no scratch buffer.
    CommandBuffer& build_acceleration_structure(
        const AccelerationStructure& structure,
        std::span<const TrianglesGeometry> geometries,
        vk::BuildAccelerationStructureModeKHR mode = vk::BuildAccelerationStructureModeKHR::eBuild);

    /// @brief Builds a top level acceleration structure from an instance buffer.
    ///
    /// @p instance_buffer holds vk::AccelerationStructureInstanceKHR entries, as
    /// produced by make_instance_buffer, and must be at least as large as the
    /// instance count the structure was created for.
    /// @param structure Top level structure to build into.
    /// @param instance_buffer Buffer of instances.
    /// @param scratch Scratch buffer.
    /// @param mode Build or update.
    /// @return This command buffer.
    /// @throws std::runtime_error if the structure is not a valid top level structure.
    CommandBuffer& build_acceleration_structure(
        const AccelerationStructure& structure,
        const Buffer& instance_buffer,
        const Buffer& scratch,
        vk::BuildAccelerationStructureModeKHR mode = vk::BuildAccelerationStructureModeKHR::eBuild);

    /// @brief Builds a top level acceleration structure using its own scratch buffer.
    /// @param structure Top level structure created with an internal scratch buffer.
    /// @param instance_buffer Buffer of instances.
    /// @param mode Build or update.
    /// @return This command buffer.
    /// @throws std::runtime_error if the structure owns no scratch buffer.
    CommandBuffer& build_acceleration_structure(
        const AccelerationStructure& structure,
        const Buffer& instance_buffer,
        vk::BuildAccelerationStructureModeKHR mode = vk::BuildAccelerationStructureModeKHR::eBuild);

    /// @brief Builds several acceleration structures in one call.
    ///
    /// The thin passthrough for callers that fill the Vulkan build info and range
    /// info themselves, for example to build mixed bottom level geometries.
    /// @param build_infos Build descriptions.
    /// @param range_infos One range pointer per build info, matching geometry order.
    /// @return This command buffer.
    /// @throws std::runtime_error if the two spans differ in size.
    CommandBuffer& build_acceleration_structures(
        std::span<const vk::AccelerationStructureBuildGeometryInfoKHR> build_infos,
        std::span<const vk::AccelerationStructureBuildRangeInfoKHR*> range_infos);

    /// @brief Resets a range of queries.
    /// @param pool Query pool.
    /// @param first_query First query to reset.
    /// @param query_count Number of queries to reset.
    /// @return This command buffer.
    CommandBuffer& reset_query_pool(vk::QueryPool pool, uint32_t first_query, uint32_t query_count);

    /// @brief Writes a timestamp query.
    /// @param pool Query pool.
    /// @param stage Pipeline stage the timestamp is written at.
    /// @param query Query index.
    /// @return This command buffer.
    CommandBuffer& write_timestamp(vk::QueryPool pool, vk::PipelineStageFlags2 stage, uint32_t query);

    /// @brief Begins a query.
    /// @param pool Query pool.
    /// @param query Query index.
    /// @param flags Query control flags.
    /// @return This command buffer.
    CommandBuffer& begin_query(vk::QueryPool pool, uint32_t query, vk::QueryControlFlags flags = {});

    /// @brief Ends a query.
    /// @param pool Query pool.
    /// @param query Query index.
    /// @return This command buffer.
    CommandBuffer& end_query(vk::QueryPool pool, uint32_t query);

    /// @brief Copies query results into a buffer.
    /// @param pool Query pool.
    /// @param first_query First query to copy.
    /// @param query_count Number of queries to copy.
    /// @param dst Destination buffer.
    /// @param dst_offset Byte offset in the destination.
    /// @param stride Byte stride between results.
    /// @param flags Query result flags.
    /// @return This command buffer.
    CommandBuffer& copy_query_pool_results(vk::QueryPool pool,
                                           uint32_t first_query,
                                           uint32_t query_count,
                                           vk::Buffer dst,
                                           vk::DeviceSize dst_offset,
                                           vk::DeviceSize stride,
                                           vk::QueryResultFlags flags);

private:
    vk::Device device_;
    vk::CommandPool pool_;
    vk::CommandBuffer cmd_;
    vk::CommandBufferLevel level_ = vk::CommandBufferLevel::ePrimary;
    PFN_vkCmdBeginDebugUtilsLabelEXT begin_label_ext_ = nullptr;
    PFN_vkCmdEndDebugUtilsLabelEXT end_label_ext_ = nullptr;
    PFN_vkCmdInsertDebugUtilsLabelEXT insert_label_ext_ = nullptr;
};

}
