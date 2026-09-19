#pragma once

#include <cstdint>
#include <map>
#include <vector>

#include <vulkan/vulkan.hpp>

#include "vulcao/reflection.h"

namespace vulcao {

class Buffer;
class BufferView;
class DescriptorSet;
class DescriptorSetWriter;
class Image;
class Sampler;

/// @brief RAII wrapper around a Vulkan descriptor set layout.
class DescriptorSetLayout {
public:
    /// @brief Creates an empty descriptor set layout.
    DescriptorSetLayout() = default;

    /// @brief Destroys the descriptor set layout.
    ~DescriptorSetLayout();

    /// @brief Not copyable.
    DescriptorSetLayout(const DescriptorSetLayout&) = delete;
    DescriptorSetLayout& operator=(const DescriptorSetLayout&) = delete;

    /// @brief Moves the descriptor set layout, leaving the source empty.
    DescriptorSetLayout(DescriptorSetLayout&& other) noexcept;

    /// @brief Move assignment. Destroys the current layout first.
    DescriptorSetLayout& operator=(DescriptorSetLayout&& other) noexcept;

    /// @brief Creates a descriptor set layout from bindings.
    ///
    /// This is the entry point for descriptor indexing (bindless-style) usage:
    /// pass vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool in
    /// @p flags and per-binding flags such as ePartiallyBound or
    /// eVariableDescriptorCount in @p binding_flags, then allocate from a pool
    /// created with vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind. These
    /// flags require the descriptor indexing device feature.
    /// @param device Device that creates the layout.
    /// @param bindings Bindings of the layout. descriptorCount must be non-zero;
    ///        give reflected runtime arrays a concrete upper bound first.
    /// @param flags Layout creation flags.
    /// @param binding_flags Per-binding flags, empty or one entry per binding.
    /// @return The created descriptor set layout.
    /// @throws std::runtime_error if a descriptorCount is zero or binding_flags
    ///         is neither empty nor one entry per binding.
    static DescriptorSetLayout create(vk::Device device,
                                      vk::ArrayProxy<const vk::DescriptorSetLayoutBinding> bindings,
                                      vk::DescriptorSetLayoutCreateFlags flags = {},
                                      vk::ArrayProxy<const vk::DescriptorBindingFlags> binding_flags = {});

    /// @brief Creates a descriptor set layout from reflected shader bindings.
    /// @param device Device that creates the layout.
    /// @param reflection Shader reflection data.
    /// @param set Descriptor set index to create the layout for.
    /// @return The created descriptor set layout.
    /// @throws std::runtime_error if the set contains a runtime array; set a
    ///         concrete count with set_binding_count and use the bindings overload.
    static DescriptorSetLayout create(vk::Device device,
                                      const ShaderReflection& reflection,
                                      uint32_t set);

    /// @brief Returns true if the layout holds a valid handle.
    bool valid() const { return static_cast<bool>(layout_); }

    /// @brief Returns true if the layout holds a valid handle.
    explicit operator bool() const { return valid(); }

    /// @brief Returns the raw Vulkan descriptor set layout handle.
    vk::DescriptorSetLayout handle() const { return layout_; }

    /// @brief Destroys the layout and resets the wrapper.
    void destroy();

private:
    vk::Device device_;
    vk::DescriptorSetLayout layout_;
};

/// @brief RAII wrapper around a Vulkan descriptor pool.
class DescriptorPool {
public:
    /// @brief Creates an empty descriptor pool.
    DescriptorPool() = default;

    /// @brief Destroys the descriptor pool and all sets allocated from it.
    ~DescriptorPool();

    /// @brief Not copyable.
    DescriptorPool(const DescriptorPool&) = delete;
    DescriptorPool& operator=(const DescriptorPool&) = delete;

    /// @brief Moves the descriptor pool, leaving the source empty.
    DescriptorPool(DescriptorPool&& other) noexcept;

    /// @brief Move assignment. Destroys the current pool first.
    DescriptorPool& operator=(DescriptorPool&& other) noexcept;

    /// @brief Creates a descriptor pool.
    ///
    /// Unless @p flags includes eFreeDescriptorSet, the pool tracks the number
    /// of live sets on the CPU and allocate() enforces @p max_sets even on
    /// drivers that let the limit slip (lavapipe does).
    /// @param device Device that creates the pool.
    /// @param sizes Number of descriptors of each type.
    /// @param max_sets Maximum number of sets that can be allocated.
    /// @param flags Pool creation flags.
    /// @return The created descriptor pool.
    static DescriptorPool create(vk::Device device,
                                 vk::ArrayProxy<const vk::DescriptorPoolSize> sizes,
                                 uint32_t max_sets,
                                 vk::DescriptorPoolCreateFlags flags = {});

    /// @brief Creates a pool sized for a number of sets with the given bindings.
    ///
    /// The pool sizes are computed from the bindings: every descriptor type gets
    /// its count summed over the bindings and multiplied by @p set_count, which
    /// also becomes the maximum number of sets. Combine with reflection
    /// (ShaderReflection::bindings_for_set) to size a pool without counting
    /// descriptors by hand, e.g. set_count = frames in flight.
    /// @param device Device that creates the pool.
    /// @param bindings Bindings of one set.
    /// @param set_count Number of sets that must be allocatable, must be non-zero.
    /// @param flags Pool creation flags.
    /// @return The created descriptor pool.
    /// @throws std::runtime_error if set_count is zero.
    static DescriptorPool create_for_bindings(vk::Device device,
                                              vk::ArrayProxy<const vk::DescriptorSetLayoutBinding> bindings,
                                              uint32_t set_count,
                                              vk::DescriptorPoolCreateFlags flags = {});

    /// @brief Returns true if the pool holds a valid handle.
    bool valid() const { return static_cast<bool>(pool_); }

    /// @brief Returns true if the pool holds a valid handle.
    explicit operator bool() const { return valid(); }

    /// @brief Returns the raw Vulkan descriptor pool handle.
    vk::DescriptorPool handle() const { return pool_; }

    /// @brief Returns the number of sets allocated since the last reset.
    uint32_t allocated_sets() const { return allocated_sets_; }

    /// @brief Allocates one descriptor set. The pool must outlive the returned set.
    /// @param layout Layout of the set.
    /// @param variable_descriptor_count When the layout's last binding has the
    ///        eVariableDescriptorCount flag, the number of descriptors allocated
    ///        for it. 0 to allocate the count declared by the layout.
    /// @return The allocated descriptor set.
    /// @throws std::runtime_error if the layout is invalid, the pool's set
    ///         limit is reached (tracked on the CPU for pools without
    ///         eFreeDescriptorSet) or the driver rejects the allocation.
    DescriptorSet allocate(const DescriptorSetLayout& layout, uint32_t variable_descriptor_count = 0);

    /// @brief Allocates one descriptor set from a raw layout handle.
    /// @param layout Layout of the set.
    /// @param variable_descriptor_count When the layout's last binding has the
    ///        eVariableDescriptorCount flag, the number of descriptors allocated
    ///        for it. 0 to allocate the count declared by the layout.
    /// @return The allocated descriptor set.
    /// @throws std::runtime_error if the layout is invalid, the pool's set
    ///         limit is reached (tracked on the CPU for pools without
    ///         eFreeDescriptorSet) or the driver rejects the allocation.
    DescriptorSet allocate(vk::DescriptorSetLayout layout, uint32_t variable_descriptor_count = 0);

    /// @brief Resets the pool, frees all sets allocated from it and clears the
    ///        tracked set count.
    /// @param flags Reset flags.
    void reset(vk::DescriptorPoolResetFlags flags = {});

    /// @brief Destroys the pool and resets the wrapper.
    void destroy();

private:
    vk::Device device_;
    vk::DescriptorPool pool_;
    /// @brief True while set counting is valid: without eFreeDescriptorSet,
    ///        sets are only reclaimed by reset(), so a counter cannot drift.
    bool track_sets_ = false;
    uint32_t max_sets_ = 0;
    uint32_t allocated_sets_ = 0;
};

/// @brief Non-owning handle to a Vulkan descriptor set with writing helpers.
///
/// The write helpers are const: they update the Vulkan descriptor set, not this
/// handle, so a const DescriptorSet can still be written to.
class DescriptorSet {
public:
    /// @brief Creates an empty descriptor set handle.
    DescriptorSet() = default;

    /// @brief Returns true if the handle is valid.
    bool valid() const { return static_cast<bool>(set_); }

    /// @brief Returns true if the handle is valid.
    explicit operator bool() const { return valid(); }

    /// @brief Returns the raw Vulkan descriptor set handle.
    vk::DescriptorSet handle() const { return set_; }

    /// @brief Writes a buffer descriptor with an explicit descriptor type.
    /// @param binding Binding index.
    /// @param buffer Buffer to bind.
    /// @param type Descriptor type.
    /// @param offset Byte offset in the buffer.
    /// @param range Byte size of the binding, or VK_WHOLE_SIZE.
    /// @return This descriptor set.
    const DescriptorSet& write_buffer(uint32_t binding,
                                const Buffer& buffer,
                                vk::DescriptorType type,
                                vk::DeviceSize offset = 0,
                                vk::DeviceSize range = VK_WHOLE_SIZE) const;

    /// @brief Writes a uniform buffer descriptor.
    /// @param binding Binding index.
    /// @param buffer Buffer to bind.
    /// @param offset Byte offset in the buffer.
    /// @param range Byte size of the binding, or VK_WHOLE_SIZE.
    /// @return This descriptor set.
    const DescriptorSet& write_uniform_buffer(uint32_t binding,
                                        const Buffer& buffer,
                                        vk::DeviceSize offset = 0,
                                        vk::DeviceSize range = VK_WHOLE_SIZE) const;

    /// @brief Writes a storage buffer descriptor.
    /// @param binding Binding index.
    /// @param buffer Buffer to bind.
    /// @param offset Byte offset in the buffer.
    /// @param range Byte size of the binding, or VK_WHOLE_SIZE.
    /// @return This descriptor set.
    const DescriptorSet& write_storage_buffer(uint32_t binding,
                                        const Buffer& buffer,
                                        vk::DeviceSize offset = 0,
                                        vk::DeviceSize range = VK_WHOLE_SIZE) const;

    /// @brief Writes a combined image sampler descriptor.
    /// @param binding Binding index.
    /// @param image Image to bind.
    /// @param sampler Sampler to bind.
    /// @param layout Layout the image is sampled in.
    /// @return This descriptor set.
    const DescriptorSet& write_image(uint32_t binding,
                               const Image& image,
                               const Sampler& sampler,
                               vk::ImageLayout layout = vk::ImageLayout::eShaderReadOnlyOptimal) const;

    /// @brief Writes a storage image descriptor.
    /// @param binding Binding index.
    /// @param image Image to bind.
    /// @param layout Layout the image is accessed in.
    /// @return This descriptor set.
    const DescriptorSet& write_storage_image(uint32_t binding,
                                       const Image& image,
                                       vk::ImageLayout layout = vk::ImageLayout::eGeneral) const;

    /// @brief Writes a standalone sampler descriptor.
    /// @param binding Binding index.
    /// @param sampler Sampler to bind.
    /// @return This descriptor set.
    const DescriptorSet& write_sampler(uint32_t binding, const Sampler& sampler) const;

    /// @brief Writes a sampled image descriptor, separate from its sampler.
    /// @param binding Binding index.
    /// @param image Image to bind.
    /// @param layout Layout the image is sampled in.
    /// @return This descriptor set.
    const DescriptorSet& write_sampled_image(
        uint32_t binding,
        const Image& image,
        vk::ImageLayout layout = vk::ImageLayout::eShaderReadOnlyOptimal) const;

    /// @brief Writes an input attachment descriptor.
    /// @param binding Binding index.
    /// @param image Image to bind.
    /// @param layout Layout the image is read in.
    /// @return This descriptor set.
    const DescriptorSet& write_input_attachment(
        uint32_t binding,
        const Image& image,
        vk::ImageLayout layout = vk::ImageLayout::eShaderReadOnlyOptimal) const;

    /// @brief Writes a uniform texel buffer descriptor.
    /// @param binding Binding index.
    /// @param view Texel buffer view to bind.
    /// @return This descriptor set.
    const DescriptorSet& write_uniform_texel_buffer(uint32_t binding, const BufferView& view) const;

    /// @brief Writes a storage texel buffer descriptor.
    /// @param binding Binding index.
    /// @param view Texel buffer view to bind.
    /// @return This descriptor set.
    const DescriptorSet& write_storage_texel_buffer(uint32_t binding, const BufferView& view) const;

private:
    friend class DescriptorPool;
    friend class DescriptorSetWriter;

    /// @brief Writes an image-info based descriptor of an arbitrary type.
    const DescriptorSet& write_image_descriptor(uint32_t binding,
                                                const vk::DescriptorImageInfo& info,
                                                vk::DescriptorType type) const;

    /// @brief Writes a texel buffer descriptor of an arbitrary type.
    const DescriptorSet& write_texel_buffer(uint32_t binding,
                                            const BufferView& view,
                                            vk::DescriptorType type) const;

    vk::Device device_;
    vk::DescriptorSet set_;
};

/// @brief Batches descriptor writes and flushes them with a single update.
class DescriptorSetWriter {
public:
    /// @brief Creates a writer targeting a descriptor set.
    /// @param set Descriptor set to write to.
    explicit DescriptorSetWriter(const DescriptorSet& set);

    /// @brief Queues a buffer descriptor write.
    /// @param binding Binding index.
    /// @param buffer Buffer to bind.
    /// @param type Descriptor type.
    /// @param array_element First array element to write.
    /// @param offset Byte offset in the buffer.
    /// @param range Byte size of the binding, or VK_WHOLE_SIZE.
    /// @return This writer.
    DescriptorSetWriter& write_buffer(uint32_t binding,
                                      const Buffer& buffer,
                                      vk::DescriptorType type,
                                      uint32_t array_element = 0,
                                      vk::DeviceSize offset = 0,
                                      vk::DeviceSize range = VK_WHOLE_SIZE);

    /// @brief Queues a uniform buffer descriptor write.
    DescriptorSetWriter& write_uniform_buffer(uint32_t binding,
                                              const Buffer& buffer,
                                              uint32_t array_element = 0,
                                              vk::DeviceSize offset = 0,
                                              vk::DeviceSize range = VK_WHOLE_SIZE);

    /// @brief Queues a storage buffer descriptor write.
    DescriptorSetWriter& write_storage_buffer(uint32_t binding,
                                              const Buffer& buffer,
                                              uint32_t array_element = 0,
                                              vk::DeviceSize offset = 0,
                                              vk::DeviceSize range = VK_WHOLE_SIZE);

    /// @brief Queues a combined image sampler descriptor write.
    DescriptorSetWriter& write_image(uint32_t binding,
                                     const Image& image,
                                     const Sampler& sampler,
                                     vk::ImageLayout layout = vk::ImageLayout::eShaderReadOnlyOptimal,
                                     uint32_t array_element = 0);

    /// @brief Queues a storage image descriptor write.
    DescriptorSetWriter& write_storage_image(uint32_t binding,
                                             const Image& image,
                                             vk::ImageLayout layout = vk::ImageLayout::eGeneral,
                                             uint32_t array_element = 0);

    /// @brief Queues a standalone sampler descriptor write.
    DescriptorSetWriter& write_sampler(uint32_t binding,
                                       const Sampler& sampler,
                                       uint32_t array_element = 0);

    /// @brief Queues a sampled image descriptor write, separate from its sampler.
    DescriptorSetWriter& write_sampled_image(
        uint32_t binding,
        const Image& image,
        vk::ImageLayout layout = vk::ImageLayout::eShaderReadOnlyOptimal,
        uint32_t array_element = 0);

    /// @brief Queues an input attachment descriptor write.
    DescriptorSetWriter& write_input_attachment(
        uint32_t binding,
        const Image& image,
        vk::ImageLayout layout = vk::ImageLayout::eShaderReadOnlyOptimal,
        uint32_t array_element = 0);

    /// @brief Queues a uniform texel buffer descriptor write.
    DescriptorSetWriter& write_uniform_texel_buffer(uint32_t binding,
                                                    const BufferView& view,
                                                    uint32_t array_element = 0);

    /// @brief Queues a storage texel buffer descriptor write.
    DescriptorSetWriter& write_storage_texel_buffer(uint32_t binding,
                                                    const BufferView& view,
                                                    uint32_t array_element = 0);

    /// @brief Applies all queued writes in one update and clears them.
    void flush();

    /// @brief Discards all queued writes.
    void clear();

private:
    struct Record {
        enum class Payload { buffer, image, texel_buffer };

        uint32_t binding = 0;
        uint32_t array_element = 0;
        vk::DescriptorType type = vk::DescriptorType::eUniformBuffer;
        Payload payload = Payload::buffer;
        vk::DescriptorBufferInfo buffer_info{};
        vk::DescriptorImageInfo image_info{};
        vk::BufferView texel_view{};
    };

    /// @brief Queues an image-info based descriptor write of an arbitrary type.
    DescriptorSetWriter& write_image_record(uint32_t binding,
                                            vk::DescriptorType type,
                                            const vk::DescriptorImageInfo& info,
                                            uint32_t array_element);

    /// @brief Queues a texel buffer descriptor write of an arbitrary type.
    DescriptorSetWriter& write_texel_buffer(uint32_t binding,
                                            const BufferView& view,
                                            vk::DescriptorType type,
                                            uint32_t array_element);

    vk::Device device_;
    vk::DescriptorSet set_;
    std::vector<Record> records_;
};

/// @brief Caches descriptor set layouts created from identical bindings.
///
/// The key covers the creation flags, and per binding the binding number,
/// descriptor type and count, stage flags, descriptor binding flags and any
/// immutable samplers, so layouts that differ in any of those do not collide.
class DescriptorSetLayoutCache {
public:
    /// @brief Creates a cache bound to a device.
    /// @param device Device that creates the layouts.
    explicit DescriptorSetLayoutCache(vk::Device device);

    /// @brief Returns a cached layout for the bindings, creating it on first use.
    /// @param bindings Bindings of the layout.
    /// @param flags Layout creation flags, part of the cache key.
    /// @param binding_flags Per-binding flags, empty or one entry per binding.
    /// @return The descriptor set layout handle.
    vk::DescriptorSetLayout get(vk::ArrayProxy<const vk::DescriptorSetLayoutBinding> bindings,
                                vk::DescriptorSetLayoutCreateFlags flags = {},
                                vk::ArrayProxy<const vk::DescriptorBindingFlags> binding_flags = {});

    /// @brief Destroys all cached layouts.
    void clear();

private:
    struct Key {
        struct Entry {
            uint32_t binding = 0;
            vk::DescriptorType type = vk::DescriptorType::eUniformBuffer;
            uint32_t count = 0;
            vk::ShaderStageFlags stages;
            /// @brief Descriptor binding flags (ePartiallyBound and friends).
            vk::DescriptorBindingFlags flags;
            /// @brief Samplers baked into the binding, empty when none are immutable.
            std::vector<vk::Sampler> immutable_samplers;

            bool operator<(const Entry& other) const;
        };

        vk::DescriptorSetLayoutCreateFlags create_flags;
        std::vector<Entry> entries;
        bool operator<(const Key& other) const;
    };

    vk::Device device_;
    std::map<Key, DescriptorSetLayout> cache_;
};

}
