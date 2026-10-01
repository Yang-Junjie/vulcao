#pragma once

#include <array>
#include <span>

#include <vulkan/vulkan.hpp>

#include "vulcao/allocator.h"
#include "vulcao/buffer.h"

namespace vulcao {

class Context;

/// @brief One triangle geometry of a bottom level acceleration structure.
///
/// The vertex and index data live in buffers created with
/// eAccelerationStructureBuildInputReadOnlyKHR and eShaderDeviceAddress; this
/// struct carries their device addresses so the structure stays independent of
/// how the geometry is stored.
struct TrianglesGeometry {
    vk::DeviceAddress vertex_data = 0; ///< Device address of the vertex buffer.
    vk::DeviceAddress index_data = 0;  ///< Device address of the index buffer, 0 when non-indexed.
    uint32_t vertex_count = 0;         ///< Number of vertices.
    uint32_t index_count = 0;          ///< Number of indices, ignored when index_data is 0.
    vk::DeviceSize vertex_stride = 0;  ///< Byte stride between vertices.
    vk::Format vertex_format = vk::Format::eR32G32B32Sfloat; ///< Vertex position format.
    vk::IndexType index_type = vk::IndexType::eUint32;       ///< Index format.
    vk::GeometryFlagsKHR flags = {};                         ///< eOpaque or eNoDuplicateAnyHitInvocation.
};

/// @brief One instance of a top level acceleration structure.
struct AccelerationStructureInstance {
    vk::AccelerationStructureKHR acceleration_structure = {}; ///< The bottom level structure.
    /// @brief Row-major 3x4 transform, matching vk::AccelerationStructureInstanceKHR.
    std::array<float, 12> transform{1.0f, 0.0f, 0.0f, 0.0f,
                                    0.0f, 1.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 1.0f, 0.0f};
    uint32_t instance_custom_index = 0;                 ///< Value visible to hit shaders.
    uint32_t mask = 0xFF;                               ///< Ray mask.
    uint32_t shader_binding_table_record_offset = 0;    ///< Hit group SBT record offset.
    vk::GeometryInstanceFlagsKHR flags = {};            ///< Instance flags.
};

/// @brief Build sizes of an acceleration structure, as reported by the driver.
struct AccelerationStructureSizes {
    vk::DeviceSize acceleration_structure_size = 0; ///< Size of the backing storage.
    vk::DeviceSize build_scratch_size = 0;          ///< Scratch size for a full build.
    vk::DeviceSize update_scratch_size = 0;         ///< Scratch size for an update.
};

/// @brief Queries the build sizes of a bottom level structure.
/// @param device Device that builds the structure.
/// @param geometries One entry per triangle geometry.
/// @param flags Build flags the structure will be built with.
/// @param mode Build mode the sizes are queried for.
/// @return Sizes of the backing storage and the scratch buffers.
/// @throws std::runtime_error if the query fails.
AccelerationStructureSizes blas_build_sizes(
    vk::Device device,
    std::span<const TrianglesGeometry> geometries,
    vk::BuildAccelerationStructureFlagsKHR flags = {},
    vk::BuildAccelerationStructureModeKHR mode = vk::BuildAccelerationStructureModeKHR::eBuild);

/// @brief Queries the build sizes of a top level structure.
/// @param device Device that builds the structure.
/// @param instance_count Number of instances the structure holds.
/// @param flags Build flags the structure will be built with.
/// @param mode Build mode the sizes are queried for.
/// @return Sizes of the backing storage and the scratch buffers.
/// @throws std::runtime_error if the query fails.
AccelerationStructureSizes tlas_build_sizes(
    vk::Device device,
    uint32_t instance_count,
    vk::BuildAccelerationStructureFlagsKHR flags = {},
    vk::BuildAccelerationStructureModeKHR mode = vk::BuildAccelerationStructureModeKHR::eBuild);

/// @brief Packs instances into a buffer usable as a top level build input.
///
/// Converts the portable @ref AccelerationStructureInstance description into the
/// driver's instance layout (looking up each bottom level device address) and
/// uploads it. The buffer carries eAccelerationStructureBuildInputReadOnlyKHR and
/// eShaderDeviceAddress, so it can be passed straight to
/// CommandBuffer::build_acceleration_structure.
/// @param context Context providing the allocator and device.
/// @param instances Instances to pack.
/// @return The instance buffer.
/// @throws std::runtime_error if a bottom level structure is invalid or upload fails.
Buffer make_instance_buffer(Context& context, std::span<const AccelerationStructureInstance> instances);

/// @brief RAII wrapper around an acceleration structure and its backing storage.
class AccelerationStructure {
public:
    /// @brief Creates an empty acceleration structure.
    AccelerationStructure() = default;

    /// @brief Destroys the structure and the buffers it owns.
    ~AccelerationStructure();

    /// @brief Not copyable.
    AccelerationStructure(const AccelerationStructure&) = delete;
    AccelerationStructure& operator=(const AccelerationStructure&) = delete;

    /// @brief Moves the structure, leaving the source empty.
    AccelerationStructure(AccelerationStructure&& other) noexcept;

    /// @brief Move assignment. Destroys the current structure first.
    AccelerationStructure& operator=(AccelerationStructure&& other) noexcept;

    /// @brief Creates an empty structure backed by a device local buffer.
    /// @param allocator Allocator used for the backing storage.
    /// @param type Bottom or top level.
    /// @param size Backing storage size, typically from a build sizes query.
    /// @return The created structure.
    /// @throws std::runtime_error if the allocator is invalid or creation fails.
    static AccelerationStructure create(Allocator& allocator,
                                        vk::AccelerationStructureTypeKHR type,
                                        vk::DeviceSize size);

    /// @brief Creates a bottom level structure sized for the geometry.
    /// @param context Context providing the allocator and device.
    /// @param geometries One entry per triangle geometry.
    /// @param flags Build flags; they are remembered and reused by the build.
    /// @param own_scratch True to also allocate an internal build scratch buffer.
    /// @return The created structure.
    static AccelerationStructure create_blas(Context& context,
                                             std::span<const TrianglesGeometry> geometries,
                                             vk::BuildAccelerationStructureFlagsKHR flags = {},
                                             bool own_scratch = false);

    /// @brief Creates a top level structure sized for @p instance_count instances.
    /// @param context Context providing the allocator and device.
    /// @param instance_count Number of instances the structure holds.
    /// @param flags Build flags; they are remembered and reused by the build.
    /// @param own_scratch True to also allocate an internal build scratch buffer.
    /// @return The created structure.
    static AccelerationStructure create_tlas(Context& context,
                                             uint32_t instance_count,
                                             vk::BuildAccelerationStructureFlagsKHR flags = {},
                                             bool own_scratch = false);

    /// @brief Returns true if the structure holds a valid handle.
    bool valid() const { return static_cast<bool>(structure_); }

    /// @brief Returns true if the structure holds a valid handle.
    explicit operator bool() const { return valid(); }

    /// @brief Returns the raw Vulkan handle.
    vk::AccelerationStructureKHR handle() const { return structure_; }

    /// @brief Returns the device address used to reference this structure from a
    ///        top level instance.
    /// @throws std::runtime_error if the structure is invalid.
    vk::DeviceAddress device_address() const;

    /// @brief Returns the structure type.
    vk::AccelerationStructureTypeKHR type() const { return type_; }

    /// @brief Returns the build flags the structure was created with.
    vk::BuildAccelerationStructureFlagsKHR build_flags() const { return build_flags_; }

    /// @brief Returns the build sizes the structure was created with.
    const AccelerationStructureSizes& sizes() const { return sizes_; }

    /// @brief Returns the instance count of a top level structure, 0 otherwise.
    uint32_t element_count() const { return element_count_; }

    /// @brief Returns the backing storage buffer.
    const Buffer& buffer() const { return buffer_; }

    /// @brief Returns the internal scratch buffer, empty when none was requested.
    const Buffer& scratch() const { return scratch_; }

    /// @brief Destroys the structure and resets the wrapper.
    void destroy();

private:
    void create_internal(Allocator& allocator,
                         vk::AccelerationStructureTypeKHR type,
                         vk::DeviceSize size);

    vk::Device device_;
    Buffer buffer_;
    Buffer scratch_;
    vk::AccelerationStructureKHR structure_;
    vk::AccelerationStructureTypeKHR type_ = vk::AccelerationStructureTypeKHR::eBottomLevel;
    vk::BuildAccelerationStructureFlagsKHR build_flags_;
    AccelerationStructureSizes sizes_{};
    uint32_t element_count_ = 0;
};

}
