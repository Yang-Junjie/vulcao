#include "vulcao/acceleration_structure.h"

#include "vulcao/check.h"
#include "vulcao/command_buffer.h"
#include "vulcao/context.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vulcao {
namespace {

// The acceleration structure commands are extension-only, so the loader does not
// export them as linkable symbols on every platform and the Vulkan-Hpp static
// dispatcher cannot reach them. Resolve them per device through vkGetDeviceProcAddr,
// as the debug utils entry points already are.
template <typename Function>
Function load_device_function(vk::Device device, const char* name) {
    const auto function = reinterpret_cast<Function>(device.getProcAddr(name));
    if (function == nullptr)
        throw std::runtime_error(std::string("vulcao: missing device entry point ") + name);
    return function;
}

PFN_vkCreateAccelerationStructureKHR create_acceleration_structure_fn(vk::Device device) {
    return load_device_function<PFN_vkCreateAccelerationStructureKHR>(
        device, "vkCreateAccelerationStructureKHR");
}

PFN_vkDestroyAccelerationStructureKHR destroy_acceleration_structure_fn(vk::Device device) {
    return load_device_function<PFN_vkDestroyAccelerationStructureKHR>(
        device, "vkDestroyAccelerationStructureKHR");
}

PFN_vkGetAccelerationStructureDeviceAddressKHR acceleration_structure_address_fn(vk::Device device) {
    return load_device_function<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
        device, "vkGetAccelerationStructureDeviceAddressKHR");
}

PFN_vkGetAccelerationStructureBuildSizesKHR build_sizes_fn(vk::Device device) {
    return load_device_function<PFN_vkGetAccelerationStructureBuildSizesKHR>(
        device, "vkGetAccelerationStructureBuildSizesKHR");
}

PFN_vkCmdBuildAccelerationStructuresKHR build_acceleration_structures_fn(vk::Device device) {
    return load_device_function<PFN_vkCmdBuildAccelerationStructuresKHR>(
        device, "vkCmdBuildAccelerationStructuresKHR");
}

/// @brief Vulkan geometry descriptions derived from the portable triangle list.
struct BlasGeometry {
    std::vector<vk::AccelerationStructureGeometryKHR> geometries;
    std::vector<vk::AccelerationStructureBuildRangeInfoKHR> ranges;
    std::vector<uint32_t> primitive_counts;
};

BlasGeometry make_blas_geometry(std::span<const TrianglesGeometry> input) {
    BlasGeometry geometry;
    geometry.geometries.reserve(input.size());
    geometry.ranges.reserve(input.size());
    geometry.primitive_counts.reserve(input.size());

    for (const TrianglesGeometry& triangles : input) {
        vk::DeviceOrHostAddressConstKHR vertex_data;
        vertex_data.deviceAddress = triangles.vertex_data;
        vk::DeviceOrHostAddressConstKHR index_data;
        index_data.deviceAddress = triangles.index_data;

        const vk::AccelerationStructureGeometryTrianglesDataKHR triangles_data{
            .vertexFormat = triangles.vertex_format,
            .vertexData = vertex_data,
            .vertexStride = triangles.vertex_stride,
            .maxVertex = triangles.vertex_count == 0 ? 0u : triangles.vertex_count - 1,
            .indexType = triangles.index_data != 0 ? triangles.index_type : vk::IndexType::eNoneKHR,
            .indexData = index_data,
        };

        vk::AccelerationStructureGeometryDataKHR geometry_data;
        geometry_data.triangles = triangles_data;
        geometry.geometries.push_back(vk::AccelerationStructureGeometryKHR{
            .geometryType = vk::GeometryTypeKHR::eTriangles,
            .geometry = geometry_data,
            .flags = triangles.flags,
        });

        const uint32_t primitive_count =
            triangles.index_data != 0 ? triangles.index_count / 3 : triangles.vertex_count / 3;
        geometry.ranges.push_back(
            vk::AccelerationStructureBuildRangeInfoKHR{.primitiveCount = primitive_count});
        geometry.primitive_counts.push_back(primitive_count);
    }

    return geometry;
}

vk::AccelerationStructureBuildGeometryInfoKHR instance_build_info(
    vk::AccelerationStructureTypeKHR type,
    vk::BuildAccelerationStructureFlagsKHR flags,
    vk::BuildAccelerationStructureModeKHR mode,
    vk::AccelerationStructureKHR source,
    vk::AccelerationStructureKHR destination,
    uint32_t geometry_count,
    const vk::AccelerationStructureGeometryKHR* geometries,
    vk::DeviceAddress scratch) {
    vk::DeviceOrHostAddressKHR scratch_data;
    scratch_data.deviceAddress = scratch;

    return vk::AccelerationStructureBuildGeometryInfoKHR{
        .type = type,
        .flags = flags,
        .mode = mode,
        .srcAccelerationStructure = source,
        .dstAccelerationStructure = destination,
        .geometryCount = geometry_count,
        .pGeometries = geometries,
        .scratchData = scratch_data,
    };
}

}

AccelerationStructureSizes blas_build_sizes(vk::Device device,
                                            std::span<const TrianglesGeometry> geometries,
                                            vk::BuildAccelerationStructureFlagsKHR flags,
                                            vk::BuildAccelerationStructureModeKHR mode) {
    if (geometries.empty())
        throw std::runtime_error("blas_build_sizes: the geometry list is empty");

    const BlasGeometry geometry = make_blas_geometry(geometries);
    const vk::AccelerationStructureBuildGeometryInfoKHR build_info{
        .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
        .flags = flags,
        .mode = mode,
        .geometryCount = static_cast<uint32_t>(geometry.geometries.size()),
        .pGeometries = geometry.geometries.data(),
    };

    VkAccelerationStructureBuildSizesInfoKHR sizes{};
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    build_sizes_fn(device)(
        device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        reinterpret_cast<const VkAccelerationStructureBuildGeometryInfoKHR*>(&build_info),
        geometry.primitive_counts.data(), &sizes);
    return AccelerationStructureSizes{
        .acceleration_structure_size = sizes.accelerationStructureSize,
        .build_scratch_size = sizes.buildScratchSize,
        .update_scratch_size = sizes.updateScratchSize,
    };
}

AccelerationStructureSizes tlas_build_sizes(vk::Device device,
                                            uint32_t instance_count,
                                            vk::BuildAccelerationStructureFlagsKHR flags,
                                            vk::BuildAccelerationStructureModeKHR mode) {
    if (instance_count == 0)
        throw std::runtime_error("tlas_build_sizes: instance_count must be non-zero");

    vk::DeviceOrHostAddressConstKHR instance_data;
    instance_data.deviceAddress = 0;
    const vk::AccelerationStructureGeometryInstancesDataKHR instances{
        .arrayOfPointers = VK_FALSE,
        .data = instance_data,
    };
    vk::AccelerationStructureGeometryDataKHR geometry_data;
    geometry_data.instances = instances;
    const vk::AccelerationStructureGeometryKHR geometry{
        .geometryType = vk::GeometryTypeKHR::eInstances,
        .geometry = geometry_data,
        .flags = {},
    };
    const vk::AccelerationStructureBuildGeometryInfoKHR build_info{
        .type = vk::AccelerationStructureTypeKHR::eTopLevel,
        .flags = flags,
        .mode = mode,
        .geometryCount = 1,
        .pGeometries = &geometry,
    };

    const std::array<uint32_t, 1> max_primitive_counts{instance_count};
    VkAccelerationStructureBuildSizesInfoKHR sizes{};
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    build_sizes_fn(device)(
        device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        reinterpret_cast<const VkAccelerationStructureBuildGeometryInfoKHR*>(&build_info),
        max_primitive_counts.data(), &sizes);
    return AccelerationStructureSizes{
        .acceleration_structure_size = sizes.accelerationStructureSize,
        .build_scratch_size = sizes.buildScratchSize,
        .update_scratch_size = sizes.updateScratchSize,
    };
}

Buffer make_instance_buffer(Context& context,
                            std::span<const AccelerationStructureInstance> instances) {
    std::vector<vk::AccelerationStructureInstanceKHR> packed;
    packed.reserve(instances.size());

    for (const AccelerationStructureInstance& instance : instances) {
        if (!instance.acceleration_structure)
            throw std::runtime_error(
                "make_instance_buffer: an instance references an invalid acceleration structure");

        vk::AccelerationStructureInstanceKHR vk_instance{};
        for (int row = 0; row < 3; ++row)
            for (int column = 0; column < 4; ++column)
                vk_instance.transform.matrix[row][column] = instance.transform[row * 4 + column];
        vk_instance.instanceCustomIndex = instance.instance_custom_index;
        vk_instance.mask = instance.mask;
        vk_instance.instanceShaderBindingTableRecordOffset = instance.shader_binding_table_record_offset;
        vk_instance.flags = static_cast<VkGeometryInstanceFlagsKHR>(instance.flags);
        const VkAccelerationStructureDeviceAddressInfoKHR address_info{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
            .accelerationStructure =
                static_cast<VkAccelerationStructureKHR>(instance.acceleration_structure),
        };
        vk_instance.accelerationStructureReference = acceleration_structure_address_fn(
            context.device())(context.device(), &address_info);
        packed.push_back(vk_instance);
    }

    return Buffer::create_with_data(
        context, packed,
        vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
            vk::BufferUsageFlagBits::eShaderDeviceAddress |
            vk::BufferUsageFlagBits::eStorageBuffer);
}

AccelerationStructure::~AccelerationStructure() {
    destroy();
}

AccelerationStructure::AccelerationStructure(AccelerationStructure&& other) noexcept
    : device_(std::exchange(other.device_, vk::Device{})),
      buffer_(std::move(other.buffer_)),
      scratch_(std::move(other.scratch_)),
      structure_(std::exchange(other.structure_, vk::AccelerationStructureKHR{})),
      type_(std::exchange(other.type_, vk::AccelerationStructureTypeKHR::eBottomLevel)),
      build_flags_(std::exchange(other.build_flags_, vk::BuildAccelerationStructureFlagsKHR{})),
      sizes_(std::exchange(other.sizes_, AccelerationStructureSizes{})),
      element_count_(std::exchange(other.element_count_, 0u)) {}

AccelerationStructure& AccelerationStructure::operator=(AccelerationStructure&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = std::exchange(other.device_, vk::Device{});
        buffer_ = std::move(other.buffer_);
        scratch_ = std::move(other.scratch_);
        structure_ = std::exchange(other.structure_, vk::AccelerationStructureKHR{});
        type_ = std::exchange(other.type_, vk::AccelerationStructureTypeKHR::eBottomLevel);
        build_flags_ =
            std::exchange(other.build_flags_, vk::BuildAccelerationStructureFlagsKHR{});
        sizes_ = std::exchange(other.sizes_, AccelerationStructureSizes{});
        element_count_ = std::exchange(other.element_count_, 0u);
    }
    return *this;
}

void AccelerationStructure::create_internal(Allocator& allocator,
                                            vk::AccelerationStructureTypeKHR type,
                                            vk::DeviceSize size) {
    device_ = allocator.device();
    type_ = type;
    buffer_ = Buffer::create(allocator, size,
                             vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
                                 vk::BufferUsageFlagBits::eShaderDeviceAddress,
                             VMA_MEMORY_USAGE_AUTO);
    const VkAccelerationStructureCreateInfoKHR create_info{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
        .buffer = static_cast<VkBuffer>(buffer_.handle()),
        .offset = 0,
        .size = size,
        .type = static_cast<VkAccelerationStructureTypeKHR>(type),
    };
    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    check(static_cast<vk::Result>(
              create_acceleration_structure_fn(device_)(device_, &create_info, nullptr, &handle)),
          "create acceleration structure");
    structure_ = vk::AccelerationStructureKHR{handle};
}

AccelerationStructure AccelerationStructure::create(Allocator& allocator,
                                                    vk::AccelerationStructureTypeKHR type,
                                                    vk::DeviceSize size) {
    if (!allocator.valid())
        throw std::runtime_error("AccelerationStructure::create: invalid allocator");
    if (size == 0)
        throw std::runtime_error("AccelerationStructure::create: size must be non-zero");

    AccelerationStructure result;
    result.create_internal(allocator, type, size);
    return result;
}

AccelerationStructure AccelerationStructure::create_blas(
    Context& context,
    std::span<const TrianglesGeometry> geometries,
    vk::BuildAccelerationStructureFlagsKHR flags,
    bool own_scratch) {
    const AccelerationStructureSizes sizes =
        blas_build_sizes(context.device(), geometries, flags, vk::BuildAccelerationStructureModeKHR::eBuild);
    AccelerationStructure result =
        create(context.allocator(), vk::AccelerationStructureTypeKHR::eBottomLevel,
               sizes.acceleration_structure_size);
    result.build_flags_ = flags;
    result.sizes_ = sizes;

    const vk::DeviceSize scratch_size =
        std::max(sizes.build_scratch_size, sizes.update_scratch_size);
    if (own_scratch && scratch_size > 0)
        result.scratch_ = Buffer::create(
            context.allocator(), scratch_size,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            VMA_MEMORY_USAGE_AUTO);
    return result;
}

AccelerationStructure AccelerationStructure::create_tlas(
    Context& context,
    uint32_t instance_count,
    vk::BuildAccelerationStructureFlagsKHR flags,
    bool own_scratch) {
    const AccelerationStructureSizes sizes =
        tlas_build_sizes(context.device(), instance_count, flags,
                         vk::BuildAccelerationStructureModeKHR::eBuild);
    AccelerationStructure result =
        create(context.allocator(), vk::AccelerationStructureTypeKHR::eTopLevel,
               sizes.acceleration_structure_size);
    result.build_flags_ = flags;
    result.sizes_ = sizes;
    result.element_count_ = instance_count;

    const vk::DeviceSize scratch_size =
        std::max(sizes.build_scratch_size, sizes.update_scratch_size);
    if (own_scratch && scratch_size > 0)
        result.scratch_ = Buffer::create(
            context.allocator(), scratch_size,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            VMA_MEMORY_USAGE_AUTO);
    return result;
}

vk::DeviceAddress AccelerationStructure::device_address() const {
    if (!structure_)
        throw std::runtime_error("AccelerationStructure::device_address: invalid structure");

    const VkAccelerationStructureDeviceAddressInfoKHR info{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
        .accelerationStructure = static_cast<VkAccelerationStructureKHR>(structure_),
    };
    return static_cast<vk::DeviceAddress>(acceleration_structure_address_fn(device_)(device_, &info));
}

void AccelerationStructure::destroy() {
    if (structure_)
        destroy_acceleration_structure_fn(device_)(device_,
                                                   static_cast<VkAccelerationStructureKHR>(structure_),
                                                   nullptr);

    scratch_ = Buffer{};
    buffer_ = Buffer{};
    structure_ = nullptr;
    device_ = nullptr;
    type_ = vk::AccelerationStructureTypeKHR::eBottomLevel;
    build_flags_ = {};
    sizes_ = {};
    element_count_ = 0;
}

CommandBuffer& CommandBuffer::build_acceleration_structure(
    const AccelerationStructure& structure,
    std::span<const TrianglesGeometry> geometries,
    const Buffer& scratch,
    vk::BuildAccelerationStructureModeKHR mode) {
    if (!structure.valid())
        throw std::runtime_error(
            "CommandBuffer::build_acceleration_structure: invalid acceleration structure");
    if (structure.type() != vk::AccelerationStructureTypeKHR::eBottomLevel)
        throw std::runtime_error(
            "CommandBuffer::build_acceleration_structure: the geometry overload builds a bottom "
            "level structure");

    const BlasGeometry geometry = make_blas_geometry(geometries);
    if (geometry.geometries.empty())
        throw std::runtime_error(
            "CommandBuffer::build_acceleration_structure: the geometry list is empty");

    const vk::AccelerationStructureBuildGeometryInfoKHR build_info = instance_build_info(
        vk::AccelerationStructureTypeKHR::eBottomLevel, structure.build_flags(), mode,
        mode == vk::BuildAccelerationStructureModeKHR::eUpdate ? structure.handle()
                                                               : vk::AccelerationStructureKHR{},
        structure.handle(), static_cast<uint32_t>(geometry.geometries.size()),
        geometry.geometries.data(), scratch.device_address());

    const VkAccelerationStructureBuildRangeInfoKHR* ranges =
        reinterpret_cast<const VkAccelerationStructureBuildRangeInfoKHR*>(geometry.ranges.data());
    build_acceleration_structures_fn(device_)(
        static_cast<VkCommandBuffer>(cmd_), 1,
        reinterpret_cast<const VkAccelerationStructureBuildGeometryInfoKHR*>(&build_info), &ranges);
    return *this;
}

CommandBuffer& CommandBuffer::build_acceleration_structure(
    const AccelerationStructure& structure,
    std::span<const TrianglesGeometry> geometries,
    vk::BuildAccelerationStructureModeKHR mode) {
    if (!structure.scratch().valid())
        throw std::runtime_error(
            "CommandBuffer::build_acceleration_structure: the structure owns no scratch buffer; "
            "create it with own_scratch or pass one explicitly");
    return build_acceleration_structure(structure, geometries, structure.scratch(), mode);
}

CommandBuffer& CommandBuffer::build_acceleration_structure(
    const AccelerationStructure& structure,
    const Buffer& instance_buffer,
    const Buffer& scratch,
    vk::BuildAccelerationStructureModeKHR mode) {
    if (!structure.valid())
        throw std::runtime_error(
            "CommandBuffer::build_acceleration_structure: invalid acceleration structure");
    if (structure.type() != vk::AccelerationStructureTypeKHR::eTopLevel)
        throw std::runtime_error(
            "CommandBuffer::build_acceleration_structure: the instance overload builds a top "
            "level structure");

    vk::DeviceOrHostAddressConstKHR instance_data;
    instance_data.deviceAddress = instance_buffer.device_address();
    const vk::AccelerationStructureGeometryInstancesDataKHR instances{
        .arrayOfPointers = VK_FALSE,
        .data = instance_data,
    };
    vk::AccelerationStructureGeometryDataKHR geometry_data;
    geometry_data.instances = instances;
    const vk::AccelerationStructureGeometryKHR geometry{
        .geometryType = vk::GeometryTypeKHR::eInstances,
        .geometry = geometry_data,
        .flags = {},
    };
    const vk::AccelerationStructureBuildRangeInfoKHR range{.primitiveCount = structure.element_count()};

    const vk::AccelerationStructureBuildGeometryInfoKHR build_info = instance_build_info(
        vk::AccelerationStructureTypeKHR::eTopLevel, structure.build_flags(), mode,
        mode == vk::BuildAccelerationStructureModeKHR::eUpdate ? structure.handle()
                                                               : vk::AccelerationStructureKHR{},
        structure.handle(), 1, &geometry, scratch.device_address());

    const VkAccelerationStructureBuildRangeInfoKHR* ranges =
        reinterpret_cast<const VkAccelerationStructureBuildRangeInfoKHR*>(&range);
    build_acceleration_structures_fn(device_)(
        static_cast<VkCommandBuffer>(cmd_), 1,
        reinterpret_cast<const VkAccelerationStructureBuildGeometryInfoKHR*>(&build_info), &ranges);
    return *this;
}

CommandBuffer& CommandBuffer::build_acceleration_structure(
    const AccelerationStructure& structure,
    const Buffer& instance_buffer,
    vk::BuildAccelerationStructureModeKHR mode) {
    if (!structure.scratch().valid())
        throw std::runtime_error(
            "CommandBuffer::build_acceleration_structure: the structure owns no scratch buffer; "
            "create it with own_scratch or pass one explicitly");
    return build_acceleration_structure(structure, instance_buffer, structure.scratch(), mode);
}

CommandBuffer& CommandBuffer::build_acceleration_structures(
    std::span<const vk::AccelerationStructureBuildGeometryInfoKHR> build_infos,
    std::span<const vk::AccelerationStructureBuildRangeInfoKHR*> range_infos) {
    if (build_infos.size() != range_infos.size())
        throw std::runtime_error(
            "CommandBuffer::build_acceleration_structures: build_infos and range_infos must have "
            "the same size");
    if (build_infos.empty())
        return *this;

    build_acceleration_structures_fn(device_)(
        static_cast<VkCommandBuffer>(cmd_), static_cast<uint32_t>(build_infos.size()),
        reinterpret_cast<const VkAccelerationStructureBuildGeometryInfoKHR*>(build_infos.data()),
        reinterpret_cast<const VkAccelerationStructureBuildRangeInfoKHR* const*>(
            range_infos.data()));
    return *this;
}

}
