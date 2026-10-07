#include "vulcao/context.h"
#include "vulcao/buffer.h"
#include "vulcao/detail/check.h"
#include "vulcao/image.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace vulcao {
namespace {

const char* device_type_name(VkPhysicalDeviceType type) {
    switch (type) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete GPU";
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated GPU";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual GPU";
        case VK_PHYSICAL_DEVICE_TYPE_CPU: return "CPU";
        default: return "other";
    }
}

LogLevel level_for_severity(VkDebugUtilsMessageSeverityFlagBitsEXT severity) {
    switch (severity) {
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT: return LogLevel::trace;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT: return LogLevel::debug;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT: return LogLevel::warning;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT: return LogLevel::error;
        default: return LogLevel::debug;
    }
}

LogCategory category_for_type(VkDebugUtilsMessageTypeFlagsEXT type) {
    if (type & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT)
        return LogCategory::validation;
    if (type & VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT)
        return LogCategory::performance;
    return LogCategory::general;
}

VKAPI_ATTR VkBool32 VKAPI_CALL validation_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void* /*user_data*/) {
    try {
        vulcao::log(level_for_severity(severity), category_for_type(type),
                    data != nullptr && data->pMessage != nullptr ? data->pMessage : "",
                    data != nullptr && data->pMessageIdName != nullptr ? data->pMessageIdName : "");
    } catch (...) {
        // Never let an exception cross the Vulkan callback boundary.
    }
    return VK_FALSE;
}

}

Context::Context(const ContextInfo& info) : info_(info) {
    create_instance(info_);
}

void Context::log(LogLevel level, std::string_view message) const {
    vulcao::log(level, LogCategory::general, message);
}

Context::~Context() {
    if (device_) {
        device_.waitIdle();

        if (command_pool_) {
            immediate_command_buffer_.destroy();
            submit_fence_.destroy();
            fence_pool_.clear();
            command_pool_.destroy();
        }

        destroy_swapchain_resources();

        if (transfer_started_) {
            // Release the in-flight uploads first: their command buffers have to
            // go back to the pool while it still exists, and their staging
            // buffers have to be freed before the allocator goes away.
            pending_uploads_.clear();
            transfer_pool_.destroy();
            transfer_timeline_.destroy();
        }

        staging_ = Buffer{};
        allocator_.destroy();

        if (vkb_swapchain_.swapchain)
            vkb::destroy_swapchain(vkb_swapchain_);
        vkb::destroy_device(vkb_device_);
    }

    if (surface_)
        instance_.destroySurfaceKHR(surface_);
    if (debug_messenger_) {
        const auto destroy_messenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            instance_.getProcAddr("vkDestroyDebugUtilsMessengerEXT"));
        if (destroy_messenger != nullptr)
            destroy_messenger(instance_, debug_messenger_, nullptr);
        debug_messenger_ = nullptr;
    }
    if (vkb_instance_.instance)
        vkb::destroy_instance(vkb_instance_);
}

void Context::initialize(vk::SurfaceKHR surface, vk::Extent2D extent, SwapchainInfo swapchain_info) {
    if (initialized())
        throw std::runtime_error("Context::initialize called twice");
    if (info_.headless)
        throw std::runtime_error("Context::initialize: the context was created headless");
    if (extent.width == 0 || extent.height == 0)
        throw std::runtime_error("Context::initialize: swapchain extent must be non-zero");

    surface_ = surface;
    swapchain_info_ = std::move(swapchain_info);
    pick_physical_device();
    create_device();
    create_allocator();
    create_swapchain({}, extent);
    create_command_pool();
}

void Context::initialize() {
    if (initialized())
        throw std::runtime_error("Context::initialize called twice");
    if (!info_.headless)
        throw std::runtime_error("Context::initialize: ContextInfo::headless must be true");

    pick_physical_device();
    create_device();
    create_allocator();
    create_command_pool();
}

void Context::wait_idle() {
    if (device_)
        device_.waitIdle();
}

void Context::create_instance(const ContextInfo& info) {
    vkb::InstanceBuilder builder;
    builder.set_app_name(info.app_name.c_str())
        .set_app_version(info.app_version)
        .set_headless(info.headless)
        .require_api_version(VK_VERSION_MAJOR(kApiVersion), VK_VERSION_MINOR(kApiVersion),
                             VK_VERSION_PATCH(kApiVersion));

    if (info.validation) {
        builder.request_validation_layers().enable_extension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    for (const char* extension : info.extensions)
        builder.enable_extension(extension);
    for (const char* layer : info.layers)
        builder.enable_layer(layer);

    vkb_instance_ = detail::check(builder.build(), "create instance");
    instance_ = vk::Instance{vkb_instance_.instance};

    // The messenger is created here instead of through vk-bootstrap. Bootstrap
    // caches its instance function pointers process wide on the first instance it
    // builds and never invalidates them, so an instance created without
    // VK_EXT_debug_utils would break the messenger of every later instance that
    // asks for validation. Resolving the entry point per instance avoids that.
    //
    // Trade-off: bootstrap also chains its messenger create info into the instance
    // create info, which makes the layers deliver messages emitted while the
    // instance itself is created. That chain cannot be reached from here, so those
    // messages are lost. Errors from a malformed instance setup still surface as
    // descriptive bootstrap errors.
    if (info.validation) {
        create_debug_messenger(info.validation_severity);
    }

    debug_utils_enabled_ = static_cast<bool>(debug_messenger_);
    for (const char* extension : info.extensions)
        if (std::string_view(extension) == VK_EXT_DEBUG_UTILS_EXTENSION_NAME)
            debug_utils_enabled_ = true;

    if (debug_utils_enabled_)
        set_debug_name_ext_ = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
            instance_.getProcAddr("vkSetDebugUtilsObjectNameEXT"));
}

void Context::create_debug_messenger(vk::DebugUtilsMessageSeverityFlagsEXT severity) {
    const auto create_messenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        instance_.getProcAddr("vkCreateDebugUtilsMessengerEXT"));
    if (create_messenger == nullptr)
        throw std::runtime_error("create debug messenger: vkCreateDebugUtilsMessengerEXT is missing");

    VkDebugUtilsMessengerCreateInfoEXT create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    create_info.messageSeverity = static_cast<VkDebugUtilsMessageSeverityFlagsEXT>(severity);
    create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    create_info.pfnUserCallback = &validation_callback;

    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    detail::check(static_cast<vk::Result>(create_messenger(instance_, &create_info, nullptr, &messenger)),
          "create debug messenger");
    debug_messenger_ = vk::DebugUtilsMessengerEXT{messenger};
}

void Context::set_debug_name(vk::ObjectType type, uint64_t handle, const char* name) const {
    if (!set_debug_name_ext_)
        return;

    VkDebugUtilsObjectNameInfoEXT info{};
    info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    info.objectType = static_cast<VkObjectType>(type);
    info.objectHandle = handle;
    info.pObjectName = name;
    set_debug_name_ext_(device_, &info);
}

std::vector<PhysicalDeviceInfo> Context::enumerate_physical_devices() const {
    std::vector<PhysicalDeviceInfo> result;
    for (const vk::PhysicalDevice& physical_device : instance_.enumeratePhysicalDevices()) {
        const vk::PhysicalDeviceProperties properties = physical_device.getProperties();
        result.push_back(PhysicalDeviceInfo{
            .name = properties.deviceName.data(),
            .type = properties.deviceType,
            .vendor_id = properties.vendorID,
            .driver_version = properties.driverVersion,
            .api_version = properties.apiVersion,
        });
    }
    return result;
}

void Context::pick_physical_device() {
    vkb::PhysicalDeviceSelector selector{vkb_instance_};
    if (!info_.headless)
        selector.set_surface(surface_);
    selector.set_minimum_version(VK_API_VERSION_MAJOR(kApiVersion), VK_API_VERSION_MINOR(kApiVersion));

    const DeviceFeatures& wanted = info_.device_features;
    selector.set_required_features(vk::PhysicalDeviceFeatures{
        .drawIndirectFirstInstance = wanted.draw_indirect_first_instance ? VK_TRUE : VK_FALSE,
        .depthClamp = wanted.depth_clamp ? VK_TRUE : VK_FALSE,
        .fillModeNonSolid = wanted.fill_mode_non_solid ? VK_TRUE : VK_FALSE,
        .wideLines = wanted.wide_lines ? VK_TRUE : VK_FALSE,
        .samplerAnisotropy = wanted.sampler_anisotropy ? VK_TRUE : VK_FALSE,
        .shaderInt64 = wanted.shader_int64 ? VK_TRUE : VK_FALSE,
    });

    selector.set_required_features_11(vk::PhysicalDeviceVulkan11Features{
        .storageBuffer16BitAccess = wanted.storage_buffer_16bit ? VK_TRUE : VK_FALSE,
        .shaderDrawParameters = wanted.shader_draw_parameters ? VK_TRUE : VK_FALSE,
    });

    selector.set_required_features_13(vk::PhysicalDeviceVulkan13Features{
        .synchronization2 = VK_TRUE,
        .dynamicRendering = VK_TRUE,
    });

    // Ray tracing builds on acceleration structures, which in turn require
    // buffer device addresses, so requesting a higher level enables the lower
    // ones instead of making the caller repeat the whole dependency chain.
    const bool wants_acceleration_structure =
        wanted.acceleration_structure || wanted.ray_query || wanted.ray_tracing_pipeline;
    const bool wants_buffer_device_address =
        wanted.buffer_device_address || wants_acceleration_structure;

    const bool wants_vulkan12 =
        wanted.timeline_semaphore || wanted.descriptor_indexing || wanted.scalar_block_layout ||
        wanted.shader_float16 || wanted.shader_int8 || wanted.storage_buffer_8bit ||
        wanted.storage_buffer_16bit || wanted.shader_subgroup_extended_types ||
        wanted.host_query_reset || wants_buffer_device_address;

    if (wants_vulkan12) {
        // All Vulkan 1.2 features live in one structure: the driver rejects the
        // same structure type twice in the chain, so they cannot be added one at
        // a time the way the extension feature structs are.
        vk::PhysicalDeviceVulkan12Features features{};
        if (wanted.descriptor_indexing) {
            features.shaderInputAttachmentArrayDynamicIndexing = VK_TRUE;
            features.shaderUniformTexelBufferArrayDynamicIndexing = VK_TRUE;
            features.shaderStorageTexelBufferArrayDynamicIndexing = VK_TRUE;
            features.shaderUniformBufferArrayNonUniformIndexing = VK_TRUE;
            features.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
            features.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
            features.shaderStorageImageArrayNonUniformIndexing = VK_TRUE;
            features.shaderInputAttachmentArrayNonUniformIndexing = VK_TRUE;
            features.shaderUniformTexelBufferArrayNonUniformIndexing = VK_TRUE;
            features.shaderStorageTexelBufferArrayNonUniformIndexing = VK_TRUE;
            features.descriptorBindingUniformBufferUpdateAfterBind = VK_TRUE;
            features.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
            features.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE;
            features.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
            features.descriptorBindingUniformTexelBufferUpdateAfterBind = VK_TRUE;
            features.descriptorBindingStorageTexelBufferUpdateAfterBind = VK_TRUE;
            features.descriptorBindingUpdateUnusedWhilePending = VK_TRUE;
            features.descriptorBindingPartiallyBound = VK_TRUE;
            features.descriptorBindingVariableDescriptorCount = VK_TRUE;
            features.runtimeDescriptorArray = VK_TRUE;
        }
        features.timelineSemaphore = wanted.timeline_semaphore ? VK_TRUE : VK_FALSE;
        features.hostQueryReset = wanted.host_query_reset ? VK_TRUE : VK_FALSE;
        features.scalarBlockLayout = wanted.scalar_block_layout ? VK_TRUE : VK_FALSE;
        features.shaderFloat16 = wanted.shader_float16 ? VK_TRUE : VK_FALSE;
        features.shaderInt8 = wanted.shader_int8 ? VK_TRUE : VK_FALSE;
        features.storageBuffer8BitAccess = wanted.storage_buffer_8bit ? VK_TRUE : VK_FALSE;
        features.shaderSubgroupExtendedTypes =
            wanted.shader_subgroup_extended_types ? VK_TRUE : VK_FALSE;
        features.bufferDeviceAddress = wants_buffer_device_address ? VK_TRUE : VK_FALSE;
        selector.set_required_features_12(features);
    }

    if (wants_acceleration_structure) {
        selector.add_required_extension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        selector.add_required_extension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        selector.add_required_extension_features(
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR{.accelerationStructure = VK_TRUE});
    }

    if (wanted.ray_query) {
        selector.add_required_extension(VK_KHR_RAY_QUERY_EXTENSION_NAME);
        selector.add_required_extension_features(
            vk::PhysicalDeviceRayQueryFeaturesKHR{.rayQuery = VK_TRUE});
    }

    if (wanted.ray_tracing_pipeline) {
        selector.add_required_extension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
        selector.add_required_extension_features(
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR{.rayTracingPipeline = VK_TRUE});
    }

    for (const char* extension : info_.device_extensions)
        selector.add_required_extension(extension);

    if (info_.separate_compute_queue)
        selector.require_separate_compute_queue();
    if (info_.separate_transfer_queue)
        selector.require_separate_transfer_queue();

    if (info_.customize_selector)
        info_.customize_selector(selector);

    vkb_physical_device_ = detail::check(selector.select(), "select physical device");
    physical_device_ = vk::PhysicalDevice{vkb_physical_device_.physical_device};

    const vk::PhysicalDeviceProperties properties = physical_device_.getProperties();
    physical_device_info_ = PhysicalDeviceInfo{
        .name = properties.deviceName.data(),
        .type = properties.deviceType,
        .vendor_id = properties.vendorID,
        .driver_version = properties.driverVersion,
        .api_version = properties.apiVersion,
    };
    log(LogLevel::info, "GPU: " + std::string(properties.deviceName.data()) + " (" +
                            device_type_name(static_cast<VkPhysicalDeviceType>(properties.deviceType)) +
                            ")");
}

void Context::create_device() {
    vkb_device_ = detail::check(vkb::DeviceBuilder{vkb_physical_device_}.build(), "create device");
    device_ = vk::Device{vkb_device_.device};

    graphics_queue_family_index_ =
        detail::check(vkb_device_.get_queue_index(vkb::QueueType::graphics), "get graphics queue index");
    graphics_queue_ = vk::Queue{detail::check(vkb_device_.get_queue(vkb::QueueType::graphics), "get graphics queue")};
    if (!info_.headless) {
        present_queue_ =
            vk::Queue{detail::check(vkb_device_.get_queue(vkb::QueueType::present), "get present queue")};
        present_queue_family_index_ =
            detail::check(vkb_device_.get_queue_index(vkb::QueueType::present), "get present queue index");
    }

    if (info_.separate_compute_queue) {
        const vkb::Result<uint32_t> index = vkb_device_.get_queue_index(vkb::QueueType::compute);
        const vkb::Result<VkQueue> queue = vkb_device_.get_queue(vkb::QueueType::compute);
        if (index && queue) {
            compute_queue_family_index_ = index.value();
            compute_queue_ = vk::Queue{queue.value()};
            has_compute_queue_ = true;
        } else {
            log(LogLevel::warning, "dedicated compute queue was requested but is not available");
        }
    }

    if (info_.separate_transfer_queue) {
        const vkb::Result<uint32_t> index = vkb_device_.get_queue_index(vkb::QueueType::transfer);
        const vkb::Result<VkQueue> queue = vkb_device_.get_queue(vkb::QueueType::transfer);
        if (index && queue) {
            transfer_queue_family_index_ = index.value();
            transfer_queue_ = vk::Queue{queue.value()};
            has_transfer_queue_ = true;
        } else {
            log(LogLevel::warning, "dedicated transfer queue was requested but is not available");
        }
    }

    std::string present = "none";
    if (!info_.headless)
        present = std::to_string(present_queue_family_index_);

    log(LogLevel::info, "queues: graphics=" + std::to_string(graphics_queue_family_index_) +
                            ", present=" + present + ", compute=" +
                            (has_compute_queue_ ? std::to_string(compute_queue_family_index_) : "none") +
                            ", transfer=" +
                            (has_transfer_queue_ ? std::to_string(transfer_queue_family_index_) : "none"));
}

void Context::create_allocator() {
    const DeviceFeatures& features = info_.device_features;
    const bool buffer_device_address = features.buffer_device_address || has_acceleration_structure();
    allocator_.create(instance_, physical_device_, device_, kApiVersion, buffer_device_address);
}

void Context::create_swapchain(vk::SwapchainKHR oldSwapchain, vk::Extent2D extent) {
    const vk::SurfaceCapabilitiesKHR capabilities = physical_device_.getSurfaceCapabilitiesKHR(surface_);
    if (capabilities.currentExtent.width == 0 || capabilities.currentExtent.height == 0)
        throw std::runtime_error("create swapchain: surface extent must be non-zero");

    vkb::SwapchainBuilder builder{vkb_device_, surface_};
    builder.set_desired_extent(extent.width, extent.height);

    if (!swapchain_info_.formats.empty()) {
        builder.set_desired_format(swapchain_info_.formats.front());
        for (size_t i = 1; i < swapchain_info_.formats.size(); ++i)
            builder.add_fallback_format(swapchain_info_.formats[i]);
    }

    if (!swapchain_info_.present_modes.empty()) {
        builder.set_desired_present_mode(
            static_cast<VkPresentModeKHR>(swapchain_info_.present_modes.front()));
        for (size_t i = 1; i < swapchain_info_.present_modes.size(); ++i)
            builder.add_fallback_present_mode(
                static_cast<VkPresentModeKHR>(swapchain_info_.present_modes[i]));
    }

    if (swapchain_info_.extra_usage != vk::ImageUsageFlags{})
        builder.add_image_usage_flags(static_cast<VkImageUsageFlags>(swapchain_info_.extra_usage));

    if (swapchain_info_.min_image_count > 0)
        builder.set_desired_min_image_count(swapchain_info_.min_image_count);

    if (oldSwapchain)
        builder.set_old_swapchain(oldSwapchain);

    vkb_swapchain_ = detail::check(builder.build(), "create swapchain");

    swapchain_ = vk::SwapchainKHR{vkb_swapchain_.swapchain};
    swapchain_format_ = static_cast<vk::Format>(vkb_swapchain_.image_format);
    swapchain_extent_ = vk::Extent2D{vkb_swapchain_.extent.width, vkb_swapchain_.extent.height};

    auto images = detail::check(vkb_swapchain_.get_images(), "get swapchain images");
    swapchain_images_.assign(images.begin(), images.end());

    auto views = detail::check(vkb_swapchain_.get_image_views(), "create swapchain image views");
    swapchain_image_views_.assign(views.begin(), views.end());

    log(LogLevel::info, "swapchain: " + std::to_string(swapchain_extent_.width) + "x" +
                            std::to_string(swapchain_extent_.height) + ", " +
                            std::to_string(swapchain_images_.size()) + " images, " +
                            vk::to_string(swapchain_format_));
}

void Context::create_command_pool() {
    command_pool_ = CommandPool::create(device_, graphics_queue_family_index_,
                                        vk::CommandPoolCreateFlagBits::eResetCommandBuffer);

    immediate_command_buffer_ = command_pool_.allocate(vk::CommandBufferLevel::ePrimary,
                                                       debug_utils_enabled_);
    submit_fence_ = Fence::create(device_);
}

void Context::destroy_swapchain_resources() {
    for (auto view : swapchain_image_views_)
        device_.destroyImageView(view);
    swapchain_image_views_.clear();
    swapchain_images_.clear();
}

bool Context::recreate_swapchain(vk::Extent2D extent) {
    if (!initialized())
        throw std::runtime_error("Context::recreate_swapchain before initialize");
    if (info_.headless)
        throw std::runtime_error("Context::recreate_swapchain: the context is headless");
    if (extent.width == 0 || extent.height == 0)
        return false;

    device_.waitIdle();

    const vk::SurfaceCapabilitiesKHR capabilities = physical_device_.getSurfaceCapabilitiesKHR(surface_);
    if (capabilities.currentExtent.width == 0 || capabilities.currentExtent.height == 0)
        return false;

    destroy_swapchain_resources();

    vkb::Swapchain old_swapchain = vkb_swapchain_;
    create_swapchain(swapchain_, extent);
    if (old_swapchain.swapchain)
        vkb::destroy_swapchain(old_swapchain);
    return true;
}

void Context::submit_and_wait(vk::CommandBuffer cmd) {
    submit_fence_.reset();
    submit(cmd, submit_fence_.handle());
    submit_fence_.wait();
}

void Context::submit(const vk::SubmitInfo2& info, vk::Fence fence) {
    submit(graphics_queue_, info, fence);
}

void Context::submit(vk::Queue queue, const vk::SubmitInfo2& info, vk::Fence fence) {
    queue.submit2(info, fence);
}

void Context::submit(vk::Queue queue,
                     vk::CommandBuffer cmd,
                     const Semaphore& timeline_semaphore,
                     uint64_t signal_value,
                     vk::Fence fence) {
    const vk::SemaphoreSubmitInfo signal_info{
        .semaphore = timeline_semaphore.handle(),
        .value = signal_value,
        .stageMask = vk::PipelineStageFlagBits2::eAllCommands,
    };
    const vk::CommandBufferSubmitInfo command_info{.commandBuffer = cmd};

    queue.submit2(vk::SubmitInfo2{
                      .commandBufferInfoCount = 1,
                      .pCommandBufferInfos = &command_info,
                      .signalSemaphoreInfoCount = 1,
                      .pSignalSemaphoreInfos = &signal_info,
                  },
                  fence);
}

void Context::submit(vk::Queue queue,
                     vk::CommandBuffer cmd,
                     const Semaphore& wait_semaphore,
                     uint64_t wait_value,
                     vk::PipelineStageFlags2 wait_stage,
                     vk::Fence fence) {
    const vk::SemaphoreSubmitInfo wait_info{
        .semaphore = wait_semaphore.handle(),
        .value = wait_value,
        .stageMask = wait_stage,
    };
    const vk::CommandBufferSubmitInfo command_info{.commandBuffer = cmd};

    queue.submit2(vk::SubmitInfo2{
                      .waitSemaphoreInfoCount = 1,
                      .pWaitSemaphoreInfos = &wait_info,
                      .commandBufferInfoCount = 1,
                      .pCommandBufferInfos = &command_info,
                  },
                  fence);
}

void Context::submit(vk::CommandBuffer cmd, vk::Fence fence) {
    const vk::CommandBufferSubmitInfo command_info{.commandBuffer = cmd};
    submit(vk::SubmitInfo2{
               .commandBufferInfoCount = 1,
               .pCommandBufferInfos = &command_info,
           },
           fence);
}

Fence Context::submit(vk::CommandBuffer cmd) {
    Fence fence = Fence::create(device_);
    submit(cmd, fence.handle());
    return fence;
}

vk::Fence Context::submit_pooled(vk::CommandBuffer cmd) {
    // A signaled fence is safe to recycle: waits on it have already completed
    // or return immediately. Fences still in flight keep their slot.
    for (Fence& fence : fence_pool_) {
        if (!fence.signaled())
            continue;
        fence.reset();
        submit(cmd, fence.handle());
        return fence.handle();
    }

    fence_pool_.push_back(Fence::create(device_));
    submit(cmd, fence_pool_.back().handle());
    return fence_pool_.back().handle();
}

void Context::upload(Buffer& dst, const void* data, vk::DeviceSize size) {
    if (size == 0)
        return;
    if (!dst.valid())
        throw std::runtime_error("Context::upload: invalid destination buffer");
    if (!(dst.usage() & vk::BufferUsageFlagBits::eTransferDst))
        throw std::runtime_error("Context::upload: destination buffer requires TransferDst usage");
    if (size > dst.size())
        throw std::runtime_error("Context::upload: data size exceeds destination buffer size");

    Buffer& stage = staging(size);
    stage.write_bytes(data, size);

    immediate([&](CommandBuffer& cmd) {
        cmd.copy_buffer(stage.handle(), dst.handle(), size);
    });
}

void Context::upload(Image& dst, const void* data, vk::DeviceSize size, vk::ImageLayout final_layout,
                     bool generate_mips) {
    if (size == 0)
        return;
    if (!dst.valid())
        throw std::runtime_error("Context::upload: invalid destination image");
    if (!(dst.usage() & vk::ImageUsageFlagBits::eTransferDst))
        throw std::runtime_error("Context::upload: destination image requires TransferDst usage");
    if (generate_mips && dst.mip_levels() > 1 &&
        !(dst.usage() & vk::ImageUsageFlagBits::eTransferSrc))
        throw std::runtime_error(
            "Context::upload: generating mipmaps requires TransferSrc usage on the destination image");

    const vk::DeviceSize required = image_byte_size(dst.extent(), dst.format());
    if (size < required)
        throw std::runtime_error("Context::upload: " + std::to_string(size) +
                                 " bytes are not enough for the destination image, which needs " +
                                 std::to_string(required));

    Buffer& stage = staging(size);
    stage.write_bytes(data, size);

    immediate([&](CommandBuffer& cmd) {
        cmd.transition(dst, vk::ImageLayout::eTransferDstOptimal);
        cmd.copy_buffer_to_image(stage.handle(), dst);
        if (generate_mips && dst.mip_levels() > 1)
            cmd.generate_mipmaps(dst, final_layout);
        else
            cmd.transition(dst, final_layout);
    });
}

void Context::download(const Buffer& src, void* data, vk::DeviceSize size) {
    if (size == 0)
        return;
    if (!src.valid())
        throw std::runtime_error("Context::download: invalid source buffer");
    if (!(src.usage() & vk::BufferUsageFlagBits::eTransferSrc))
        throw std::runtime_error("Context::download: source buffer requires TransferSrc usage");
    if (size > src.size())
        throw std::runtime_error("Context::download: data size exceeds source buffer size");

    Buffer& stage = staging(size);
    immediate([&](CommandBuffer& cmd) {
        cmd.copy_buffer(src.handle(), stage.handle(), size);
    });

    stage.invalidate(0, size);
    std::memcpy(data, stage.map(), static_cast<size_t>(size));
}

void Context::download(Image& src, void* data, vk::DeviceSize size) {
    if (size == 0)
        return;
    if (!src.valid())
        throw std::runtime_error("Context::download: invalid source image");
    if (!(src.usage() & vk::ImageUsageFlagBits::eTransferSrc))
        throw std::runtime_error("Context::download: source image requires TransferSrc usage");

    const vk::DeviceSize required = image_byte_size(src.extent(), src.format());
    if (size < required)
        throw std::runtime_error("Context::download: " + std::to_string(size) +
                                 " bytes are not enough for the source image, which holds " +
                                 std::to_string(required));

    Buffer& stage = staging(size);
    const vk::ImageLayout previous = src.layout();
    immediate([&](CommandBuffer& cmd) {
        cmd.transition(src, vk::ImageLayout::eTransferSrcOptimal);
        cmd.copy_image_to_buffer(stage.handle(), src);
        cmd.transition(src, previous);
    });

    stage.invalidate(0, size);
    std::memcpy(data, stage.map(), static_cast<size_t>(size));
}

Buffer& Context::staging(vk::DeviceSize size) {
    if (!staging_ || staging_.size() < size) {
        vk::DeviceSize capacity = staging_ ? staging_.size() : (1u << 16);
        while (capacity < size)
            capacity *= 2;

        staging_ = Buffer::create(allocator_, capacity,
                                  vk::BufferUsageFlagBits::eTransferSrc |
                                      vk::BufferUsageFlagBits::eTransferDst,
                                  VMA_MEMORY_USAGE_AUTO,
                                  VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
    }
    return staging_;
}

std::vector<uint32_t> Context::transfer_sharing_families() const {
    if (has_transfer_queue_)
        return {graphics_queue_family_index_, transfer_queue_family_index_};
    return {graphics_queue_family_index_};
}

void Context::ensure_transfer_objects() {
    if (transfer_started_)
        return;
    if (!initialized())
        throw std::runtime_error("Context::upload_async before initialize");
    if (!info_.device_features.timeline_semaphore)
        throw std::runtime_error(
            "Context::upload_async requires DeviceFeatures::timeline_semaphore");

    // Without a dedicated transfer queue the uploads fall back to the graphics
    // queue; the flow is identical minus the ownership transfer.
    const uint32_t family =
        has_transfer_queue_ ? transfer_queue_family_index_ : graphics_queue_family_index_;
    transfer_pool_ =
        CommandPool::create(device_, family, vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    transfer_timeline_ = Semaphore::create_timeline(device_, 0);
    transfer_started_ = true;
}

void Context::reclaim_uploads() {
    if (!transfer_started_)
        return;

    const uint64_t reached = transfer_timeline_.value();
    std::erase_if(pending_uploads_,
                  [reached](const PendingUpload& pending) { return pending.value <= reached; });
}

Context::AsyncUpload Context::upload_async(Buffer& dst, const void* data, vk::DeviceSize size) {
    if (size == 0)
        return {};
    if (!dst.valid())
        throw std::runtime_error("Context::upload_async: invalid destination buffer");
    if (!(dst.usage() & vk::BufferUsageFlagBits::eTransferDst))
        throw std::runtime_error("Context::upload_async: destination buffer requires TransferDst usage");
    if (size > dst.size())
        throw std::runtime_error("Context::upload_async: data size exceeds destination buffer size");
    if (has_transfer_queue_ && dst.sharing_mode() != vk::SharingMode::eConcurrent)
        throw std::runtime_error(
            "Context::upload_async: with a dedicated transfer queue the destination must be "
            "created with eConcurrent sharing, see Context::transfer_sharing_families");

    ensure_transfer_objects();
    reclaim_uploads();

    Buffer stage = Buffer::create(allocator_, size, vk::BufferUsageFlagBits::eTransferSrc,
                                  VMA_MEMORY_USAGE_AUTO,
                                  VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
    stage.write_bytes(data, size);

    // A command buffer of its own: uploads are submitted back to back, and the
    // previous one is still pending, so it must not be reset or reused.
    CommandBuffer cmd =
        transfer_pool_.allocate(vk::CommandBufferLevel::ePrimary, debug_utils_enabled_);
    cmd.begin();
    cmd.copy_buffer(stage.handle(), dst.handle(), size);
    cmd.end();

    const uint64_t value = ++transfer_counter_;
    submit(has_transfer_queue_ ? transfer_queue_ : graphics_queue_, cmd.handle(),
           transfer_timeline_, value);

    pending_uploads_.push_back(PendingUpload{value, std::move(stage), std::move(cmd)});
    return AsyncUpload{value};
}

Context::AsyncUpload Context::upload_async(Image& dst,
                                           const void* data,
                                           vk::DeviceSize size,
                                           vk::ImageLayout final_layout) {
    if (size == 0)
        return {};
    if (!dst.valid())
        throw std::runtime_error("Context::upload_async: invalid destination image");
    if (!(dst.usage() & vk::ImageUsageFlagBits::eTransferDst))
        throw std::runtime_error("Context::upload_async: destination image requires TransferDst usage");

    const vk::DeviceSize required = image_byte_size(dst.extent(), dst.format());
    if (size < required)
        throw std::runtime_error("Context::upload_async: " + std::to_string(size) +
                                 " bytes are not enough for the destination image, which needs " +
                                 std::to_string(required));
    if (has_transfer_queue_ && dst.sharing_mode() != vk::SharingMode::eConcurrent)
        throw std::runtime_error(
            "Context::upload_async: with a dedicated transfer queue the destination must be "
            "created with eConcurrent sharing, see Context::transfer_sharing_families");

    ensure_transfer_objects();
    reclaim_uploads();

    Buffer stage = Buffer::create(allocator_, size, vk::BufferUsageFlagBits::eTransferSrc,
                                  VMA_MEMORY_USAGE_AUTO,
                                  VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
    stage.write_bytes(data, size);

    // A command buffer of its own: uploads are submitted back to back, and the
    // previous one is still pending, so it must not be reset or reused.
    CommandBuffer cmd =
        transfer_pool_.allocate(vk::CommandBufferLevel::ePrimary, debug_utils_enabled_);
    cmd.begin();

    // A dedicated transfer queue only supports the transfer stage, so both
    // transitions spell their dependency out rather than deriving fragment or
    // compute stages from the image's layout, which that queue cannot name
    // (VUID-vkCmdPipelineBarrier2-srcStageMask-09675). Ordering against the
    // graphics side comes from the timeline, not from these barriers.
    const bool transfer_only = has_transfer_queue_;
    if (transfer_only && dst.layout() != vk::ImageLayout::eUndefined)
        cmd.transition(dst, vk::ImageLayout::eTransferDstOptimal,
                       vk::PipelineStageFlagBits2::eTransfer,
                       vk::AccessFlagBits2::eTransferWrite,
                       vk::PipelineStageFlagBits2::eTransfer,
                       vk::AccessFlagBits2::eTransferWrite);
    else
        cmd.transition(dst, vk::ImageLayout::eTransferDstOptimal);

    cmd.copy_buffer_to_image(stage.handle(), dst);

    if (transfer_only)
        cmd.transition(dst, final_layout, vk::PipelineStageFlagBits2::eTransfer,
                       vk::AccessFlagBits2::eTransferWrite,
                       vk::PipelineStageFlagBits2::eTransfer,
                       vk::AccessFlagBits2::eTransferWrite);
    else
        cmd.transition(dst, final_layout);
    cmd.end();

    const uint64_t value = ++transfer_counter_;
    submit(has_transfer_queue_ ? transfer_queue_ : graphics_queue_, cmd.handle(),
           transfer_timeline_, value);

    pending_uploads_.push_back(PendingUpload{value, std::move(stage), std::move(cmd)});
    return AsyncUpload{value};
}

void Context::wait_upload(const AsyncUpload& upload) {
    if (upload.value == 0)
        return;

    transfer_timeline_.wait(upload.value);
    reclaim_uploads();
}

void Context::wait_uploads() {
    if (!transfer_started_)
        return;

    transfer_timeline_.wait(transfer_counter_);
    reclaim_uploads();
}

}
