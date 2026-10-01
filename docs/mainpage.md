# vulcao

A thin, RAII wrapper around Vulkan. It maps close to the Vulkan API, owns objects
with move-only copy semantics, throws on failure and derives descriptor and vertex
layouts from SPIR-V reflection.

The goal is to remove repetitive Vulkan boilerplate without hiding Vulkan: raw
handles, creation structs and explicit synchronization stay visible, so render
graphs, bindless registries and renderers remain the application's job.

## Features

- **[@ref vulcao::Context]** owns the instance, physical and logical device,
  swapchain, queues, command pool and the VMA allocator. It can also be created
  headless for compute-only or offscreen work, and reports the selected device
  along with its timestamp period and memory properties.
- **[@ref vulcao::FrameManager]** owns frames in flight, the swapchain acquire and
  the present loop.
- **Resources** are VMA backed: [@ref vulcao::Buffer], [@ref vulcao::Image] and
  its [@ref vulcao::ImageView], [@ref vulcao::Sampler] and
  [@ref vulcao::CommandPool].
- **Transfer** helpers upload and download buffers and images through a reused
  staging buffer, and expose the full `vk::BufferImageCopy` region for
  row-pitched data.
- **Synchronization** primitives: [@ref vulcao::Fence], [@ref vulcao::Semaphore]
  (binary and timeline), [@ref vulcao::Event] and [@ref vulcao::QueryPool].
- **Recording** through [@ref vulcao::CommandBuffer], including layout
  transitions, single and multi-region copies and blits, resolve, mipmap
  generation, attachment clears, draws (with the indirect-count variants),
  dispatches, events, dynamic state, debug labels and queue family ownership
  transfers.
- **Ray tracing** through [@ref vulcao::AccelerationStructure], which builds
  bottom and top level structures from triangle geometry and instances, plus the
  `DeviceFeatures::ray_query` / `ray_tracing_pipeline` switches and the
  acceleration structure descriptor write.
- **Pipelines** built from [@ref vulcao::ShaderModule] reflection:
  [@ref vulcao::PipelineLayout], [@ref vulcao::Pipeline],
  [@ref vulcao::DescriptorSetLayout] and [@ref vulcao::PipelineCache]. Graphics
  pipelines cover the vertex, tessellation, geometry and fragment stages,
  explicit blend and multisample state, and a raw
  `vk::GraphicsPipelineCreateInfo` overload.
- A global, structured **logging** callback ([@ref vulcao::set_log_callback])
  that also receives validation layer messages.

## Building

vulcao requires Vulkan 1.3: it records with synchronization2 and renders with
dynamic rendering.

```sh
git clone --recursive <repository>
cmake -S . -B build
cmake --build build
```

Run the tests with `ctest --test-dir build`.

The samples live in a separate repository,
[vulcao-samples](https://github.com/Yang-Junjie/vulcao-samples).

## Where to start

- [@ref vulcao::Context] for device and swapchain setup.
- [@ref vulcao::CommandBuffer] for recording.
- [@ref vulcao::Pipeline] and [@ref vulcao::DescriptorSet] for drawing.
- [@ref vulcao::FrameManager] for a complete frame loop.
