# vulcao

A thin, RAII wrapper around Vulkan. It maps close to the Vulkan API, owns objects
with move-only copy semantics, throws on failure and derives descriptor and vertex
layouts from SPIR-V reflection.

## Features

- `Context`: instance, physical device, logical device, swapchain, queues and
  command pool, with configurable device features/extensions and dedicated
  compute/transfer queues. Created headless (`ContextInfo::headless`) it skips
  the surface and swapchain for compute-only or offscreen work.
- Resources: VMA-backed `Allocator`, `Buffer` (including `create_with_data`, the
  create-and-upload pair in one call), `BufferView` for texel buffers, `Image`
  (2D/depth factories, mipmap generation), `Sampler` and `CommandPool`.
- Transfer: staging-backed `upload`/`download` for buffers and images, with the
  full `vk::BufferImageCopy` region exposed for row-pitched data. `upload_async`
  runs uploads on the dedicated transfer queue (when requested) and reports
  completion through a timeline semaphore; destinations are created with
  `Context::transfer_sharing_families()` so no ownership ceremony is needed.
- Synchronization: `Fence`, binary and timeline `Semaphore`, `QueryPool` and a
  `FrameManager` that owns frames in flight, swapchain acquire and present,
  plus per-slot `DeletionQueue`s: `defer_destroy` retires a resource once the
  GPU is done with the frames that referenced it.
- Recording: `CommandBuffer` with layout transitions (including swapchain
  `transition_to_render`/`transition_to_present` helpers), copies, mipmaps,
  draws, dispatches, queries, dynamic state, debug labels and queue family
  ownership transfers (release/acquire), plus `color_attachment`/
  `depth_attachment` builders and a full-extent `begin_rendering` convenience.
- Descriptors: `DescriptorPool::create_for_bindings` sizes a pool from a
  reflected set's bindings (e.g. one set per frame in flight) without
  hand-counted pool sizes. Writes cover buffers, combined and separate
  image/sampler pairs, storage and input attachments, and texel buffers.
- Pipelines: `ShaderModule` with SPIR-V reflection, `DescriptorSetLayout/Pool/Set`
  with a batching writer and layout cache, `PipelineLayout`, `Pipeline` with
  graphics/compute factories, specialization constants and `PipelineCache`.
- Descriptor indexing: layouts accept creation flags and per-binding flags
  (`PARTIALLY_BOUND`, `UPDATE_AFTER_BIND`, `VARIABLE_DESCRIPTOR_COUNT`), pools
  allocate variable descriptor counts, and reflected runtime arrays report
  descriptor count 0 which `set_binding_count` turns into a concrete bound. The
  building blocks for bindless, without prescribing a bindless design.
- Reflection: descriptor sets, push constants and vertex attributes extracted from
  SPIR-V with SPIRV-Reflect, plus `make_vertex_layout`.
- Logging: a global `set_log_callback` receives structured `LogMessage`s from the
  library and from the validation layers; `set_log_level` filters them and the
  default prints warnings, errors and validation output to stderr.

## Requirements

- CMake 3.24+
- A C++20 compiler
- The Vulkan SDK (also provides `slangc`, used by the samples)

All other dependencies are git submodules.

## Building

```sh
git clone --recursive https://github.com/<you>/vulcao
cmake -S . -B build
cmake --build build
```

`third_party/` holds the library's own dependencies; GLFW and GLM serve the
samples only and live under `samples/third_party`. A recursive clone fetches
everything, but to skip the sample dependencies initialize just what is needed
(add `third_party/doctest` for the tests) and configure with
`-DVULCAO_BUILD_SAMPLES=OFF`:

```sh
git clone https://github.com/<you>/vulcao && cd vulcao
git submodule update --init \
    third_party/vk-bootstrap third_party/VulkanMemoryAllocator third_party/spirv-reflect
```

Options:

- `VULCAO_BUILD_SAMPLES` (default `ON`): build the samples.
- `VULCAO_BUILD_TESTS` (default `ON` when top level): build and register tests.
- `VULCAO_INSTALL` (default `ON` when top level): generate install rules.

Run the tests with `ctest --test-dir build`.

## Samples

- `01_hello_triangle`: a CPU-side vertex and index buffer drawn through a full
  graphics pipeline.
- `02_uniforms`: the triangle spun by a uniform buffer and descriptor set.
- `03_compute`: a storage buffer transformed by a compute shader, with the result
  verified on the CPU. Needs no display, and exits non-zero on a wrong result or
  on a validation error.
- `04_offscreen`: a triangle rendered into an image, then read back and checked
  pixel by pixel. Also needs no display.
- `05_texture`: a PNG decoded at run time, uploaded as a mipmapped texture and
  sampled through a combined image sampler onto a quad that fills the window.
  Opens a window; pass any image path as the first argument to show another one.

```sh
./build/samples/01_hello_triangle/hello_triangle
./build/samples/02_uniforms/uniforms
./build/samples/03_compute/compute
./build/samples/04_offscreen/offscreen
./build/samples/05_texture/texture
```

## Threading model

vulcao follows the Vulkan rule: objects are cheap to create on any thread, but
anything that mutates shared state is externally synchronized. Concretely:

- **Safe from multiple threads**: creating and destroying independent
  `Buffer`/`Image`/`Pipeline`/`Sampler`/descriptor objects (the VMA allocator
  and the driver handle their own locking), recording into command buffers
  allocated from *different* `CommandPool`s, and logging through the global
  log callback.
- **Single thread only**: `Context::immediate()` and `upload()`/`download()` (they
  share one internal command buffer and the staging buffer), `upload_async()` (the
  transfer pool, the timeline and the pending upload bookkeeping are shared even
  though each call gets its own staging and command buffer), `FrameManager` as a
  whole, `DescriptorSetLayoutCache`, and any individual `CommandPool` (allocate
  from separate pools per thread instead).
- **Queue submission**: `vkQueueSubmit` must not run concurrently on the same
  queue, so serialize calls to the `Context::submit*` family that target the
  same queue (different queues are fine).
- **Image layout tracking** is CPU-side state: the thread that records
  transitions for an `Image` must be the one that reads `layout()`.

## Staging and transfer lifetime

`upload()` and `download()` copy through one shared staging buffer that grows to
the largest request seen so far and never shrinks, so a single large transfer
keeps that capacity for the lifetime of the `Context`. The copies themselves are
synchronous: when the call returns, the host bytes are complete in the
destination (or in your buffer), so nothing you receive aliases the staging
buffer and you do not have to copy it out before the next call. The capacity, on
the other hand, is retained, which is worth knowing before uploading a one-off
large texture.

`upload_async()` differs: it gives each call a private staging buffer, so the
source bytes only need to stay valid until the call returns, and the buffer is
released once the transfer timeline passes its value.

## Image layouts

vulcao tracks an image's layout on the CPU so barriers can fill in their stage
and access masks automatically. `Image::layout()` is that bookkeeping, not
something read back from the driver: it starts at the image's `initialLayout`
and the helpers that take an `Image&` — `transition()`, `generate_mipmaps()`,
`transition_to_render()` — update it as they record. The raw-handle overloads
take a bare `vk::Image` and so cannot see or update it. Prefer the `Image&`
forms; if you do transition through a raw handle, keep `Image::layout()` in step
yourself, because a stale value makes later barriers record the wrong
`oldLayout` and the resulting hazard is silent outside the validation layers.

## Documentation

The API reference is generated with Doxygen from `docs/`:

```sh
cmake -S . -B build -DVULCAO_BUILD_DOCS=ON
cmake --build build --target vulcao_docs
```

The HTML output is written to `docs/generated/html` and is published to GitHub
Pages by CI. The library itself is built and tested by CI on every push.

## Using from another project

```sh
cmake --install build --prefix <prefix>
```

```cmake
find_package(vulcao REQUIRED)
target_link_libraries(app PRIVATE vulcao::vulcao)
```

The installed package is self-contained: it bundles vk-bootstrap, SPIRV-Reflect and
the VMA headers.

## Example

```cpp
#include <vulcao/context.h>
#include <vulcao/shader_module.h>

vulcao::Context ctx{{.app_name = "my-app"}};
ctx.initialize(surface, extent);

auto shader = vulcao::ShaderModule::create_from_file(
    ctx.device(), vk::ShaderStageFlagBits::eCompute, "reduce.comp.spv");

// reflection-driven layouts
auto layout = vulcao::PipelineLayout::create_from_reflection(
    ctx.device(), std::span(&shader.reflection(), 1));
auto pipeline = vulcao::Pipeline::create_compute(ctx.device(), layout, shader, "compMain");
```

## License

MIT, see [LICENSE](LICENSE).
