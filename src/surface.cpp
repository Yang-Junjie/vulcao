#include "vulcao/surface.h"

#include <stdexcept>
#include <utility>

namespace vulcao {

Surface::~Surface() { destroy(); }

Surface::Surface(Surface&& other) noexcept
        : instance_(std::exchange(other.instance_, vk::Instance{})),
          surface_(std::exchange(other.surface_, vk::SurfaceKHR{})) {}

Surface& Surface::operator=(Surface&& other) noexcept {
    if (this != &other) {
        destroy();
        instance_ = std::exchange(other.instance_, vk::Instance{});
        surface_ = std::exchange(other.surface_, vk::SurfaceKHR{});
    }
    return *this;
}

Surface Surface::adopt(vk::Instance instance, vk::SurfaceKHR surface) {
    // Destruction is issued through the instance, so adopting without one would
    // create a handle that nothing can ever free.
    if (!instance)
        throw std::runtime_error("Surface::adopt: an instance is required to destroy the surface");

    Surface result;
    result.instance_ = instance;
    result.surface_ = surface;
    return result;
}

void Surface::destroy() {
    if (surface_ && instance_)
        instance_.destroySurfaceKHR(surface_);
    instance_ = nullptr;
    surface_ = nullptr;
}

} // namespace vulcao
