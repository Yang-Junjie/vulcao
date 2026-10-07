#include <stdexcept>
#include <utility>

#include <doctest/doctest.h>

#include <vulcao/context.h>
#include <vulcao/surface.h>

#include "common.h"

// The suite runs headless, so it cannot create a real presentation surface: no
// platform window backend is initialized and no surface extension is enabled.
// What is reachable here is the ownership contract itself, which is what callers
// get wrong: adopting nothing must not become a second error path, destroying
// must be idempotent, and moving must not leave two owners behind.
TEST_CASE("an empty surface owns nothing and survives every ownership operation") {
    VULCAO_REQUIRE_DEVICE();

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;
    vulcao::Context context{info};
    context.initialize();

    // A platform layer that could not create a surface reports it as null; the
    // caller then holds an empty wrapper instead of branching on a raw handle.
    vulcao::Surface empty = vulcao::Surface::adopt(context.instance(), vk::SurfaceKHR{});
    CHECK(!empty.valid());
    CHECK(!empty);
    CHECK(!empty.handle());

    // Destroying is a no-op and stays a no-op when repeated.
    empty.destroy();
    empty.destroy();
    CHECK(!empty.valid());

    // Moving out leaves the source empty rather than a second owner of the same
    // handle, so only one destructor can ever release it.
    vulcao::Surface moved = std::move(empty);
    CHECK(!moved.valid());
    CHECK(!empty.valid());

    // Move assignment releases the target's previous surface first; with two
    // empty wrappers that is the same no-op path.
    vulcao::Surface assigned = vulcao::Surface::adopt(context.instance(), vk::SurfaceKHR{});
    assigned = std::move(moved);
    CHECK(!assigned.valid());
    CHECK(!moved.valid());
}

TEST_CASE("adopting a non-null surface without an instance is rejected") {
    // An instance is what destruction is issued through, so accepting a surface
    // without one would create a handle nothing can ever free.
    CHECK_THROWS_AS(vulcao::Surface::adopt(vk::Instance{}, vk::SurfaceKHR{}), std::runtime_error);
}
