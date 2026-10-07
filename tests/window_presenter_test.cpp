#include <stdexcept>

#include <doctest/doctest.h>

#include <vulcao/context.h>
#include <vulcao/surface.h>
#include <vulcao/window_presenter.h>

#include "common.h"

// This case also pins header hygiene: it includes <vulcao/window_presenter.h>
// directly and uses nothing from vulcao beyond Context and Surface, so the
// header must be self-contained.
//
// The suite runs headless, so there is no surface to present to and the
// acquire/present paths stay untested here; one empty Surface is enough to pin
// the constructor's fail-fast contract.
TEST_CASE("a window presenter rejects a missing surface before touching the device") {
    VULCAO_REQUIRE_DEVICE();

    vulcao::ContextInfo info;
    info.headless = true;
    info.validation = true;
    vulcao::Context context{info};
    context.initialize();

    vulcao::Surface empty;
    CHECK_THROWS_AS(vulcao::WindowPresenter(context, empty, vulcao::WindowPresenterInfo{}),
                    std::runtime_error);

    // Rejecting it must leave the context usable and must not have adopted the
    // surface, so a second attempt fails the same way.
    CHECK_THROWS_AS(vulcao::WindowPresenter(context, empty, vulcao::WindowPresenterInfo{}),
                    std::runtime_error);
    CHECK(context.initialized());
    CHECK(!context.surface());
}
