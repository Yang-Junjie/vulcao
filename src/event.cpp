#include "vulcao/event.h"

#include <stdexcept>
#include <utility>

namespace vulcao {

Event::~Event() {
    destroy();
}

Event::Event(Event&& other) noexcept
    : device_(std::exchange(other.device_, vk::Device{})),
      event_(std::exchange(other.event_, vk::Event{})) {}

Event& Event::operator=(Event&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = std::exchange(other.device_, vk::Device{});
        event_ = std::exchange(other.event_, vk::Event{});
    }
    return *this;
}

Event Event::create(vk::Device device, vk::EventCreateFlags flags) {
    Event event;
    event.device_ = device;
    event.event_ = device.createEvent(vk::EventCreateInfo{.flags = flags});
    return event;
}

void Event::set() const {
    device_.setEvent(event_);
}

void Event::reset() const {
    device_.resetEvent(event_);
}

bool Event::signaled() const {
    const vk::Result result = device_.getEventStatus(event_);
    if (result == vk::Result::eEventSet)
        return true;
    if (result == vk::Result::eEventReset)
        return false;
    throw std::runtime_error("Event::signaled: " + vk::to_string(result));
}

void Event::destroy() {
    if (event_)
        device_.destroyEvent(event_);

    device_ = nullptr;
    event_ = nullptr;
}

}
