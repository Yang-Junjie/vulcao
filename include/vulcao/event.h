#pragma once

#include <vulkan/vulkan.hpp>

namespace vulcao {

/// @brief RAII wrapper around a Vulkan event.
///
/// Events synchronize between the host and the device or between two device
/// queues. The device side is recorded through CommandBuffer::set_event,
/// reset_event and wait_event; the host side through set, reset and signaled.
class Event {
public:
    /// @brief Creates an empty event.
    Event() = default;

    /// @brief Destroys the event.
    ~Event();

    /// @brief Not copyable.
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    /// @brief Moves the event, leaving the source empty.
    Event(Event&& other) noexcept;

    /// @brief Move assignment. Destroys the current event first.
    Event& operator=(Event&& other) noexcept;

    /// @brief Creates an event.
    /// @param device Device that creates the event.
    /// @param flags Event creation flags.
    /// @return The created event.
    /// @throws std::runtime_error if creation fails.
    static Event create(vk::Device device, vk::EventCreateFlags flags = {});

    /// @brief Returns true if the event holds a valid handle.
    bool valid() const { return static_cast<bool>(event_); }

    /// @brief Returns true if the event holds a valid handle.
    explicit operator bool() const { return valid(); }

    /// @brief Returns the raw Vulkan event handle.
    vk::Event handle() const { return event_; }

    /// @brief Sets the event from the host.
    void set() const;

    /// @brief Resets the event from the host.
    void reset() const;

    /// @brief Returns true if the event is currently signaled.
    bool signaled() const;

    /// @brief Destroys the event and resets the wrapper.
    void destroy();

private:
    vk::Device device_;
    vk::Event event_;
};

}
