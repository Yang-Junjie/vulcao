#include "vulcao/semaphore.h"

#include "vulcao/detail/check.h"

#include <utility>

namespace vulcao {

Semaphore::~Semaphore() {
    destroy();
}

Semaphore::Semaphore(Semaphore&& other) noexcept
    : device_(std::exchange(other.device_, vk::Device{})),
      semaphore_(std::exchange(other.semaphore_, vk::Semaphore{})),
      type_(std::exchange(other.type_, vk::SemaphoreType::eBinary)) {}

Semaphore& Semaphore::operator=(Semaphore&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = std::exchange(other.device_, vk::Device{});
        semaphore_ = std::exchange(other.semaphore_, vk::Semaphore{});
        type_ = std::exchange(other.type_, vk::SemaphoreType::eBinary);
    }
    return *this;
}

Semaphore Semaphore::create(vk::Device device, vk::SemaphoreCreateFlags flags) {
    Semaphore semaphore;
    semaphore.device_ = device;
    semaphore.semaphore_ = device.createSemaphore(vk::SemaphoreCreateInfo{.flags = flags});
    semaphore.type_ = vk::SemaphoreType::eBinary;
    return semaphore;
}

Semaphore Semaphore::create_timeline(vk::Device device, uint64_t initial_value) {
    Semaphore semaphore;
    semaphore.device_ = device;

    const vk::SemaphoreTypeCreateInfo type_info{
        .semaphoreType = vk::SemaphoreType::eTimeline,
        .initialValue = initial_value,
    };
    semaphore.semaphore_ =
        device.createSemaphore(vk::SemaphoreCreateInfo{.pNext = &type_info});
    semaphore.type_ = vk::SemaphoreType::eTimeline;
    return semaphore;
}

uint64_t Semaphore::value() const {
    return device_.getSemaphoreCounterValue(semaphore_);
}

void Semaphore::signal(uint64_t value) const {
    device_.signalSemaphore(vk::SemaphoreSignalInfo{
        .semaphore = semaphore_,
        .value = value,
    });
}

void Semaphore::wait(uint64_t value, uint64_t timeout) const {
    const vk::SemaphoreWaitInfo wait_info{
        .semaphoreCount = 1,
        .pSemaphores = &semaphore_,
        .pValues = &value,
    };
    detail::check(device_.waitSemaphores(wait_info, timeout), "wait semaphore");
}

void Semaphore::destroy() {
    if (semaphore_)
        device_.destroySemaphore(semaphore_);

    device_ = nullptr;
    semaphore_ = nullptr;
    type_ = vk::SemaphoreType::eBinary;
}

}
