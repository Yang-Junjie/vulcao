#include "vulcao/query_pool.h"

#include "vulcao/detail/check.h"

#include <utility>

namespace vulcao {

QueryPool::~QueryPool() {
    destroy();
}

QueryPool::QueryPool(QueryPool&& other) noexcept
    : device_(std::exchange(other.device_, vk::Device{})),
      pool_(std::exchange(other.pool_, vk::QueryPool{})),
      query_count_(std::exchange(other.query_count_, 0)),
      type_(std::exchange(other.type_, vk::QueryType::eOcclusion)) {}

QueryPool& QueryPool::operator=(QueryPool&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = std::exchange(other.device_, vk::Device{});
        pool_ = std::exchange(other.pool_, vk::QueryPool{});
        query_count_ = std::exchange(other.query_count_, 0);
        type_ = std::exchange(other.type_, vk::QueryType::eOcclusion);
    }
    return *this;
}

QueryPool QueryPool::create(vk::Device device,
                            vk::QueryType type,
                            uint32_t query_count,
                            vk::QueryPoolCreateFlags flags) {
    QueryPool query_pool;
    query_pool.device_ = device;
    query_pool.pool_ = device.createQueryPool(vk::QueryPoolCreateInfo{
        .flags = flags,
        .queryType = type,
        .queryCount = query_count,
    });
    query_pool.query_count_ = query_count;
    query_pool.type_ = type;
    return query_pool;
}

void QueryPool::reset(uint32_t first_query, uint32_t query_count) const {
    device_.resetQueryPool(pool_, first_query, query_count);
}

uint64_t QueryPool::result(uint32_t query, vk::QueryResultFlags flags) const {
    return results(query, 1, flags).front();
}

std::vector<uint64_t> QueryPool::results(uint32_t first_query,
                                         uint32_t query_count,
                                         vk::QueryResultFlags flags) const {
    std::vector<uint64_t> values(query_count);
    if (query_count == 0)
        return values;

    detail::check(device_.getQueryPoolResults(pool_, first_query, query_count,
                                              values.size() * sizeof(uint64_t), values.data(),
                                              sizeof(uint64_t), flags),
                  "get query pool results");
    return values;
}

void QueryPool::destroy() {
    if (pool_)
        device_.destroyQueryPool(pool_);

    device_ = nullptr;
    pool_ = nullptr;
    query_count_ = 0;
    type_ = vk::QueryType::eOcclusion;
}

}
