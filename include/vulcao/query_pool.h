#pragma once

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.hpp>

namespace vulcao {

/// @brief RAII wrapper around a Vulkan query pool.
class QueryPool {
public:
    /// @brief Creates an empty query pool.
    QueryPool() = default;

    /// @brief Destroys the query pool.
    ~QueryPool();

    /// @brief Not copyable.
    QueryPool(const QueryPool&) = delete;
    QueryPool& operator=(const QueryPool&) = delete;

    /// @brief Moves the query pool, leaving the source empty.
    QueryPool(QueryPool&& other) noexcept;

    /// @brief Move assignment. Destroys the current query pool first.
    QueryPool& operator=(QueryPool&& other) noexcept;

    /// @brief Creates a query pool.
    /// @param device Device that creates the pool.
    /// @param type Type of the queries.
    /// @param query_count Number of queries in the pool.
    /// @param flags Query pool creation flags.
    /// @return The created query pool.
    static QueryPool create(vk::Device device,
                            vk::QueryType type,
                            uint32_t query_count,
                            vk::QueryPoolCreateFlags flags = {});

    /// @brief Returns true if the query pool holds a valid handle.
    bool valid() const { return static_cast<bool>(pool_); }

    /// @brief Returns true if the query pool holds a valid handle.
    explicit operator bool() const { return valid(); }

    /// @brief Returns the raw Vulkan query pool handle.
    vk::QueryPool handle() const { return pool_; }

    /// @brief Returns the number of queries in the pool.
    uint32_t query_count() const { return query_count_; }

    /// @brief Returns the type of the queries.
    vk::QueryType type() const { return type_; }

    /// @brief Resets a range of queries from the host.
    /// @param first_query First query to reset.
    /// @param query_count Number of queries to reset.
    void reset(uint32_t first_query, uint32_t query_count) const;

    /// @brief Reads one 64 bit query result.
    /// @param query Query index.
    /// @param flags Query result flags; add eWait to block until the result is ready.
    /// @return The query result.
    /// @throws std::runtime_error if the result is unavailable or the read fails.
    uint64_t result(uint32_t query,
                    vk::QueryResultFlags flags = vk::QueryResultFlagBits::e64) const;

    /// @brief Reads a range of 64 bit query results.
    /// @param first_query First query to read.
    /// @param query_count Number of queries to read.
    /// @param flags Query result flags; add eWait to block until the results are ready.
    /// @return One result per query.
    /// @throws std::runtime_error if a result is unavailable or the read fails.
    std::vector<uint64_t> results(uint32_t first_query,
                                  uint32_t query_count,
                                  vk::QueryResultFlags flags = vk::QueryResultFlagBits::e64) const;

    /// @brief Destroys the query pool and resets the wrapper.
    void destroy();

private:
    vk::Device device_;
    vk::QueryPool pool_;
    uint32_t query_count_ = 0;
    vk::QueryType type_ = vk::QueryType::eOcclusion;
};

}
