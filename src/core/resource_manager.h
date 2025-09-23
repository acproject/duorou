#pragma once

#include <string>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <functional>
#include <atomic>
#include <thread>

namespace duorou {
namespace core {

/**
 * @brief 资源类型枚举
 */
enum class ResourceType {
    MODEL,          ///< Model resource
    GPU_MEMORY,     ///< GPU memory resource
    CPU_MEMORY,     ///< CPU memory resource
    COMPUTE_UNIT,   ///< Compute unit resource
    STORAGE,        ///< Storage resource
    NETWORK         ///< Network resource
};

/**
 * @brief 资源锁定模式
 */
enum class LockMode {
    SHARED,         ///< Shared lock (read lock)
    EXCLUSIVE       ///< Exclusive lock (write lock)
};

/**
 * @brief 资源信息
 */
struct ResourceInfo {
    std::string id;                 ///< Resource ID
    ResourceType type;              ///< Resource type
    std::string name;               ///< Resource name
    size_t capacity;                ///< Resource capacity
    size_t used;                    ///< Used amount
    bool available;                 ///< Whether available
    std::chrono::system_clock::time_point last_accessed;  ///< Last access time
    std::unordered_set<std::string> holders;  ///< Holder list
    
    ResourceInfo() : capacity(0), used(0), available(true) {}
};

/**
 * @brief 资源锁定信息
 */
struct ResourceLock {
    std::string resource_id;        ///< Resource ID
    std::string holder_id;          ///< Holder ID
    LockMode mode;                  ///< Lock mode
    std::chrono::system_clock::time_point acquired_time;  ///< Acquisition time
    std::chrono::milliseconds timeout;  ///< Timeout duration
    
    ResourceLock() : mode(LockMode::SHARED), timeout(0) {}
};

/**
 * @brief 资源预留信息
 */
struct ResourceReservation {
    std::string resource_id;        ///< Resource ID
    std::string requester_id;       ///< Requester ID
    size_t amount;                  ///< Reserved amount
    std::chrono::system_clock::time_point reserved_time;  ///< Reserved time
    std::chrono::milliseconds duration;  ///< Reservation duration
    
    ResourceReservation() : amount(0), duration(0) {}
};

/**
 * @brief Resource Manager
 */
class ResourceManager {
public:
    /**
     * @brief Constructor
     */
    ResourceManager();
    
    /**
     * @brief Destructor
     */
    ~ResourceManager();
    
    /**
     * @brief Register resource
     * @param resource_info Resource information
     * @return Whether successful
     */
    bool registerResource(const ResourceInfo& resource_info);
    
    /**
     * @brief Unregister resource
     * @param resource_id Resource ID
     * @return Whether successful
     */
    bool unregisterResource(const std::string& resource_id);
    
    /**
     * @brief Acquire resource lock
     * @param resource_id Resource ID
     * @param holder_id Holder ID
     * @param mode Lock mode
     * @param timeout_ms Timeout in milliseconds
     * @return Whether lock acquired successfully
     */
    bool acquireLock(const std::string& resource_id, 
                     const std::string& holder_id,
                     LockMode mode,
                     int timeout_ms = 5000);
    
    /**
     * @brief Release resource lock
     * @param resource_id Resource ID
     * @param holder_id Holder ID
     * @return Whether lock released successfully
     */
    bool releaseLock(const std::string& resource_id, const std::string& holder_id);
    
    /**
     * @brief Reserve resource
     * @param resource_id Resource ID
     * @param requester_id Requester ID
     * @param amount Reserved amount
     * @param duration_ms Reservation duration in milliseconds
     * @return Whether reservation successful
     */
    bool reserveResource(const std::string& resource_id,
                        const std::string& requester_id,
                        size_t amount,
                        int duration_ms = 30000);
    
    /**
     * @brief Release resource reservation
     * @param resource_id Resource ID
     * @param requester_id Requester ID
     * @return Whether release successful
     */
    bool releaseReservation(const std::string& resource_id, const std::string& requester_id);
    
    /**
     * @brief Check if resource is available
     * @param resource_id Resource ID
     * @param mode Lock mode
     * @return Whether available
     */
    bool isResourceAvailable(const std::string& resource_id, LockMode mode = LockMode::SHARED) const;
    
    /**
     * @brief Get resource information
     * @param resource_id Resource ID
     * @return Resource information
     */
    ResourceInfo getResourceInfo(const std::string& resource_id) const;
    
    /**
     * @brief Get resource utilization
     * @param resource_id Resource ID
     * @return Utilization rate (0.0-1.0)
     */
    double getResourceUtilization(const std::string& resource_id) const;
    
    /**
     * @brief Get all resource list
     * @param type Resource type filter (optional)
     * @return Resource ID list
     */
    std::vector<std::string> getResourceList(ResourceType type = ResourceType::MODEL) const;
    
    /**
     * @brief Clean up expired locks and reservations
     */
    void cleanupExpiredLocks();
    
    /**
     * @brief Set resource status change callback
     * @param callback Callback function
     */
    void setResourceStatusCallback(std::function<void(const std::string&, bool)> callback);
    
    /**
     * @brief Get resource statistics
     * @return Statistics mapping
     */
    std::unordered_map<std::string, size_t> getResourceStatistics() const;
    
    /**
     * @brief Force release all locks held by holder
     * @param holder_id Holder ID
     * @return Number of locks released
     */
    size_t forceReleaseHolderLocks(const std::string& holder_id);
    
    /**
     * @brief Check for deadlock
     * @return Whether deadlock exists
     */
    bool detectDeadlock() const;
    
    /**
     * @brief Get waiting queue length
     * @param resource_id Resource ID
     * @return Waiting queue length
     */
    size_t getWaitingQueueLength(const std::string& resource_id) const;
    
private:
    /**
     * @brief Check lock compatibility
     * @param resource_id Resource ID
     * @param mode Requested lock mode
     * @return Whether compatible
     */
    bool isLockCompatible(const std::string& resource_id, LockMode mode) const;
    
    /**
     * @brief Clean up expired reservations
     */
    void cleanupExpiredReservations();
    
    /**
     * @brief Start cleanup thread
     */
    void startCleanupThread();
    
    /**
     * @brief Stop cleanup thread
     */
    void stopCleanupThread();
    
    /**
     * @brief Cleanup thread function
     */
    void cleanupThreadFunc();
    
private:
    mutable std::mutex resources_mutex_;                    ///< Resource mutex
    mutable std::mutex locks_mutex_;                        ///< Lock mutex
    mutable std::mutex reservations_mutex_;                 ///< Reservation mutex
    
    std::unordered_map<std::string, ResourceInfo> resources_;           ///< Resource mapping
    std::unordered_map<std::string, std::vector<ResourceLock>> locks_;   ///< Lock mapping
    std::unordered_map<std::string, std::vector<ResourceReservation>> reservations_;  ///< Reservation mapping
    
    std::unordered_map<std::string, std::condition_variable> wait_conditions_;  ///< Wait condition variables
    std::unordered_map<std::string, size_t> waiting_counts_;            ///< Waiting counts
    
    std::function<void(const std::string&, bool)> status_callback_;     ///< Status callback
    
    std::atomic<bool> cleanup_running_;                     ///< Cleanup thread running flag
    std::thread cleanup_thread_;                            ///< Cleanup thread
    std::condition_variable cleanup_condition_;             ///< Cleanup condition variable
    std::mutex cleanup_mutex_;                              ///< Cleanup mutex
};

/**
 * @brief RAII Resource Lock
 */
class ResourceLockGuard {
public:
    /**
     * @brief Constructor
     * @param manager Resource manager
     * @param resource_id Resource ID
     * @param holder_id Holder ID
     * @param mode Lock mode
     * @param timeout_ms Timeout in milliseconds
     */
    ResourceLockGuard(ResourceManager& manager,
                     const std::string& resource_id,
                     const std::string& holder_id,
                     LockMode mode = LockMode::SHARED,
                     int timeout_ms = 5000);
    
    /**
     * @brief Destructor
     */
    ~ResourceLockGuard();
    
    /**
     * @brief Check if lock acquired successfully
     * @return Whether successful
     */
    bool isLocked() const { return locked_; }
    
    /**
     * @brief Manually release lock
     */
    void unlock();
    
    // Disable copy and assignment
    ResourceLockGuard(const ResourceLockGuard&) = delete;
    ResourceLockGuard& operator=(const ResourceLockGuard&) = delete;
    
private:
    ResourceManager& manager_;
    std::string resource_id_;
    std::string holder_id_;
    bool locked_;
};

} // namespace core
} // namespace duorou