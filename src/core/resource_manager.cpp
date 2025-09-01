#include "resource_manager.h"
#include <iostream>
#include <algorithm>
#include <thread>

namespace duorou {
namespace core {

// ResourceManager实现

ResourceManager::ResourceManager() : cleanup_running_(false) {
    startCleanupThread();
}

ResourceManager::~ResourceManager() {
    stopCleanupThread();
}

bool ResourceManager::registerResource(const ResourceInfo& resource_info) {
    std::lock_guard<std::mutex> lock(resources_mutex_);
    
    if (resources_.find(resource_info.id) != resources_.end()) {
        std::cerr << "Resource already registered: " << resource_info.id << std::endl;
        return false;
    }
    
    resources_[resource_info.id] = resource_info;
    resources_[resource_info.id].last_accessed = std::chrono::system_clock::now();
    
    std::cout << "Resource registered: " << resource_info.id << " (" << resource_info.name << ")" << std::endl;
    return true;
}

bool ResourceManager::unregisterResource(const std::string& resource_id) {
    std::lock_guard<std::mutex> resources_lock(resources_mutex_);
    std::lock_guard<std::mutex> locks_lock(locks_mutex_);
    std::lock_guard<std::mutex> reservations_lock(reservations_mutex_);
    
    auto it = resources_.find(resource_id);
    if (it == resources_.end()) {
        return false;
    }
    
    // 强制释放所有相关的锁
    locks_.erase(resource_id);
    
    // 清理预留
    reservations_.erase(resource_id);
    
    // 清理等待条件
    wait_conditions_.erase(resource_id);
    waiting_counts_.erase(resource_id);
    
    resources_.erase(it);
    
    std::cout << "Resource unregistered: " << resource_id << std::endl;
    return true;
}

bool ResourceManager::acquireLock(const std::string& resource_id, 
                                 const std::string& holder_id,
                                 LockMode mode,
                                 int timeout_ms) {
    std::unique_lock<std::mutex> lock(locks_mutex_);
    
    // 检查资源是否存在
    {
        std::lock_guard<std::mutex> resources_lock(resources_mutex_);
        if (resources_.find(resource_id) == resources_.end()) {
            std::cerr << "Resource not found: " << resource_id << std::endl;
            return false;
        }
    }
    
    auto timeout_time = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    
    // 等待直到可以获取锁
    while (!isLockCompatible(resource_id, mode)) {
        if (timeout_ms > 0) {
            waiting_counts_[resource_id]++;
            
            if (wait_conditions_[resource_id].wait_until(lock, timeout_time) == std::cv_status::timeout) {
                waiting_counts_[resource_id]--;
                std::cerr << "Lock acquisition timeout for resource: " << resource_id << std::endl;
                return false;
            }
            
            waiting_counts_[resource_id]--;
        } else {
            waiting_counts_[resource_id]++;
            wait_conditions_[resource_id].wait(lock);
            waiting_counts_[resource_id]--;
        }
    }
    
    // 获取锁
    ResourceLock resource_lock;
    resource_lock.resource_id = resource_id;
    resource_lock.holder_id = holder_id;
    resource_lock.mode = mode;
    resource_lock.acquired_time = std::chrono::system_clock::now();
    resource_lock.timeout = std::chrono::milliseconds(timeout_ms > 0 ? timeout_ms : 300000); // 默认5分钟超时
    
    locks_[resource_id].push_back(resource_lock);
    
    // 更新资源信息
    {
        std::lock_guard<std::mutex> resources_lock(resources_mutex_);
        auto& resource = resources_[resource_id];
        resource.holders.insert(holder_id);
        resource.last_accessed = std::chrono::system_clock::now();
    }
    
    std::cout << "Lock acquired: " << resource_id << " by " << holder_id 
              << " (mode: " << (mode == LockMode::SHARED ? "SHARED" : "EXCLUSIVE") << ")" << std::endl;
    
    return true;
}

bool ResourceManager::releaseLock(const std::string& resource_id, const std::string& holder_id) {
    std::lock_guard<std::mutex> lock(locks_mutex_);
    
    auto it = locks_.find(resource_id);
    if (it == locks_.end()) {
        return false;
    }
    
    auto& resource_locks = it->second;
    auto lock_it = std::find_if(resource_locks.begin(), resource_locks.end(),
        [&holder_id](const ResourceLock& lock) {
            return lock.holder_id == holder_id;
        });
    
    if (lock_it == resource_locks.end()) {
        return false;
    }
    
    resource_locks.erase(lock_it);
    
    // 更新资源信息
    {
        std::lock_guard<std::mutex> resources_lock(resources_mutex_);
        auto resource_it = resources_.find(resource_id);
        if (resource_it != resources_.end()) {
            resource_it->second.holders.erase(holder_id);
        }
    }
    
    // 通知等待的线程
    wait_conditions_[resource_id].notify_all();
    
    std::cout << "Lock released: " << resource_id << " by " << holder_id << std::endl;
    
    return true;
}

bool ResourceManager::reserveResource(const std::string& resource_id,
                                     const std::string& requester_id,
                                     size_t amount,
                                     int duration_ms) {
    std::lock_guard<std::mutex> reservations_lock(reservations_mutex_);
    std::lock_guard<std::mutex> resources_lock(resources_mutex_);
    
    auto resource_it = resources_.find(resource_id);
    if (resource_it == resources_.end()) {
        return false;
    }
    
    auto& resource = resource_it->second;
    if (resource.used + amount > resource.capacity) {
        std::cerr << "Insufficient resource capacity: " << resource_id << std::endl;
        return false;
    }
    
    ResourceReservation reservation;
    reservation.resource_id = resource_id;
    reservation.requester_id = requester_id;
    reservation.amount = amount;
    reservation.reserved_time = std::chrono::system_clock::now();
    reservation.duration = std::chrono::milliseconds(duration_ms);
    
    reservations_[resource_id].push_back(reservation);
    resource.used += amount;
    
    std::cout << "Resource reserved: " << resource_id << " (" << amount << " units) by " << requester_id << std::endl;
    
    return true;
}

bool ResourceManager::releaseReservation(const std::string& resource_id, const std::string& requester_id) {
    std::lock_guard<std::mutex> reservations_lock(reservations_mutex_);
    std::lock_guard<std::mutex> resources_lock(resources_mutex_);
    
    auto it = reservations_.find(resource_id);
    if (it == reservations_.end()) {
        return false;
    }
    
    auto& reservations = it->second;
    auto reservation_it = std::find_if(reservations.begin(), reservations.end(),
        [&requester_id](const ResourceReservation& reservation) {
            return reservation.requester_id == requester_id;
        });
    
    if (reservation_it == reservations.end()) {
        return false;
    }
    
    size_t amount = reservation_it->amount;
    reservations.erase(reservation_it);
    
    // 更新资源使用量
    auto resource_it = resources_.find(resource_id);
    if (resource_it != resources_.end()) {
        resource_it->second.used -= amount;
    }
    
    std::cout << "Resource reservation released: " << resource_id << " (" << amount << " units) by " << requester_id << std::endl;
    
    return true;
}

bool ResourceManager::isResourceAvailable(const std::string& resource_id, LockMode mode) const {
    std::lock_guard<std::mutex> resources_lock(resources_mutex_);
    std::lock_guard<std::mutex> locks_lock(locks_mutex_);
    
    auto resource_it = resources_.find(resource_id);
    if (resource_it == resources_.end() || !resource_it->second.available) {
        return false;
    }
    
    return isLockCompatible(resource_id, mode);
}

ResourceInfo ResourceManager::getResourceInfo(const std::string& resource_id) const {
    std::lock_guard<std::mutex> lock(resources_mutex_);
    
    auto it = resources_.find(resource_id);
    if (it != resources_.end()) {
        return it->second;
    }
    
    return ResourceInfo{};
}

double ResourceManager::getResourceUtilization(const std::string& resource_id) const {
    std::lock_guard<std::mutex> lock(resources_mutex_);
    
    auto it = resources_.find(resource_id);
    if (it != resources_.end() && it->second.capacity > 0) {
        return static_cast<double>(it->second.used) / it->second.capacity;
    }
    
    return 0.0;
}

std::vector<std::string> ResourceManager::getResourceList(ResourceType type) const {
    std::lock_guard<std::mutex> lock(resources_mutex_);
    
    std::vector<std::string> result;
    for (const auto& pair : resources_) {
        if (pair.second.type == type) {
            result.push_back(pair.first);
        }
    }
    
    return result;
}

void ResourceManager::cleanupExpiredLocks() {
    std::lock_guard<std::mutex> lock(locks_mutex_);
    
    auto now = std::chrono::system_clock::now();
    
    for (auto& pair : locks_) {
        auto& locks = pair.second;
        locks.erase(std::remove_if(locks.begin(), locks.end(),
            [now](const ResourceLock& lock) {
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lock.acquired_time);
                return elapsed > lock.timeout;
            }), locks.end());
    }
    
    cleanupExpiredReservations();
}

void ResourceManager::setResourceStatusCallback(std::function<void(const std::string&, bool)> callback) {
    status_callback_ = callback;
}

std::unordered_map<std::string, size_t> ResourceManager::getResourceStatistics() const {
    std::lock_guard<std::mutex> resources_lock(resources_mutex_);
    std::lock_guard<std::mutex> locks_lock(locks_mutex_);
    
    std::unordered_map<std::string, size_t> stats;
    stats["total_resources"] = resources_.size();
    
    size_t total_locks = 0;
    size_t total_waiting = 0;
    
    for (const auto& pair : locks_) {
        total_locks += pair.second.size();
    }
    
    for (const auto& pair : waiting_counts_) {
        total_waiting += pair.second;
    }
    
    stats["total_locks"] = total_locks;
    stats["total_waiting"] = total_waiting;
    
    return stats;
}

size_t ResourceManager::forceReleaseHolderLocks(const std::string& holder_id) {
    std::lock_guard<std::mutex> lock(locks_mutex_);
    
    size_t released_count = 0;
    
    for (auto& pair : locks_) {
        auto& locks = pair.second;
        auto original_size = locks.size();
        
        locks.erase(std::remove_if(locks.begin(), locks.end(),
            [&holder_id](const ResourceLock& lock) {
                return lock.holder_id == holder_id;
            }), locks.end());
        
        released_count += (original_size - locks.size());
        
        // 通知等待的线程
        if (original_size != locks.size()) {
            wait_conditions_[pair.first].notify_all();
        }
    }
    
    // 更新资源持有者信息
    {
        std::lock_guard<std::mutex> resources_lock(resources_mutex_);
        for (auto& pair : resources_) {
            pair.second.holders.erase(holder_id);
        }
    }
    
    if (released_count > 0) {
        std::cout << "Force released " << released_count << " locks for holder: " << holder_id << std::endl;
    }
    
    return released_count;
}

bool ResourceManager::detectDeadlock() const {
    // 简单的死锁检测：检查是否有循环等待
    // 这里实现一个基础版本，实际应用中可能需要更复杂的算法
    
    std::lock_guard<std::mutex> lock(locks_mutex_);
    
    // 如果有资源被多个持有者等待，且这些持有者互相持有对方需要的资源，则可能存在死锁
    for (const auto& pair : waiting_counts_) {
        if (pair.second > 1) {
            // 简单检测：如果等待队列过长，可能存在死锁风险
            if (pair.second > 10) {
                std::cerr << "Potential deadlock detected for resource: " << pair.first 
                         << " (waiting count: " << pair.second << ")" << std::endl;
                return true;
            }
        }
    }
    
    return false;
}

size_t ResourceManager::getWaitingQueueLength(const std::string& resource_id) const {
    std::lock_guard<std::mutex> lock(locks_mutex_);
    
    auto it = waiting_counts_.find(resource_id);
    if (it != waiting_counts_.end()) {
        return it->second;
    }
    
    return 0;
}

bool ResourceManager::isLockCompatible(const std::string& resource_id, LockMode mode) const {
    auto it = locks_.find(resource_id);
    if (it == locks_.end() || it->second.empty()) {
        return true; // 没有现有锁，兼容
    }
    
    const auto& existing_locks = it->second;
    
    if (mode == LockMode::SHARED) {
        // 共享锁只与其他共享锁兼容
        for (const auto& lock : existing_locks) {
            if (lock.mode == LockMode::EXCLUSIVE) {
                return false;
            }
        }
        return true;
    } else {
        // 独占锁与任何锁都不兼容
        return false;
    }
}

void ResourceManager::cleanupExpiredReservations() {
    std::lock_guard<std::mutex> reservations_lock(reservations_mutex_);
    std::lock_guard<std::mutex> resources_lock(resources_mutex_);
    
    auto now = std::chrono::system_clock::now();
    
    for (auto& pair : reservations_) {
        auto& reservations = pair.second;
        const std::string& resource_id = pair.first;
        
        auto original_size = reservations.size();
        size_t released_amount = 0;
        
        auto new_end = std::remove_if(reservations.begin(), reservations.end(),
            [now, &released_amount](const ResourceReservation& reservation) {
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - reservation.reserved_time);
                if (elapsed > reservation.duration) {
                    released_amount += reservation.amount;
                    return true;
                }
                return false;
            });
        
        reservations.erase(new_end, reservations.end());
        
        // 更新资源使用量
        if (released_amount > 0) {
            auto resource_it = resources_.find(resource_id);
            if (resource_it != resources_.end()) {
                resource_it->second.used -= released_amount;
            }
        }
    }
}

void ResourceManager::startCleanupThread() {
    cleanup_running_.store(true);
    cleanup_thread_ = std::thread(&ResourceManager::cleanupThreadFunc, this);
}

void ResourceManager::stopCleanupThread() {
    if (cleanup_running_.load()) {
        cleanup_running_.store(false);
        cleanup_condition_.notify_all();
        
        if (cleanup_thread_.joinable()) {
            cleanup_thread_.join();
        }
    }
}

void ResourceManager::cleanupThreadFunc() {
    std::unique_lock<std::mutex> lock(cleanup_mutex_);
    
    while (cleanup_running_.load()) {
        // 每30秒清理一次过期的锁和预留
        if (cleanup_condition_.wait_for(lock, std::chrono::seconds(30)) == std::cv_status::timeout) {
            cleanupExpiredLocks();
            
            // 检查死锁
            if (detectDeadlock()) {
                std::cerr << "Deadlock detected, consider manual intervention" << std::endl;
            }
        }
    }
}

// ResourceLockGuard实现

ResourceLockGuard::ResourceLockGuard(ResourceManager& manager,
                                   const std::string& resource_id,
                                   const std::string& holder_id,
                                   LockMode mode,
                                   int timeout_ms)
    : manager_(manager)
    , resource_id_(resource_id)
    , holder_id_(holder_id)
    , locked_(false) {
    
    locked_ = manager_.acquireLock(resource_id_, holder_id_, mode, timeout_ms);
}

ResourceLockGuard::~ResourceLockGuard() {
    unlock();
}

void ResourceLockGuard::unlock() {
    if (locked_) {
        manager_.releaseLock(resource_id_, holder_id_);
        locked_ = false;
    }
}

} // namespace core
} // namespace duorou