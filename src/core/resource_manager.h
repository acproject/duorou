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
    MODEL,          ///< 模型资源
    GPU_MEMORY,     ///< GPU内存资源
    CPU_MEMORY,     ///< CPU内存资源
    COMPUTE_UNIT,   ///< 计算单元资源
    STORAGE,        ///< 存储资源
    NETWORK         ///< 网络资源
};

/**
 * @brief 资源锁定模式
 */
enum class LockMode {
    SHARED,         ///< 共享锁（读锁）
    EXCLUSIVE       ///< 独占锁（写锁）
};

/**
 * @brief 资源信息
 */
struct ResourceInfo {
    std::string id;                 ///< 资源ID
    ResourceType type;              ///< 资源类型
    std::string name;               ///< 资源名称
    size_t capacity;                ///< 资源容量
    size_t used;                    ///< 已使用量
    bool available;                 ///< 是否可用
    std::chrono::system_clock::time_point last_accessed;  ///< 最后访问时间
    std::unordered_set<std::string> holders;  ///< 持有者列表
    
    ResourceInfo() : capacity(0), used(0), available(true) {}
};

/**
 * @brief 资源锁定信息
 */
struct ResourceLock {
    std::string resource_id;        ///< 资源ID
    std::string holder_id;          ///< 持有者ID
    LockMode mode;                  ///< 锁定模式
    std::chrono::system_clock::time_point acquired_time;  ///< 获取时间
    std::chrono::milliseconds timeout;  ///< 超时时间
    
    ResourceLock() : mode(LockMode::SHARED), timeout(0) {}
};

/**
 * @brief 资源预留信息
 */
struct ResourceReservation {
    std::string resource_id;        ///< 资源ID
    std::string requester_id;       ///< 请求者ID
    size_t amount;                  ///< 预留数量
    std::chrono::system_clock::time_point reserved_time;  ///< 预留时间
    std::chrono::milliseconds duration;  ///< 预留时长
    
    ResourceReservation() : amount(0), duration(0) {}
};

/**
 * @brief 资源管理器
 */
class ResourceManager {
public:
    /**
     * @brief 构造函数
     */
    ResourceManager();
    
    /**
     * @brief 析构函数
     */
    ~ResourceManager();
    
    /**
     * @brief 注册资源
     * @param resource_info 资源信息
     * @return 是否成功
     */
    bool registerResource(const ResourceInfo& resource_info);
    
    /**
     * @brief 注销资源
     * @param resource_id 资源ID
     * @return 是否成功
     */
    bool unregisterResource(const std::string& resource_id);
    
    /**
     * @brief 获取资源锁
     * @param resource_id 资源ID
     * @param holder_id 持有者ID
     * @param mode 锁定模式
     * @param timeout_ms 超时时间（毫秒）
     * @return 是否成功获取锁
     */
    bool acquireLock(const std::string& resource_id, 
                     const std::string& holder_id,
                     LockMode mode,
                     int timeout_ms = 5000);
    
    /**
     * @brief 释放资源锁
     * @param resource_id 资源ID
     * @param holder_id 持有者ID
     * @return 是否成功释放锁
     */
    bool releaseLock(const std::string& resource_id, const std::string& holder_id);
    
    /**
     * @brief 预留资源
     * @param resource_id 资源ID
     * @param requester_id 请求者ID
     * @param amount 预留数量
     * @param duration_ms 预留时长（毫秒）
     * @return 是否成功预留
     */
    bool reserveResource(const std::string& resource_id,
                        const std::string& requester_id,
                        size_t amount,
                        int duration_ms = 30000);
    
    /**
     * @brief 释放资源预留
     * @param resource_id 资源ID
     * @param requester_id 请求者ID
     * @return 是否成功释放
     */
    bool releaseReservation(const std::string& resource_id, const std::string& requester_id);
    
    /**
     * @brief 检查资源是否可用
     * @param resource_id 资源ID
     * @param mode 锁定模式
     * @return 是否可用
     */
    bool isResourceAvailable(const std::string& resource_id, LockMode mode = LockMode::SHARED) const;
    
    /**
     * @brief 获取资源信息
     * @param resource_id 资源ID
     * @return 资源信息
     */
    ResourceInfo getResourceInfo(const std::string& resource_id) const;
    
    /**
     * @brief 获取资源使用率
     * @param resource_id 资源ID
     * @return 使用率（0.0-1.0）
     */
    double getResourceUtilization(const std::string& resource_id) const;
    
    /**
     * @brief 获取所有资源列表
     * @param type 资源类型过滤（可选）
     * @return 资源ID列表
     */
    std::vector<std::string> getResourceList(ResourceType type = ResourceType::MODEL) const;
    
    /**
     * @brief 清理过期的锁和预留
     */
    void cleanupExpiredLocks();
    
    /**
     * @brief 设置资源状态变化回调
     * @param callback 回调函数
     */
    void setResourceStatusCallback(std::function<void(const std::string&, bool)> callback);
    
    /**
     * @brief 获取资源统计信息
     * @return 统计信息映射
     */
    std::unordered_map<std::string, size_t> getResourceStatistics() const;
    
    /**
     * @brief 强制释放持有者的所有锁
     * @param holder_id 持有者ID
     * @return 释放的锁数量
     */
    size_t forceReleaseHolderLocks(const std::string& holder_id);
    
    /**
     * @brief 检查死锁
     * @return 是否存在死锁
     */
    bool detectDeadlock() const;
    
    /**
     * @brief 获取等待队列长度
     * @param resource_id 资源ID
     * @return 等待队列长度
     */
    size_t getWaitingQueueLength(const std::string& resource_id) const;
    
private:
    /**
     * @brief 检查锁兼容性
     * @param resource_id 资源ID
     * @param mode 请求的锁模式
     * @return 是否兼容
     */
    bool isLockCompatible(const std::string& resource_id, LockMode mode) const;
    
    /**
     * @brief 清理过期预留
     */
    void cleanupExpiredReservations();
    
    /**
     * @brief 启动清理线程
     */
    void startCleanupThread();
    
    /**
     * @brief 停止清理线程
     */
    void stopCleanupThread();
    
    /**
     * @brief 清理线程函数
     */
    void cleanupThreadFunc();
    
private:
    mutable std::mutex resources_mutex_;                    ///< 资源互斥锁
    mutable std::mutex locks_mutex_;                        ///< 锁互斥锁
    mutable std::mutex reservations_mutex_;                 ///< 预留互斥锁
    
    std::unordered_map<std::string, ResourceInfo> resources_;           ///< 资源映射
    std::unordered_map<std::string, std::vector<ResourceLock>> locks_;   ///< 锁映射
    std::unordered_map<std::string, std::vector<ResourceReservation>> reservations_;  ///< 预留映射
    
    std::unordered_map<std::string, std::condition_variable> wait_conditions_;  ///< 等待条件变量
    std::unordered_map<std::string, size_t> waiting_counts_;            ///< 等待计数
    
    std::function<void(const std::string&, bool)> status_callback_;     ///< 状态回调
    
    std::atomic<bool> cleanup_running_;                     ///< 清理线程运行标志
    std::thread cleanup_thread_;                            ///< 清理线程
    std::condition_variable cleanup_condition_;             ///< 清理条件变量
    std::mutex cleanup_mutex_;                              ///< 清理互斥锁
};

/**
 * @brief RAII资源锁
 */
class ResourceLockGuard {
public:
    /**
     * @brief 构造函数
     * @param manager 资源管理器
     * @param resource_id 资源ID
     * @param holder_id 持有者ID
     * @param mode 锁定模式
     * @param timeout_ms 超时时间
     */
    ResourceLockGuard(ResourceManager& manager,
                     const std::string& resource_id,
                     const std::string& holder_id,
                     LockMode mode = LockMode::SHARED,
                     int timeout_ms = 5000);
    
    /**
     * @brief 析构函数
     */
    ~ResourceLockGuard();
    
    /**
     * @brief 检查是否成功获取锁
     * @return 是否成功
     */
    bool isLocked() const { return locked_; }
    
    /**
     * @brief 手动释放锁
     */
    void unlock();
    
    // 禁止拷贝和赋值
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