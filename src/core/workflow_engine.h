#pragma once

#include <string>
#include <memory>
#include <vector>
#include <queue>
#include <unordered_map>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <chrono>
#include "resource_manager.h"

namespace duorou {
namespace core {

/**
 * @brief Task status enumeration
 */
enum class TaskStatus {
    PENDING,        ///< Waiting for execution
    RUNNING,        ///< Currently executing
    COMPLETED,      ///< Completed
    FAILED,         ///< Execution failed
    CANCELLED       ///< Cancelled
};

/**
 * @brief Task priority enumeration
 */
enum class TaskPriority {
    LOW = 0,
    NORMAL = 1,
    HIGH = 2,
    URGENT = 3
};

/**
 * @brief Task result structure
 */
struct TaskResult {
    bool success;                   ///< Whether successful
    std::string message;            ///< Result message
    std::string output_data;        ///< Output data
    std::chrono::milliseconds duration;  ///< Execution duration
    
    TaskResult() : success(false), duration(0) {}
};

/**
 * @brief Base task class
 */
class BaseTask {
public:
    /**
     * @brief Constructor
     * @param id Task ID
     * @param name Task name
     * @param priority Task priority
     */
    BaseTask(const std::string& id, const std::string& name, TaskPriority priority = TaskPriority::NORMAL);
    
    /**
     * @brief Virtual destructor
     */
    virtual ~BaseTask() = default;
    
    /**
     * @brief Execute task
     * @return Task result
     */
    virtual TaskResult execute() = 0;
    
    /**
     * @brief Cancel task
     */
    virtual void cancel();
    
    /**
     * @brief Get required model for task
     * @return Model name, empty string means no specific model required
     */
    virtual std::string getRequiredModel() const { return ""; }
    
    /**
     * @brief Get task ID
     * @return Task ID
     */
    const std::string& getId() const { return id_; }
    
    /**
     * @brief Get task name
     * @return Task name
     */
    const std::string& getName() const { return name_; }
    
    /**
     * @brief Get task priority
     * @return Task priority
     */
    TaskPriority getPriority() const { return priority_; }
    
    /**
     * @brief Get task status
     * @return Task status
     */
    TaskStatus getStatus() const { return status_; }
    
    /**
     * @brief Set task status
     * @param status New status
     */
    void setStatus(TaskStatus status) { status_ = status; }
    
    /**
     * @brief Check if task is cancelled
     * @return Returns true if cancelled, false otherwise
     */
    bool isCancelled() const { return cancelled_.load(); }
    
    /**
     * @brief Get creation time
     * @return Creation time
     */
    std::chrono::system_clock::time_point getCreatedTime() const { return created_time_; }
    
protected:
    std::string id_;                                        ///< Task ID
    std::string name_;                                      ///< Task name
    TaskPriority priority_;                                 ///< Task priority
    TaskStatus status_;                                     ///< Task status
    std::atomic<bool> cancelled_;                           ///< Whether cancelled
    std::chrono::system_clock::time_point created_time_;   ///< Creation time
};

/**
 * @brief Task comparator (for priority queue)
 */
struct TaskComparator {
    bool operator()(const std::shared_ptr<BaseTask>& a, const std::shared_ptr<BaseTask>& b) const {
        // Higher priority tasks execute first
        if (a->getPriority() != b->getPriority()) {
            return a->getPriority() < b->getPriority();
        }
        // When priority is same, earlier created tasks execute first
        return a->getCreatedTime() > b->getCreatedTime();
    }
};

/**
 * @brief Workflow engine class
 * 
 * Responsible for task scheduling, execution and management, supports priority queue and concurrent execution
 */
class WorkflowEngine {
public:
    /**
     * @brief Constructor
     */
    WorkflowEngine();
    
    /**
     * @brief Destructor
     */
    ~WorkflowEngine();
    
    /**
     * @brief Initialize workflow engine
     * @param worker_count Number of worker threads, 0 means use CPU core count
     * @return Returns true on success, false on failure
     */
    bool initialize(size_t worker_count = 0);
    
    /**
     * @brief Start workflow engine
     * @return Returns true on success, false on failure
     */
    bool start();
    
    /**
     * @brief Stop workflow engine
     */
    void stop();
    
    /**
     * @brief Submit task
     * @param task Task pointer
     * @return Returns true on success, false on failure
     */
    bool submitTask(std::shared_ptr<BaseTask> task);
    
    /**
     * @brief Submit task that requires resources
     * @param task Task pointer
     * @param required_resources List of required resources
     * @param lock_mode Lock mode
     * @return Returns true on success, false on failure
     */
    bool submitTaskWithResources(std::shared_ptr<BaseTask> task, const std::vector<std::string>& required_resources, LockMode lock_mode = LockMode::EXCLUSIVE);
    
    /**
     * @brief Cancel task
     * @param task_id Task ID
     * @return Returns true on success, false on failure
     */
    bool cancelTask(const std::string& task_id);
    
    /**
     * @brief Wait for task completion
     * @param task_id Task ID
     * @param timeout_ms Timeout in milliseconds, 0 means infinite wait
     * @return Task result
     */
    TaskResult waitForTask(const std::string& task_id, int timeout_ms = 0);
    
    /**
     * @brief Get task status
     * @param task_id Task ID
     * @return Task status
     */
    TaskStatus getTaskStatus(const std::string& task_id) const;
    
    /**
     * @brief Get task result
     * @param task_id Task ID
     * @return Task result, returns empty result if task doesn't exist or is not completed
     */
    TaskResult getTaskResult(const std::string& task_id) const;
    
    /**
     * @brief Get number of tasks in waiting queue
     * @return Number of pending tasks
     */
    size_t getPendingTaskCount() const;
    
    /**
     * @brief Get number of running tasks
     * @return Number of running tasks
     */
    size_t getRunningTaskCount() const;
    
    /**
     * @brief Get number of completed tasks
     * @return Number of completed tasks
     */
    size_t getCompletedTaskCount() const;
    
    /**
     * @brief Clean up completed task records
     */
    void cleanupCompletedTasks();
    
    /**
     * @brief Set task completion callback function
     * @param callback Callback function
     */
    void setTaskCompletionCallback(std::function<void(const std::string&, const TaskResult&)> callback);
    
    /**
     * @brief Get resource manager
     * @return Resource manager reference
     */
    ResourceManager& getResourceManager() { return *resource_manager_; }
    
    /**
     * @brief Get resource manager (const version)
     * @return Resource manager const reference
     */
    const ResourceManager& getResourceManager() const { return *resource_manager_; }
    
    /**
     * @brief Enable/disable model switching optimization
     * @param enable Whether to enable
     */
    void optimizeModelSwitching(bool enable = true) { optimize_model_switching_ = enable; }
    
    /**
     * @brief Check if model switching optimization is enabled
     * @return Returns true if enabled, false otherwise
     */
    bool isModelSwitchingOptimized() const { return optimize_model_switching_; }
    
    /**
     * @brief Get number of worker threads
     * @return Number of worker threads
     */
    size_t getWorkerCount() const { return worker_count_; }
    
    /**
     * @brief Check if engine is running
     * @return Returns true if running, false otherwise
     */
    bool isRunning() const { return running_.load(); }
    
private:
    /**
     * @brief Worker thread function
     */
    void workerThread();
    
    /**
     * @brief Execute task
     * @param task Task pointer
     */
    void executeTask(std::shared_ptr<BaseTask> task);
    
    /**
     * @brief Generate unique task ID
     * @return Task ID
     */
    std::string generateTaskId();
    
private:
    // Task queue and management
    std::priority_queue<std::shared_ptr<BaseTask>, 
                       std::vector<std::shared_ptr<BaseTask>>, 
                       TaskComparator> task_queue_;                     ///< Task priority queue
    std::unordered_map<std::string, std::shared_ptr<BaseTask>> all_tasks_;  ///< All tasks mapping
    std::unordered_map<std::string, TaskResult> task_results_;          ///< Task results mapping
    
    // Thread management
    std::vector<std::thread> worker_threads_;                           ///< Worker thread pool
    size_t worker_count_;                                               ///< Number of worker threads
    std::atomic<bool> running_;                                         ///< Whether running
    std::atomic<bool> stop_requested_;                                  ///< Whether stop requested
    
    // Synchronization primitives
    mutable std::mutex queue_mutex_;                                    ///< Queue mutex
    mutable std::mutex results_mutex_;                                  ///< Results mutex
    std::condition_variable queue_condition_;                           ///< Queue condition variable
    
    // Statistics
    std::atomic<size_t> running_task_count_;                           ///< Number of running tasks
    std::atomic<size_t> completed_task_count_;                         ///< Number of completed tasks
    
    // Callback functions
    std::function<void(const std::string&, const TaskResult&)> completion_callback_;  ///< Task completion callback
    
    // Resource management
    std::unique_ptr<ResourceManager> resource_manager_;                 ///< Resource manager
    std::unordered_map<std::string, std::vector<std::string>> task_resources_; ///< Task resource mapping
    mutable std::mutex task_resources_mutex_;                          ///< Task resource mapping mutex
    
    /// Model switching optimization related
    std::atomic<bool> optimize_model_switching_;                        ///< Whether model switching optimization is enabled
    std::string current_loaded_model_;                                  ///< Currently loaded model
    
    bool initialized_;                                                  ///< Whether initialized
};

} // namespace core
} // namespace duorou