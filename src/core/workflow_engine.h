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

namespace duorou {
namespace core {

/**
 * @brief 任务状态枚举
 */
enum class TaskStatus {
    PENDING,        ///< 等待执行
    RUNNING,        ///< 正在执行
    COMPLETED,      ///< 已完成
    FAILED,         ///< 执行失败
    CANCELLED       ///< 已取消
};

/**
 * @brief 任务优先级枚举
 */
enum class TaskPriority {
    LOW = 0,
    NORMAL = 1,
    HIGH = 2,
    URGENT = 3
};

/**
 * @brief 任务结果结构
 */
struct TaskResult {
    bool success;                   ///< 是否成功
    std::string message;            ///< 结果消息
    std::string output_data;        ///< 输出数据
    std::chrono::milliseconds duration;  ///< 执行时长
    
    TaskResult() : success(false), duration(0) {}
};

/**
 * @brief 任务基类
 */
class BaseTask {
public:
    /**
     * @brief 构造函数
     * @param id 任务ID
     * @param name 任务名称
     * @param priority 任务优先级
     */
    BaseTask(const std::string& id, const std::string& name, TaskPriority priority = TaskPriority::NORMAL);
    
    /**
     * @brief 虚析构函数
     */
    virtual ~BaseTask() = default;
    
    /**
     * @brief 执行任务
     * @return 任务结果
     */
    virtual TaskResult execute() = 0;
    
    /**
     * @brief 取消任务
     */
    virtual void cancel();
    
    /**
     * @brief 获取任务ID
     * @return 任务ID
     */
    const std::string& getId() const { return id_; }
    
    /**
     * @brief 获取任务名称
     * @return 任务名称
     */
    const std::string& getName() const { return name_; }
    
    /**
     * @brief 获取任务优先级
     * @return 任务优先级
     */
    TaskPriority getPriority() const { return priority_; }
    
    /**
     * @brief 获取任务状态
     * @return 任务状态
     */
    TaskStatus getStatus() const { return status_; }
    
    /**
     * @brief 设置任务状态
     * @param status 新状态
     */
    void setStatus(TaskStatus status) { status_ = status; }
    
    /**
     * @brief 检查任务是否被取消
     * @return 被取消返回true，否则返回false
     */
    bool isCancelled() const { return cancelled_.load(); }
    
    /**
     * @brief 获取创建时间
     * @return 创建时间
     */
    std::chrono::system_clock::time_point getCreatedTime() const { return created_time_; }
    
protected:
    std::string id_;                                        ///< 任务ID
    std::string name_;                                      ///< 任务名称
    TaskPriority priority_;                                 ///< 任务优先级
    TaskStatus status_;                                     ///< 任务状态
    std::atomic<bool> cancelled_;                           ///< 是否被取消
    std::chrono::system_clock::time_point created_time_;   ///< 创建时间
};

/**
 * @brief 任务比较器（用于优先队列）
 */
struct TaskComparator {
    bool operator()(const std::shared_ptr<BaseTask>& a, const std::shared_ptr<BaseTask>& b) const {
        // 优先级高的任务优先执行
        if (a->getPriority() != b->getPriority()) {
            return a->getPriority() < b->getPriority();
        }
        // 优先级相同时，创建时间早的优先执行
        return a->getCreatedTime() > b->getCreatedTime();
    }
};

/**
 * @brief 工作流引擎类
 * 
 * 负责任务的调度、执行和管理，支持优先级队列和并发执行
 */
class WorkflowEngine {
public:
    /**
     * @brief 构造函数
     */
    WorkflowEngine();
    
    /**
     * @brief 析构函数
     */
    ~WorkflowEngine();
    
    /**
     * @brief 初始化工作流引擎
     * @param worker_count 工作线程数量，0表示使用CPU核心数
     * @return 成功返回true，失败返回false
     */
    bool initialize(size_t worker_count = 0);
    
    /**
     * @brief 启动工作流引擎
     * @return 成功返回true，失败返回false
     */
    bool start();
    
    /**
     * @brief 停止工作流引擎
     */
    void stop();
    
    /**
     * @brief 提交任务
     * @param task 任务指针
     * @return 成功返回true，失败返回false
     */
    bool submitTask(std::shared_ptr<BaseTask> task);
    
    /**
     * @brief 取消任务
     * @param task_id 任务ID
     * @return 成功返回true，失败返回false
     */
    bool cancelTask(const std::string& task_id);
    
    /**
     * @brief 等待任务完成
     * @param task_id 任务ID
     * @param timeout_ms 超时时间（毫秒），0表示无限等待
     * @return 任务结果
     */
    TaskResult waitForTask(const std::string& task_id, int timeout_ms = 0);
    
    /**
     * @brief 获取任务状态
     * @param task_id 任务ID
     * @return 任务状态
     */
    TaskStatus getTaskStatus(const std::string& task_id) const;
    
    /**
     * @brief 获取任务结果
     * @param task_id 任务ID
     * @return 任务结果，如果任务不存在或未完成返回空结果
     */
    TaskResult getTaskResult(const std::string& task_id) const;
    
    /**
     * @brief 获取等待队列中的任务数量
     * @return 等待任务数量
     */
    size_t getPendingTaskCount() const;
    
    /**
     * @brief 获取正在执行的任务数量
     * @return 正在执行的任务数量
     */
    size_t getRunningTaskCount() const;
    
    /**
     * @brief 获取已完成的任务数量
     * @return 已完成的任务数量
     */
    size_t getCompletedTaskCount() const;
    
    /**
     * @brief 清理已完成的任务记录
     */
    void cleanupCompletedTasks();
    
    /**
     * @brief 设置任务完成回调函数
     * @param callback 回调函数
     */
    void setTaskCompletionCallback(std::function<void(const std::string&, const TaskResult&)> callback);
    
    /**
     * @brief 获取工作线程数量
     * @return 工作线程数量
     */
    size_t getWorkerCount() const { return worker_count_; }
    
    /**
     * @brief 检查引擎是否正在运行
     * @return 正在运行返回true，否则返回false
     */
    bool isRunning() const { return running_.load(); }
    
private:
    /**
     * @brief 工作线程函数
     */
    void workerThread();
    
    /**
     * @brief 执行任务
     * @param task 任务指针
     */
    void executeTask(std::shared_ptr<BaseTask> task);
    
    /**
     * @brief 生成唯一的任务ID
     * @return 任务ID
     */
    std::string generateTaskId();
    
private:
    // 任务队列和管理
    std::priority_queue<std::shared_ptr<BaseTask>, 
                       std::vector<std::shared_ptr<BaseTask>>, 
                       TaskComparator> task_queue_;                     ///< 任务优先队列
    std::unordered_map<std::string, std::shared_ptr<BaseTask>> all_tasks_;  ///< 所有任务映射
    std::unordered_map<std::string, TaskResult> task_results_;          ///< 任务结果映射
    
    // 线程管理
    std::vector<std::thread> worker_threads_;                           ///< 工作线程池
    size_t worker_count_;                                               ///< 工作线程数量
    std::atomic<bool> running_;                                         ///< 是否正在运行
    std::atomic<bool> stop_requested_;                                  ///< 是否请求停止
    
    // 同步原语
    mutable std::mutex queue_mutex_;                                    ///< 队列互斥锁
    mutable std::mutex results_mutex_;                                  ///< 结果互斥锁
    std::condition_variable queue_condition_;                           ///< 队列条件变量
    
    // 统计信息
    std::atomic<size_t> running_task_count_;                           ///< 正在执行的任务数量
    std::atomic<size_t> completed_task_count_;                         ///< 已完成的任务数量
    
    // 回调函数
    std::function<void(const std::string&, const TaskResult&)> completion_callback_;  ///< 任务完成回调
    
    bool initialized_;                                                  ///< 是否已初始化
};

} // namespace core
} // namespace duorou