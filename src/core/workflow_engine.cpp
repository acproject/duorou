#include "workflow_engine.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <random>

namespace duorou {
namespace core {

// BaseTask implementation
BaseTask::BaseTask(const std::string& id, const std::string& name, TaskPriority priority)
    : id_(id)
    , name_(name)
    , priority_(priority)
    , status_(TaskStatus::PENDING)
    , cancelled_(false)
    , created_time_(std::chrono::system_clock::now()) {
}

void BaseTask::cancel() {
    cancelled_.store(true);
    status_ = TaskStatus::CANCELLED;
}

// WorkflowEngine implementation
WorkflowEngine::WorkflowEngine()
    : worker_count_(0)
    , running_(false)
    , stop_requested_(false)
    , running_task_count_(0)
    , completed_task_count_(0)
    , resource_manager_(std::make_unique<ResourceManager>())
    , optimize_model_switching_(false)
    , initialized_(false) {
}

WorkflowEngine::~WorkflowEngine() {
    stop();
}

bool WorkflowEngine::initialize(size_t worker_count) {
    if (initialized_) {
        return true;
    }
    
    // If worker count not specified, use CPU core count
    if (worker_count == 0) {
        worker_count_ = std::thread::hardware_concurrency();
        if (worker_count_ == 0) {
            worker_count_ = 4; // Default 4 threads
        }
    } else {
        worker_count_ = worker_count;
    }
    
    initialized_ = true;
    std::cout << "WorkflowEngine initialized with " << worker_count_ << " worker threads" << std::endl;
    return true;
}

bool WorkflowEngine::start() {
    if (!initialized_) {
        std::cerr << "WorkflowEngine not initialized" << std::endl;
        return false;
    }
    
    if (running_.load()) {
        std::cout << "WorkflowEngine already running" << std::endl;
        return true;
    }
    
    // Register default resources
    ResourceInfo llama_resource;
    llama_resource.id = "llama_model";
    llama_resource.type = ResourceType::MODEL;
    llama_resource.name = "LLaMA Model";
    llama_resource.capacity = 1;
    llama_resource.used = 0;
    llama_resource.available = true;
    resource_manager_->registerResource(llama_resource);
    
    ResourceInfo sd_resource;
    sd_resource.id = "stable_diffusion_model";
    sd_resource.type = ResourceType::MODEL;
    sd_resource.name = "Stable Diffusion Model";
    sd_resource.capacity = 1;
    sd_resource.used = 0;
    sd_resource.available = true;
    resource_manager_->registerResource(sd_resource);
    
    ResourceInfo gpu_resource;
    gpu_resource.id = "gpu_memory";
    gpu_resource.type = ResourceType::GPU_MEMORY;
    gpu_resource.name = "GPU Memory";
    gpu_resource.capacity = 1;
    gpu_resource.used = 0;
    gpu_resource.available = true;
    resource_manager_->registerResource(gpu_resource);
    
    ResourceInfo cpu_resource;
    cpu_resource.id = "cpu_cores";
    cpu_resource.type = ResourceType::COMPUTE_UNIT;
    cpu_resource.name = "CPU Cores";
    cpu_resource.capacity = worker_count_;
    cpu_resource.used = 0;
    cpu_resource.available = true;
    resource_manager_->registerResource(cpu_resource);
    
    running_.store(true);
    stop_requested_.store(false);
    
    // Create worker threads
    worker_threads_.reserve(worker_count_);
    for (size_t i = 0; i < worker_count_; ++i) {
        worker_threads_.emplace_back(&WorkflowEngine::workerThread, this);
    }
    
    std::cout << "WorkflowEngine started with " << worker_count_ << " worker threads" << std::endl;
    return true;
}

void WorkflowEngine::stop() {
    if (!running_.load()) {
        return;
    }
    
    std::cout << "Stopping WorkflowEngine..." << std::endl;
    
    // Request stop
    stop_requested_.store(true);
    running_.store(false);
    
    // Wake up all waiting worker threads
    queue_condition_.notify_all();
    
    // Wait for all worker threads to finish
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    worker_threads_.clear();
    
    // Cancel all pending tasks
    std::lock_guard<std::mutex> lock(queue_mutex_);
    while (!task_queue_.empty()) {
        auto task = task_queue_.top();
        task_queue_.pop();
        task->cancel();
    }
    
    std::cout << "WorkflowEngine stopped" << std::endl;
}

bool WorkflowEngine::submitTask(std::shared_ptr<BaseTask> task) {
    if (!task) {
        std::cerr << "Cannot submit null task" << std::endl;
        return false;
    }
    
    if (!running_.load()) {
        std::cerr << "WorkflowEngine not running" << std::endl;
        return false;
    }
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        
        // Check if task ID already exists
        if (all_tasks_.find(task->getId()) != all_tasks_.end()) {
            std::cerr << "Task with ID already exists: " << task->getId() << std::endl;
            return false;
        }
        
        // Add to queue and mapping table
        task_queue_.push(task);
        all_tasks_[task->getId()] = task;
    }
    
    // Wake up one worker thread
    queue_condition_.notify_one();
    
    std::cout << "Task submitted: " << task->getId() << " (" << task->getName() << ")" << std::endl;
    return true;
}

bool WorkflowEngine::submitTaskWithResources(std::shared_ptr<BaseTask> task, const std::vector<std::string>& required_resources, LockMode lock_mode) {
    if (!task) {
        std::cerr << "Cannot submit null task" << std::endl;
        return false;
    }
    
    if (!running_.load()) {
        std::cerr << "WorkflowEngine not running" << std::endl;
        return false;
    }
    
    // Try to acquire required resources
    std::vector<std::string> acquired_resources;
    for (const auto& resource_id : required_resources) {
        bool success = resource_manager_->acquireLock(resource_id, task->getId(), lock_mode);
        if (!success) {
            std::cerr << "Failed to acquire resource lock: " << resource_id << " for task: " << task->getId() << std::endl;
            // Release acquired resources
            for (const auto& acquired : acquired_resources) {
                resource_manager_->releaseLock(acquired, task->getId());
            }
            return false;
        }
        acquired_resources.push_back(resource_id);
    }
    
    // Record task resource locks
    {
        std::lock_guard<std::mutex> lock(task_resources_mutex_);
        task_resources_[task->getId()] = acquired_resources;
    }
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        
        // 检查任务ID是否已存在
        if (all_tasks_.find(task->getId()) != all_tasks_.end()) {
            std::cerr << "Task with ID already exists: " << task->getId() << std::endl;
            return false;
        }
        
        // 添加到队列和映射表
        task_queue_.push(task);
        all_tasks_[task->getId()] = task;
    }
    
    // 唤醒一个工作线程
    queue_condition_.notify_one();
    
    std::cout << "Task with resources submitted: " << task->getId() << " (" << task->getName() << ")" << std::endl;
    return true;
}

bool WorkflowEngine::cancelTask(const std::string& task_id) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    auto it = all_tasks_.find(task_id);
    if (it == all_tasks_.end()) {
        std::cerr << "Task not found: " << task_id << std::endl;
        return false;
    }
    
    auto task = it->second;
    if (task->getStatus() == TaskStatus::PENDING) {
        task->cancel();
        std::cout << "Task cancelled: " << task_id << std::endl;
        return true;
    } else if (task->getStatus() == TaskStatus::RUNNING) {
        // For running tasks, set cancel flag
        task->cancel();
        std::cout << "Cancel signal sent to running task: " << task_id << std::endl;
        return true;
    } else {
        std::cout << "Task cannot be cancelled (status: " << static_cast<int>(task->getStatus()) << "): " << task_id << std::endl;
        return false;
    }
}

TaskResult WorkflowEngine::waitForTask(const std::string& task_id, int timeout_ms) {
    auto start_time = std::chrono::steady_clock::now();
    
    while (true) {
        // Check task result
        {
            std::lock_guard<std::mutex> lock(results_mutex_);
            auto it = task_results_.find(task_id);
            if (it != task_results_.end()) {
                return it->second;
            }
        }
        
        // Check timeout
        if (timeout_ms > 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_time).count();
            if (elapsed >= timeout_ms) {
                TaskResult timeout_result;
                timeout_result.success = false;
                timeout_result.message = "Task wait timeout";
                return timeout_result;
            }
        }
        
        // Brief sleep
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

TaskStatus WorkflowEngine::getTaskStatus(const std::string& task_id) const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    auto it = all_tasks_.find(task_id);
    if (it != all_tasks_.end()) {
        return it->second->getStatus();
    }
    
    return TaskStatus::PENDING; // Default status
}

TaskResult WorkflowEngine::getTaskResult(const std::string& task_id) const {
    std::lock_guard<std::mutex> lock(results_mutex_);
    
    auto it = task_results_.find(task_id);
    if (it != task_results_.end()) {
        return it->second;
    }
    
    return TaskResult(); // Return empty result
}

size_t WorkflowEngine::getPendingTaskCount() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return task_queue_.size();
}

size_t WorkflowEngine::getRunningTaskCount() const {
    return running_task_count_.load();
}

size_t WorkflowEngine::getCompletedTaskCount() const {
    return completed_task_count_.load();
}

void WorkflowEngine::cleanupCompletedTasks() {
    std::lock_guard<std::mutex> queue_lock(queue_mutex_);
    std::lock_guard<std::mutex> results_lock(results_mutex_);
    
    // Clean up completed tasks
    auto it = all_tasks_.begin();
    while (it != all_tasks_.end()) {
        auto status = it->second->getStatus();
        if (status == TaskStatus::COMPLETED || 
            status == TaskStatus::FAILED || 
            status == TaskStatus::CANCELLED) {
            
            // Keep results, delete task references
            it = all_tasks_.erase(it);
        } else {
            ++it;
        }
    }
    
    std::cout << "Completed tasks cleaned up" << std::endl;
}

void WorkflowEngine::setTaskCompletionCallback(std::function<void(const std::string&, const TaskResult&)> callback) {
    completion_callback_ = std::move(callback);
}

void WorkflowEngine::workerThread() {
    while (running_.load()) {
        std::shared_ptr<BaseTask> task;
        
        // Get task
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            
            // Wait for task or stop signal
            queue_condition_.wait(lock, [this] {
                return !task_queue_.empty() || stop_requested_.load();
            });
            
            // Check if need to stop
            if (stop_requested_.load()) {
                break;
            }
            
            // Get task
            if (!task_queue_.empty()) {
                task = task_queue_.top();
                task_queue_.pop();
            }
        }
        
        // Execute task
        if (task) {
            executeTask(task);
        }
    }
}

void WorkflowEngine::executeTask(std::shared_ptr<BaseTask> task) {
    if (!task || task->isCancelled()) {
        return;
    }
    
    // Update task status
    task->setStatus(TaskStatus::RUNNING);
    running_task_count_.fetch_add(1);
    
    std::cout << "Executing task: " << task->getId() << " (" << task->getName() << ")" << std::endl;
    
    // Model switching optimization logic
    if (optimize_model_switching_) {
        // Check if model switching is needed
        std::string required_model = task->getRequiredModel();
        if (!required_model.empty() && required_model != current_loaded_model_) {
            std::cout << "Switching model from " << current_loaded_model_ << " to " << required_model << std::endl;
            // Actual model switching logic can be added here
            current_loaded_model_ = required_model;
        }
    }
    
    // Record start time
    auto start_time = std::chrono::steady_clock::now();
    
    // Execute task
    TaskResult result;
    try {
        result = task->execute();
    } catch (const std::exception& e) {
        result.success = false;
        result.message = std::string("Exception: ") + e.what();
        std::cerr << "Task execution failed with exception: " << e.what() << std::endl;
    } catch (...) {
        result.success = false;
        result.message = "Unknown exception";
        std::cerr << "Task execution failed with unknown exception" << std::endl;
    }
    
    // Calculate execution time
    auto end_time = std::chrono::steady_clock::now();
    result.duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Update task status
    if (task->isCancelled()) {
        task->setStatus(TaskStatus::CANCELLED);
        result.success = false;
        result.message = "Task was cancelled";
    } else if (result.success) {
        task->setStatus(TaskStatus::COMPLETED);
    } else {
        task->setStatus(TaskStatus::FAILED);
    }
    
    // Save result
    {
        std::lock_guard<std::mutex> lock(results_mutex_);
        task_results_[task->getId()] = result;
    }
    
    // Update counters
    running_task_count_.fetch_sub(1);
    completed_task_count_.fetch_add(1);
    
    // Call completion callback
    if (completion_callback_) {
        try {
            completion_callback_(task->getId(), result);
        } catch (const std::exception& e) {
            std::cerr << "Exception in completion callback: " << e.what() << std::endl;
        }
    }
    
    // Release task locked resources
    {
        std::lock_guard<std::mutex> lock(task_resources_mutex_);
        auto it = task_resources_.find(task->getId());
        if (it != task_resources_.end()) {
            for (const auto& resource_id : it->second) {
                resource_manager_->releaseLock(resource_id, task->getId());
            }
            task_resources_.erase(it);
        }
    }
    
    std::cout << "Task completed: " << task->getId() 
              << " (success: " << (result.success ? "true" : "false")
              << ", duration: " << result.duration.count() << "ms)" << std::endl;
}

std::string WorkflowEngine::generateTaskId() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);
    
    std::stringstream ss;
    ss << "task_";
    
    // Generate simple random ID
    for (int i = 0; i < 8; ++i) {
        ss << std::hex << dis(gen);
    }
    
    return ss.str();
}

} // namespace core
} // namespace duorou