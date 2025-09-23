#include "image_pipeline.h"
#include <iostream>
#include <sstream>
#include <random>
#include <thread>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace duorou {
namespace invokeai {

// ImagePipeline的私有实现
class ImagePipeline::Impl {
public:
    Impl() : initialized_(false), debug_mode_(false), next_task_id_(1) {}
    
    ~Impl() {
        shutdown();
    }
    
    bool initialize(const PipelineConfig& config) {
        if (initialized_) {
            return true;
        }
        
        config_ = config;
        
        // Initialize InvokeAI engine
        if (!engine_.initialize()) {
            std::cerr << "Failed to initialize InvokeAI engine" << std::endl;
            return false;
        }
        
        // Initialize model manager
        if (!model_manager_.initialize("/tmp/models")) {
            std::cerr << "Failed to initialize model manager" << std::endl;
            return false;
        }
        
        // Set engine configuration
        engine_.set_device(config_.device);
        engine_.set_precision(config_.precision);
        if (config_.num_threads > 0) {
            engine_.set_threads(config_.num_threads);
        }
        
        // Start worker threads
        stop_workers_ = false;
        for (int i = 0; i < std::max(1, config_.num_threads); ++i) {
            workers_.emplace_back(&Impl::worker_thread, this);
        }
        
        initialized_ = true;
        std::cout << "ImagePipeline initialized successfully" << std::endl;
        return true;
    }
    
    void shutdown() {
        if (!initialized_) {
            return;
        }
        
        // 停止所有任务
        cancel_all_tasks();
        
        // Stop worker threads
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            stop_workers_ = true;
        }
        queue_cv_.notify_all();
        
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers_.clear();
        
        // Shutdown engine and model manager
        engine_.shutdown();
        model_manager_.shutdown();
        
        initialized_ = false;
        std::cout << "ImagePipeline shutdown completed" << std::endl;
    }
    
    bool is_initialized() const {
        return initialized_;
    }
    
    void set_config(const PipelineConfig& config) {
        config_ = config;
        if (initialized_) {
            engine_.set_device(config_.device);
            engine_.set_precision(config_.precision);
            if (config_.num_threads > 0) {
                engine_.set_threads(config_.num_threads);
            }
        }
    }
    
    PipelineConfig get_config() const {
        return config_;
    }
    
    std::string generate_image_async(const ImageGenerationParams& params,
                                    PipelineProgressCallback progress_cb,
                                    TaskCompletedCallback completed_cb) {
        if (!initialized_) {
            return "";
        }
        
        // Create new task
        GenerationTask task;
        task.task_id = generate_task_id();
        task.pipeline_type = config_.type;
        task.params = params;
        task.status = TaskStatus::PENDING;
        task.created_time = std::chrono::system_clock::now();
        
        // Add to task queue
        {
            std::lock_guard<std::mutex> lock(tasks_mutex_);
            tasks_[task.task_id] = task;
        }
        
        // Add to work queue
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            task_queue_.push({task.task_id, progress_cb, completed_cb});
        }
        queue_cv_.notify_one();
        
        return task.task_id;
    }
    
    ImageGenerationResult generate_image_sync(const ImageGenerationParams& params,
                                             PipelineProgressCallback progress_cb) {
        if (!initialized_) {
            return ImageGenerationResult{};
        }
        
        // Generate image directly using engine
        return engine_.generate_image(params);
    }
    
    bool cancel_task(const std::string& task_id) {
        std::lock_guard<std::mutex> lock(tasks_mutex_);
        auto it = tasks_.find(task_id);
        if (it != tasks_.end() && it->second.status == TaskStatus::RUNNING) {
            it->second.status = TaskStatus::CANCELLED;
            engine_.cancel_generation();
            return true;
        }
        return false;
    }
    
    bool cancel_all_tasks() {
        std::lock_guard<std::mutex> lock(tasks_mutex_);
        bool cancelled_any = false;
        for (auto& [task_id, task] : tasks_) {
            if (task.status == TaskStatus::PENDING || task.status == TaskStatus::RUNNING) {
                task.status = TaskStatus::CANCELLED;
                cancelled_any = true;
            }
        }
        if (cancelled_any) {
            engine_.cancel_generation();
        }
        return cancelled_any;
    }
    
    GenerationTask get_task_info(const std::string& task_id) const {
        std::lock_guard<std::mutex> lock(tasks_mutex_);
        auto it = tasks_.find(task_id);
        if (it != tasks_.end()) {
            return it->second;
        }
        return GenerationTask{};
    }
    
    std::vector<GenerationTask> get_all_tasks() const {
        std::lock_guard<std::mutex> lock(tasks_mutex_);
        std::vector<GenerationTask> result;
        for (const auto& [task_id, task] : tasks_) {
            result.push_back(task);
        }
        return result;
    }
    
    std::vector<GenerationTask> get_tasks_by_status(TaskStatus status) const {
        std::lock_guard<std::mutex> lock(tasks_mutex_);
        std::vector<GenerationTask> result;
        for (const auto& [task_id, task] : tasks_) {
            if (task.status == status) {
                result.push_back(task);
            }
        }
        return result;
    }
    
    void clear_completed_tasks() {
        std::lock_guard<std::mutex> lock(tasks_mutex_);
        auto it = tasks_.begin();
        while (it != tasks_.end()) {
            if (it->second.status == TaskStatus::COMPLETED || 
                it->second.status == TaskStatus::FAILED ||
                it->second.status == TaskStatus::CANCELLED) {
                it = tasks_.erase(it);
            } else {
                ++it;
            }
        }
    }
    
    void clear_all_tasks() {
        std::lock_guard<std::mutex> lock(tasks_mutex_);
        tasks_.clear();
    }
    
    bool is_busy() const {
        std::lock_guard<std::mutex> lock(tasks_mutex_);
        for (const auto& [task_id, task] : tasks_) {
            if (task.status == TaskStatus::RUNNING) {
                return true;
            }
        }
        return false;
    }
    
    int get_active_task_count() const {
        std::lock_guard<std::mutex> lock(tasks_mutex_);
        int count = 0;
        for (const auto& [task_id, task] : tasks_) {
            if (task.status == TaskStatus::RUNNING) {
                count++;
            }
        }
        return count;
    }
    
    int get_pending_task_count() const {
        std::lock_guard<std::mutex> lock(tasks_mutex_);
        int count = 0;
        for (const auto& [task_id, task] : tasks_) {
            if (task.status == TaskStatus::PENDING) {
                count++;
            }
        }
        return count;
    }
    
    std::string get_current_status() const {
        if (!initialized_) {
            return "Not initialized";
        }
        
        int active = get_active_task_count();
        int pending = get_pending_task_count();
        
        if (active > 0) {
            return "Generating (" + std::to_string(active) + " active, " + 
                   std::to_string(pending) + " pending)";
        } else if (pending > 0) {
            return "Idle (" + std::to_string(pending) + " pending)";
        } else {
            return "Idle";
        }
    }
    
    bool load_pipeline_models() {
        if (!initialized_) {
            return false;
        }
        
        // Load main model
        if (!config_.model_name.empty()) {
            if (!model_manager_.load_model(config_.model_name)) {
                std::cerr << "Failed to load main model: " << config_.model_name << std::endl;
                return false;
            }
        }
        
        // Load VAE model
        if (!config_.vae_model.empty()) {
            if (!model_manager_.load_model(config_.vae_model)) {
                std::cerr << "Failed to load VAE model: " << config_.vae_model << std::endl;
            }
        }
        
        // Load ControlNet model
        if (!config_.controlnet_model.empty()) {
            if (!model_manager_.load_model(config_.controlnet_model)) {
                std::cerr << "Failed to load ControlNet model: " << config_.controlnet_model << std::endl;
            }
        }
        
        return true;
    }
    
    bool unload_pipeline_models() {
        if (!initialized_) {
            return false;
        }
        
        // Unload all loaded models
        auto loaded_models = model_manager_.get_loaded_models();
        for (const auto& model_name : loaded_models) {
            model_manager_.unload_model(model_name);
        }
        
        return true;
    }
    
    void set_event_callback(PipelineEventCallback callback) {
        event_callback_ = callback;
    }
    
    void enable_debug_mode(bool enabled) {
        debug_mode_ = enabled;
    }
    
private:
    struct QueuedTask {
        std::string task_id;
        PipelineProgressCallback progress_cb;
        TaskCompletedCallback completed_cb;
    };
    
    void worker_thread() {
        while (true) {
            QueuedTask queued_task;
            
            // Get task
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [this] { return !task_queue_.empty() || stop_workers_; });
                
                if (stop_workers_) {
                    break;
                }
                
                queued_task = task_queue_.front();
                task_queue_.pop();
            }
            
            // Execute task
            execute_task(queued_task);
        }
    }
    
    void execute_task(const QueuedTask& queued_task) {
        GenerationTask* task = nullptr;
        
        // Get task information
        {
            std::lock_guard<std::mutex> lock(tasks_mutex_);
            auto it = tasks_.find(queued_task.task_id);
            if (it == tasks_.end() || it->second.status != TaskStatus::PENDING) {
                return;
            }
            task = &it->second;
            task->status = TaskStatus::RUNNING;
            task->started_time = std::chrono::system_clock::now();
        }
        
        try {
            // Progress callback wrapper
            auto progress_wrapper = [&](float progress, const std::string& step) {
                {
                    std::lock_guard<std::mutex> lock(tasks_mutex_);
                    task->progress = progress;
                    task->current_step = step;
                }
                
                if (queued_task.progress_cb) {
                    queued_task.progress_cb(task->task_id, progress, step);
                }
            };
            
            // Execute image generation
            progress_wrapper(0.0f, "Starting generation...");
            
            // Simulate generation process
            for (int i = 0; i <= 100; i += 10) {
                if (task->status == TaskStatus::CANCELLED) {
                    break;
                }
                
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                progress_wrapper(i / 100.0f, "Generating step " + std::to_string(i) + "/100");
            }
            
            if (task->status != TaskStatus::CANCELLED) {
                // Generate image using engine
                task->result = engine_.generate_image(task->params);
                
                {
                    std::lock_guard<std::mutex> lock(tasks_mutex_);
                    task->status = TaskStatus::COMPLETED;
                    task->completed_time = std::chrono::system_clock::now();
                    task->progress = 1.0f;
                    task->current_step = "Completed";
                }
                
                if (queued_task.completed_cb) {
                    queued_task.completed_cb(*task);
                }
            }
            
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(tasks_mutex_);
            task->status = TaskStatus::FAILED;
            task->error_message = e.what();
            task->completed_time = std::chrono::system_clock::now();
            
            if (queued_task.completed_cb) {
                queued_task.completed_cb(*task);
            }
        }
    }
    
    std::string generate_task_id() {
        return "task_" + std::to_string(next_task_id_++);
    }
    
    // Member variables
    bool initialized_;
    bool debug_mode_;
    PipelineConfig config_;
    InvokeAIEngine engine_;
    ModelManager model_manager_;
    PipelineEventCallback event_callback_;
    
    // Task management
    mutable std::mutex tasks_mutex_;
    std::map<std::string, GenerationTask> tasks_;
    std::atomic<int> next_task_id_;
    
    // Worker threads
    std::vector<std::thread> workers_;
    std::queue<QueuedTask> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> stop_workers_;
};

// ImagePipeline implementation
ImagePipeline::ImagePipeline() : pimpl_(std::make_unique<Impl>()) {}

ImagePipeline::~ImagePipeline() = default;

bool ImagePipeline::initialize(const PipelineConfig& config) {
    return pimpl_->initialize(config);
}

void ImagePipeline::shutdown() {
    pimpl_->shutdown();
}

bool ImagePipeline::is_initialized() const {
    return pimpl_->is_initialized();
}

void ImagePipeline::set_config(const PipelineConfig& config) {
    pimpl_->set_config(config);
}

PipelineConfig ImagePipeline::get_config() const {
    return pimpl_->get_config();
}

std::string ImagePipeline::generate_image_async(const ImageGenerationParams& params,
                                               PipelineProgressCallback progress_cb,
                                               TaskCompletedCallback completed_cb) {
    return pimpl_->generate_image_async(params, progress_cb, completed_cb);
}

ImageGenerationResult ImagePipeline::generate_image_sync(const ImageGenerationParams& params,
                                                        PipelineProgressCallback progress_cb) {
    return pimpl_->generate_image_sync(params, progress_cb);
}

bool ImagePipeline::cancel_task(const std::string& task_id) {
    return pimpl_->cancel_task(task_id);
}

bool ImagePipeline::cancel_all_tasks() {
    return pimpl_->cancel_all_tasks();
}

GenerationTask ImagePipeline::get_task_info(const std::string& task_id) const {
    return pimpl_->get_task_info(task_id);
}

std::vector<GenerationTask> ImagePipeline::get_all_tasks() const {
    return pimpl_->get_all_tasks();
}

std::vector<GenerationTask> ImagePipeline::get_tasks_by_status(TaskStatus status) const {
    return pimpl_->get_tasks_by_status(status);
}

void ImagePipeline::clear_completed_tasks() {
    pimpl_->clear_completed_tasks();
}

void ImagePipeline::clear_all_tasks() {
    pimpl_->clear_all_tasks();
}

bool ImagePipeline::is_busy() const {
    return pimpl_->is_busy();
}

int ImagePipeline::get_active_task_count() const {
    return pimpl_->get_active_task_count();
}

int ImagePipeline::get_pending_task_count() const {
    return pimpl_->get_pending_task_count();
}

std::string ImagePipeline::get_current_status() const {
    return pimpl_->get_current_status();
}

bool ImagePipeline::load_pipeline_models() {
    return pimpl_->load_pipeline_models();
}

bool ImagePipeline::unload_pipeline_models() {
    return pimpl_->unload_pipeline_models();
}

void ImagePipeline::set_event_callback(PipelineEventCallback callback) {
    pimpl_->set_event_callback(callback);
}

void ImagePipeline::enable_debug_mode(bool enabled) {
    pimpl_->enable_debug_mode(enabled);
}

// Utility function implementations
std::string pipeline_type_to_string(PipelineType type) {
    switch (type) {
        case PipelineType::TEXT_TO_IMAGE: return "text_to_image";
        case PipelineType::IMAGE_TO_IMAGE: return "image_to_image";
        case PipelineType::INPAINTING: return "inpainting";
        case PipelineType::OUTPAINTING: return "outpainting";
        case PipelineType::UPSCALING: return "upscaling";
        case PipelineType::CONTROLNET: return "controlnet";
        case PipelineType::DEPTH_TO_IMAGE: return "depth_to_image";
        case PipelineType::POSE_TO_IMAGE: return "pose_to_image";
        default: return "unknown";
    }
}

PipelineType string_to_pipeline_type(const std::string& type_str) {
    if (type_str == "text_to_image") return PipelineType::TEXT_TO_IMAGE;
    if (type_str == "image_to_image") return PipelineType::IMAGE_TO_IMAGE;
    if (type_str == "inpainting") return PipelineType::INPAINTING;
    if (type_str == "outpainting") return PipelineType::OUTPAINTING;
    if (type_str == "upscaling") return PipelineType::UPSCALING;
    if (type_str == "controlnet") return PipelineType::CONTROLNET;
    if (type_str == "depth_to_image") return PipelineType::DEPTH_TO_IMAGE;
    if (type_str == "pose_to_image") return PipelineType::POSE_TO_IMAGE;
    return PipelineType::TEXT_TO_IMAGE;
}

std::string task_status_to_string(TaskStatus status) {
    switch (status) {
        case TaskStatus::PENDING: return "pending";
        case TaskStatus::RUNNING: return "running";
        case TaskStatus::COMPLETED: return "completed";
        case TaskStatus::FAILED: return "failed";
        case TaskStatus::CANCELLED: return "cancelled";
        default: return "unknown";
    }
}

TaskStatus string_to_task_status(const std::string& status_str) {
    if (status_str == "pending") return TaskStatus::PENDING;
    if (status_str == "running") return TaskStatus::RUNNING;
    if (status_str == "completed") return TaskStatus::COMPLETED;
    if (status_str == "failed") return TaskStatus::FAILED;
    if (status_str == "cancelled") return TaskStatus::CANCELLED;
    return TaskStatus::PENDING;
}

std::string generate_task_id() {
    static std::atomic<int> counter{1};
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    return "task_" + std::to_string(timestamp) + "_" + std::to_string(counter++);
}

bool validate_generation_params(const ImageGenerationParams& params) {
    if (params.prompt.empty()) {
        return false;
    }
    if (params.width <= 0 || params.height <= 0) {
        return false;
    }
    if (params.steps <= 0) {
        return false;
    }
    if (params.cfg_scale < 0.0f) {
        return false;
    }
    return true;
}

std::string format_generation_time(double seconds) {
    if (seconds < 1.0) {
        return std::to_string(static_cast<int>(seconds * 1000)) + "ms";
    } else if (seconds < 60.0) {
        return std::to_string(static_cast<int>(seconds)) + "s";
    } else {
        int minutes = static_cast<int>(seconds / 60);
        int secs = static_cast<int>(seconds) % 60;
        return std::to_string(minutes) + "m " + std::to_string(secs) + "s";
    }
}

// Factory function implementations
std::unique_ptr<ImagePipeline> create_text_to_image_pipeline(const PipelineConfig& config) {
    auto pipeline = std::make_unique<ImagePipeline>();
    PipelineConfig cfg = config;
    cfg.type = PipelineType::TEXT_TO_IMAGE;
    if (pipeline->initialize(cfg)) {
        return pipeline;
    }
    return nullptr;
}

std::unique_ptr<ImagePipeline> create_image_to_image_pipeline(const PipelineConfig& config) {
    auto pipeline = std::make_unique<ImagePipeline>();
    PipelineConfig cfg = config;
    cfg.type = PipelineType::IMAGE_TO_IMAGE;
    if (pipeline->initialize(cfg)) {
        return pipeline;
    }
    return nullptr;
}

std::unique_ptr<ImagePipeline> create_inpainting_pipeline(const PipelineConfig& config) {
    auto pipeline = std::make_unique<ImagePipeline>();
    PipelineConfig cfg = config;
    cfg.type = PipelineType::INPAINTING;
    if (pipeline->initialize(cfg)) {
        return pipeline;
    }
    return nullptr;
}

std::unique_ptr<ImagePipeline> create_controlnet_pipeline(const PipelineConfig& config) {
    auto pipeline = std::make_unique<ImagePipeline>();
    PipelineConfig cfg = config;
    cfg.type = PipelineType::CONTROLNET;
    if (pipeline->initialize(cfg)) {
        return pipeline;
    }
    return nullptr;
}

} // namespace invokeai
} // namespace duorou