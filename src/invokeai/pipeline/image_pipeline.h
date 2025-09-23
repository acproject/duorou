#pragma once

#include "../core/invokeai_engine.h"
#include "../models/model_manager.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <map>
#include <future>

namespace duorou {
namespace invokeai {

// Pipeline type enumeration
enum class PipelineType {
    TEXT_TO_IMAGE,      // Text to image
    IMAGE_TO_IMAGE,     // Image to image
    INPAINTING,         // Image inpainting
    OUTPAINTING,        // Image outpainting
    UPSCALING,          // Image upscaling
    CONTROLNET,         // ControlNet guided generation
    DEPTH_TO_IMAGE,     // Depth map to image
    POSE_TO_IMAGE       // Pose to image
};

// Pipeline configuration
struct PipelineConfig {
    PipelineType type = PipelineType::TEXT_TO_IMAGE;
    std::string model_name;
    std::string vae_model;
    std::string controlnet_model;
    std::string lora_models; // Comma-separated LoRA model list
    
    // Device and performance settings
    std::string device = "auto";
    std::string precision = "auto";
    int num_threads = 0; // 0 means automatic
    bool enable_memory_efficient = true;
    bool enable_attention_slicing = true;
    bool enable_cpu_offload = false;
    
    // Safety settings
    bool enable_safety_checker = true;
    bool enable_watermark = false;
    
    // Cache settings
    bool enable_model_cache = true;
    size_t max_cache_size_mb = 2048;
};

// Generation task status
enum class TaskStatus {
    PENDING,     // Pending
    RUNNING,     // Running
    COMPLETED,   // Completed
    FAILED,      // Failed
    CANCELLED    // Cancelled
};

// Generation task information
struct GenerationTask {
    std::string task_id;
    PipelineType pipeline_type;
    ImageGenerationParams params;
    TaskStatus status = TaskStatus::PENDING;
    std::string error_message;
    float progress = 0.0f; // 0.0 - 1.0
    std::string current_step;
    
    // Time information
    std::chrono::system_clock::time_point created_time;
    std::chrono::system_clock::time_point started_time;
    std::chrono::system_clock::time_point completed_time;
    
    // Result
    ImageGenerationResult result;
};

// Batch generation parameters
struct BatchGenerationParams {
    std::vector<ImageGenerationParams> params_list;
    int max_concurrent_tasks = 1;
    bool stop_on_error = false;
    std::string output_directory;
    std::string naming_pattern = "batch_{index}_{timestamp}";
};

// Progress callback function types
using PipelineProgressCallback = std::function<void(const std::string& task_id, float progress, const std::string& step)>;
using TaskCompletedCallback = std::function<void(const GenerationTask& task)>;

// Image generation pipeline class
class ImagePipeline {
public:
    ImagePipeline();
    ~ImagePipeline();
    
    // Initialization and configuration
    bool initialize(const PipelineConfig& config);
    void shutdown();
    bool is_initialized() const;
    
    // Configuration management
    void set_config(const PipelineConfig& config);
    PipelineConfig get_config() const;
    void update_config(const std::map<std::string, std::string>& updates);
    
    // Model management
    bool load_pipeline_models();
    bool unload_pipeline_models();
    bool switch_model(const std::string& model_name);
    std::vector<std::string> get_compatible_models(PipelineType type) const;
    
    // Single image generation
    std::string generate_image_async(const ImageGenerationParams& params,
                                    PipelineProgressCallback progress_cb = nullptr,
                                    TaskCompletedCallback completed_cb = nullptr);
    ImageGenerationResult generate_image_sync(const ImageGenerationParams& params,
                                             PipelineProgressCallback progress_cb = nullptr);
    
    // Batch generation
    std::vector<std::string> generate_batch_async(const BatchGenerationParams& batch_params,
                                                  PipelineProgressCallback progress_cb = nullptr,
                                                  TaskCompletedCallback completed_cb = nullptr);
    std::vector<ImageGenerationResult> generate_batch_sync(const BatchGenerationParams& batch_params,
                                                          PipelineProgressCallback progress_cb = nullptr);
    
    // Task management
    bool cancel_task(const std::string& task_id);
    bool cancel_all_tasks();
    GenerationTask get_task_info(const std::string& task_id) const;
    std::vector<GenerationTask> get_all_tasks() const;
    std::vector<GenerationTask> get_tasks_by_status(TaskStatus status) const;
    void clear_completed_tasks();
    void clear_all_tasks();
    
    // Pipeline status
    bool is_busy() const;
    int get_active_task_count() const;
    int get_pending_task_count() const;
    std::string get_current_status() const;
    
    // Preprocessing and postprocessing
    bool preprocess_input(ImageGenerationParams& params) const;
    bool postprocess_output(ImageGenerationResult& result) const;
    
    // Image processing tools
    bool resize_image(const std::string& input_path, const std::string& output_path, 
                     int width, int height) const;
    bool crop_image(const std::string& input_path, const std::string& output_path,
                   int x, int y, int width, int height) const;
    bool apply_mask(const std::string& image_path, const std::string& mask_path,
                   const std::string& output_path) const;
    
    // Quality assessment
    float evaluate_image_quality(const std::string& image_path) const;
    bool detect_nsfw_content(const std::string& image_path) const;
    std::map<std::string, float> analyze_image_metrics(const std::string& image_path) const;
    
    // Optimization and debugging
    void optimize_memory_usage();
    void clear_cache();
    std::map<std::string, std::string> get_performance_stats() const;
    void enable_debug_mode(bool enabled);
    
    // Event callbacks
    using PipelineEventCallback = std::function<void(const std::string& event, const std::string& data)>;
    void set_event_callback(PipelineEventCallback callback);
    
private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

// Pipeline factory functions
std::unique_ptr<ImagePipeline> create_text_to_image_pipeline(const PipelineConfig& config = {});
std::unique_ptr<ImagePipeline> create_image_to_image_pipeline(const PipelineConfig& config = {});
std::unique_ptr<ImagePipeline> create_inpainting_pipeline(const PipelineConfig& config = {});
std::unique_ptr<ImagePipeline> create_controlnet_pipeline(const PipelineConfig& config = {});

// Utility functions
std::string pipeline_type_to_string(PipelineType type);
PipelineType string_to_pipeline_type(const std::string& type_str);
std::string task_status_to_string(TaskStatus status);
TaskStatus string_to_task_status(const std::string& status_str);
std::string generate_task_id();
bool validate_generation_params(const ImageGenerationParams& params);
std::string format_generation_time(double seconds);

} // namespace invokeai
} // namespace duorou