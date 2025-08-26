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

// 管道类型枚举
enum class PipelineType {
    TEXT_TO_IMAGE,      // 文本到图像
    IMAGE_TO_IMAGE,     // 图像到图像
    INPAINTING,         // 图像修复
    OUTPAINTING,        // 图像扩展
    UPSCALING,          // 图像放大
    CONTROLNET,         // ControlNet引导生成
    DEPTH_TO_IMAGE,     // 深度图到图像
    POSE_TO_IMAGE       // 姿态到图像
};

// 管道配置
struct PipelineConfig {
    PipelineType type = PipelineType::TEXT_TO_IMAGE;
    std::string model_name;
    std::string vae_model;
    std::string controlnet_model;
    std::string lora_models; // 逗号分隔的LoRA模型列表
    
    // 设备和性能设置
    std::string device = "auto";
    std::string precision = "auto";
    int num_threads = 0; // 0表示自动
    bool enable_memory_efficient = true;
    bool enable_attention_slicing = true;
    bool enable_cpu_offload = false;
    
    // 安全设置
    bool enable_safety_checker = true;
    bool enable_watermark = false;
    
    // 缓存设置
    bool enable_model_cache = true;
    size_t max_cache_size_mb = 2048;
};

// 生成任务状态
enum class TaskStatus {
    PENDING,     // 等待中
    RUNNING,     // 运行中
    COMPLETED,   // 已完成
    FAILED,      // 失败
    CANCELLED    // 已取消
};

// 生成任务信息
struct GenerationTask {
    std::string task_id;
    PipelineType pipeline_type;
    ImageGenerationParams params;
    TaskStatus status = TaskStatus::PENDING;
    std::string error_message;
    float progress = 0.0f; // 0.0 - 1.0
    std::string current_step;
    
    // 时间信息
    std::chrono::system_clock::time_point created_time;
    std::chrono::system_clock::time_point started_time;
    std::chrono::system_clock::time_point completed_time;
    
    // 结果
    ImageGenerationResult result;
};

// 批量生成参数
struct BatchGenerationParams {
    std::vector<ImageGenerationParams> params_list;
    int max_concurrent_tasks = 1;
    bool stop_on_error = false;
    std::string output_directory;
    std::string naming_pattern = "batch_{index}_{timestamp}";
};

// 进度回调函数类型
using PipelineProgressCallback = std::function<void(const std::string& task_id, float progress, const std::string& step)>;
using TaskCompletedCallback = std::function<void(const GenerationTask& task)>;

// 图像生成管道类
class ImagePipeline {
public:
    ImagePipeline();
    ~ImagePipeline();
    
    // 初始化和配置
    bool initialize(const PipelineConfig& config);
    void shutdown();
    bool is_initialized() const;
    
    // 配置管理
    void set_config(const PipelineConfig& config);
    PipelineConfig get_config() const;
    void update_config(const std::map<std::string, std::string>& updates);
    
    // 模型管理
    bool load_pipeline_models();
    bool unload_pipeline_models();
    bool switch_model(const std::string& model_name);
    std::vector<std::string> get_compatible_models(PipelineType type) const;
    
    // 单个图像生成
    std::string generate_image_async(const ImageGenerationParams& params,
                                    PipelineProgressCallback progress_cb = nullptr,
                                    TaskCompletedCallback completed_cb = nullptr);
    ImageGenerationResult generate_image_sync(const ImageGenerationParams& params,
                                             PipelineProgressCallback progress_cb = nullptr);
    
    // 批量生成
    std::vector<std::string> generate_batch_async(const BatchGenerationParams& batch_params,
                                                  PipelineProgressCallback progress_cb = nullptr,
                                                  TaskCompletedCallback completed_cb = nullptr);
    std::vector<ImageGenerationResult> generate_batch_sync(const BatchGenerationParams& batch_params,
                                                          PipelineProgressCallback progress_cb = nullptr);
    
    // 任务管理
    bool cancel_task(const std::string& task_id);
    bool cancel_all_tasks();
    GenerationTask get_task_info(const std::string& task_id) const;
    std::vector<GenerationTask> get_all_tasks() const;
    std::vector<GenerationTask> get_tasks_by_status(TaskStatus status) const;
    void clear_completed_tasks();
    void clear_all_tasks();
    
    // 管道状态
    bool is_busy() const;
    int get_active_task_count() const;
    int get_pending_task_count() const;
    std::string get_current_status() const;
    
    // 预处理和后处理
    bool preprocess_input(ImageGenerationParams& params) const;
    bool postprocess_output(ImageGenerationResult& result) const;
    
    // 图像处理工具
    bool resize_image(const std::string& input_path, const std::string& output_path, 
                     int width, int height) const;
    bool crop_image(const std::string& input_path, const std::string& output_path,
                   int x, int y, int width, int height) const;
    bool apply_mask(const std::string& image_path, const std::string& mask_path,
                   const std::string& output_path) const;
    
    // 质量评估
    float evaluate_image_quality(const std::string& image_path) const;
    bool detect_nsfw_content(const std::string& image_path) const;
    std::map<std::string, float> analyze_image_metrics(const std::string& image_path) const;
    
    // 优化和调试
    void optimize_memory_usage();
    void clear_cache();
    std::map<std::string, std::string> get_performance_stats() const;
    void enable_debug_mode(bool enabled);
    
    // 事件回调
    using PipelineEventCallback = std::function<void(const std::string& event, const std::string& data)>;
    void set_event_callback(PipelineEventCallback callback);
    
private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

// 管道工厂函数
std::unique_ptr<ImagePipeline> create_text_to_image_pipeline(const PipelineConfig& config = {});
std::unique_ptr<ImagePipeline> create_image_to_image_pipeline(const PipelineConfig& config = {});
std::unique_ptr<ImagePipeline> create_inpainting_pipeline(const PipelineConfig& config = {});
std::unique_ptr<ImagePipeline> create_controlnet_pipeline(const PipelineConfig& config = {});

// 工具函数
std::string pipeline_type_to_string(PipelineType type);
PipelineType string_to_pipeline_type(const std::string& type_str);
std::string task_status_to_string(TaskStatus status);
TaskStatus string_to_task_status(const std::string& status_str);
std::string generate_task_id();
bool validate_generation_params(const ImageGenerationParams& params);
std::string format_generation_time(double seconds);

} // namespace invokeai
} // namespace duorou