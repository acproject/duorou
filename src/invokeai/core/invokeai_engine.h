#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <map>

namespace duorou {
namespace invokeai {

// 图像生成参数结构
struct ImageGenerationParams {
    std::string prompt;
    std::string negative_prompt;
    int width = 512;
    int height = 512;
    int steps = 20;
    float cfg_scale = 7.5f;
    int seed = -1;
    std::string sampler = "euler_a";
    std::string model_name;
    float strength = 0.8f; // for img2img
    std::string init_image_path; // for img2img
};

// 图像生成结果
struct ImageGenerationResult {
    bool success = false;
    std::string error_message;
    std::string output_image_path;
    std::string metadata; // JSON格式的元数据
    double generation_time = 0.0; // 生成时间（秒）
};

// 模型信息
struct ModelInfo {
    std::string name;
    std::string path;
    std::string type; // "checkpoint", "diffusers", "safetensors"
    std::string description;
    bool is_loaded = false;
    size_t memory_usage = 0; // 内存使用量（字节）
};

// 进度回调函数类型
using ProgressCallback = std::function<void(int step, int total_steps, const std::string& status)>;

// InvokeAI engine core class
class InvokeAIEngine {
public:
    InvokeAIEngine();
    ~InvokeAIEngine();

    // Initialize engine
    bool initialize(const std::string& models_path = "");
    
    // Shutdown engine
    void shutdown();
    
    // Model management
    bool load_model(const std::string& model_name);
    bool unload_model();
    std::vector<ModelInfo> get_available_models() const;
    ModelInfo get_current_model() const;
    bool is_model_loaded() const;
    
    // Image generation
    ImageGenerationResult generate_image(const ImageGenerationParams& params);
    ImageGenerationResult generate_image_async(const ImageGenerationParams& params, 
                                              ProgressCallback progress_cb = nullptr);
    
    // img2img functionality
    ImageGenerationResult image_to_image(const ImageGenerationParams& params);
    
    // Cancel current generation task
    void cancel_generation();
    
    // Get engine status
    bool is_busy() const;
    std::string get_status() const;
    
    // Set configuration
    void set_device(const std::string& device); // "cpu", "cuda", "mps"
    void set_precision(const std::string& precision); // "fp32", "fp16"
    void set_threads(int threads);
    void set_memory_limit(size_t limit_mb);
    
    // Get system information
    std::map<std::string, std::string> get_system_info() const;
    
private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

// Factory function
std::unique_ptr<InvokeAIEngine> create_invokeai_engine();

} // namespace invokeai
} // namespace duorou