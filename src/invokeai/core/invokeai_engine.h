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

// InvokeAI引擎核心类
class InvokeAIEngine {
public:
    InvokeAIEngine();
    ~InvokeAIEngine();

    // 初始化引擎
    bool initialize(const std::string& models_path = "");
    
    // 关闭引擎
    void shutdown();
    
    // 模型管理
    bool load_model(const std::string& model_name);
    bool unload_model();
    std::vector<ModelInfo> get_available_models() const;
    ModelInfo get_current_model() const;
    bool is_model_loaded() const;
    
    // 图像生成
    ImageGenerationResult generate_image(const ImageGenerationParams& params);
    ImageGenerationResult generate_image_async(const ImageGenerationParams& params, 
                                              ProgressCallback progress_cb = nullptr);
    
    // img2img功能
    ImageGenerationResult image_to_image(const ImageGenerationParams& params);
    
    // 取消当前生成任务
    void cancel_generation();
    
    // 获取引擎状态
    bool is_busy() const;
    std::string get_status() const;
    
    // 设置配置
    void set_device(const std::string& device); // "cpu", "cuda", "mps"
    void set_precision(const std::string& precision); // "fp32", "fp16"
    void set_threads(int threads);
    void set_memory_limit(size_t limit_mb);
    
    // 获取系统信息
    std::map<std::string, std::string> get_system_info() const;
    
private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

// 工厂函数
std::unique_ptr<InvokeAIEngine> create_invokeai_engine();

} // namespace invokeai
} // namespace duorou