#include "ggml_feed_forward.h"
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <chrono>

extern "C" {
#include "../../../../third_party/llama.cpp/ggml/include/ggml.h"
#include "../../../../third_party/llama.cpp/ggml/include/ggml-cpu.h"
}
#include <iostream>

namespace duorou {
namespace extensions {
namespace ollama {
namespace algorithms {

// GGMLFeedForward实现
GGMLFeedForward::~GGMLFeedForward() {
    cleanup();
}

bool GGMLFeedForward::initialize(const ModelConfig &config, const AlgorithmContext &context) {
    config_ = config;
    context_ = context;
    
    // 计算所需内存大小基于模型配置
    size_t estimated_memory = static_cast<size_t>(config.hidden_size) * config.intermediate_size * 4 * sizeof(float);
    size_t min_memory = 1024 * 1024 * 512;  // 最小512MB
    size_t context_memory = std::max(estimated_memory * 2, min_memory);  // 预留2倍空间
    
    // 初始化GGML上下文
    ggml_init_params params = {
        /* .mem_size   = */ context_memory,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false
    };
    
    log("INFO", "Initializing GGML context with " + std::to_string(context_memory / (1024 * 1024)) + "MB memory");
    
    ctx_ = ggml_init(params);
    if (!ctx_) {
        log("ERROR", "Failed to initialize GGML context");
        return false;
    }
    
    // 预分配工作缓冲区 - 基于模型大小动态调整
    work_buffer_size_ = std::max(static_cast<size_t>(1024 * 1024 * 64), context_memory / 8);  // 至少64MB或上下文内存的1/8
    work_buffer_.resize(work_buffer_size_);
    
    log("INFO", "Allocated work buffer: " + std::to_string(work_buffer_size_ / (1024 * 1024)) + "MB");
    
    log("INFO", "GGMLFeedForward initialized successfully");
    return true;
}

bool GGMLFeedForward::validateInput(const Tensor &input) const {
    if (input.shape.empty()) {
        log("ERROR", "Input tensor has empty shape");
        return false;
    }
    
    if (input.data.empty()) {
        log("ERROR", "Input tensor has no data");
        return false;
    }
    
    return true;
}

void GGMLFeedForward::cleanup() {
    if (ctx_) {
        ggml_free(ctx_);
        ctx_ = nullptr;
    }
    
    work_buffer_.clear();
    work_buffer_size_ = 0;
}

bool GGMLFeedForward::allocateWorkBuffer(size_t size) {
    if (size > work_buffer_size_) {
        work_buffer_size_ = std::max(size, work_buffer_size_ * 2);
        work_buffer_.resize(work_buffer_size_);
    }
    return true;
}

struct ggml_tensor *GGMLFeedForward::tensorToGGML(struct ggml_context *ctx, const Tensor &tensor) {
    if (tensor.shape.empty()) {
        log("ERROR", "Tensor has empty shape");
        return nullptr;
    }
    
    std::vector<int64_t> dims(tensor.shape.begin(), tensor.shape.end());
    
    // 检查内存使用情况
    size_t tensor_size = tensor.data.size() * sizeof(float);
    size_t available_mem = ggml_get_mem_size(ctx) - ggml_used_mem(ctx);
    
    if (tensor_size > available_mem) {
        log("ERROR", "Insufficient memory for tensor creation. Need: " + std::to_string(tensor_size / (1024 * 1024)) + "MB, Available: " + std::to_string(available_mem / (1024 * 1024)) + "MB");
        return nullptr;
    }
    
    struct ggml_tensor *result = ggml_new_tensor(ctx, GGML_TYPE_F32, dims.size(), dims.data());
    
    if (!result) {
        log("ERROR", "Failed to create GGML tensor - out of memory");
        return nullptr;
    }
    
    // 复制数据
    if (!tensor.data.empty()) {
        std::memcpy(result->data, tensor.data.data(), tensor.data.size() * sizeof(float));
    }
    
    return result;
}

Tensor GGMLFeedForward::ggmlToTensor(struct ggml_tensor *ggml_tensor) {
    if (!ggml_tensor) {
        return Tensor();
    }
    
    std::vector<uint32_t> shape;
    for (int i = 0; i < ggml_n_dims(ggml_tensor); ++i) {
        shape.push_back(static_cast<uint32_t>(ggml_tensor->ne[i]));
    }
    
    Tensor result(shape);
    
    if (ggml_tensor->data && result.data.size() > 0) {
        std::memcpy(result.data.data(), ggml_tensor->data, result.data.size() * sizeof(float));
    }
    
    return result;
}

struct ggml_tensor *GGMLFeedForward::applyActivation(struct ggml_context *ctx, struct ggml_tensor *input, const std::string &activation_type) {
    if (activation_type == "swiglu") {
        // SwiGLU: x * sigmoid(x) for GLU variant
        struct ggml_tensor *sigmoid = ggml_sigmoid(ctx, input);
        return ggml_mul(ctx, input, sigmoid);
    } else if (activation_type == "gelu") {
        // GELU
        return ggml_gelu(ctx, input);
    } else if (activation_type == "silu") {
        // SiLU/Swish
        return ggml_silu(ctx, input);
    } else {
        // 默认ReLU
        return ggml_relu(ctx, input);
    }
}

Tensor GGMLFeedForward::compute(const Tensor &input, const Tensor &gate_weights,
                               const Tensor &up_weights, const Tensor &down_weights) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (!validateInput(input)) {
        return Tensor();
    }
    
    if (!ctx_) {
        log("ERROR", "GGML context not initialized");
        return Tensor();
    }
    
    // 转换张量到GGML格式
    struct ggml_tensor *ggml_input = tensorToGGML(ctx_, input);
    struct ggml_tensor *ggml_gate_weights = tensorToGGML(ctx_, gate_weights);
    struct ggml_tensor *ggml_up_weights = tensorToGGML(ctx_, up_weights);
    struct ggml_tensor *ggml_down_weights = tensorToGGML(ctx_, down_weights);
    
    if (!ggml_input || !ggml_gate_weights || !ggml_up_weights || !ggml_down_weights) {
        log("ERROR", "Failed to convert tensors to GGML format");
        return Tensor();
    }
    
    // 构建计算图
    struct ggml_tensor *gate_proj = ggml_mul_mat(ctx_, ggml_gate_weights, ggml_input);
    struct ggml_tensor *up_proj = ggml_mul_mat(ctx_, ggml_up_weights, ggml_input);
    
    // 应用激活函数（默认SwiGLU）
    struct ggml_tensor *activated = applyActivation(ctx_, gate_proj, "swiglu");
    
    // 元素级乘法
    struct ggml_tensor *intermediate = ggml_mul(ctx_, activated, up_proj);
    
    // 最终投影
    struct ggml_tensor *output = ggml_mul_mat(ctx_, ggml_down_weights, intermediate);
    
    // 构建前向计算图
    gf_ = ggml_new_graph(ctx_);
    ggml_build_forward_expand(gf_, output);
    
    // 执行计算
    int n_threads = static_cast<int>(context_.num_threads);
    enum ggml_status status = ggml_graph_compute_with_ctx(ctx_, gf_, n_threads);
    
    if (status != 0) {  // GGML_STATUS_SUCCESS is 0
        log("ERROR", "GGML computation failed with status: " + std::to_string(status));
        return Tensor();
    }
    
    // 转换结果回Tensor格式
    Tensor result = ggmlToTensor(output);
    
    // 清理计算图以释放内存
    if (gf_) {
        ggml_graph_clear(gf_);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    context_.total_time += duration.count() / 1000000.0;
    context_.call_count++;
    
    if (context_.verbose) {
        log("DEBUG", "FeedForward computation completed in " + std::to_string(duration.count()) + " μs");
        log("DEBUG", "Memory usage: " + std::to_string(ggml_used_mem(ctx_) / (1024 * 1024)) + "MB / " + std::to_string(ggml_get_mem_size(ctx_) / (1024 * 1024)) + "MB");
    }
    
    return result;
}

// SwiGLUFeedForward实现
Tensor SwiGLUFeedForward::compute(const Tensor &input, const Tensor &gate_weights,
                                 const Tensor &up_weights, const Tensor &down_weights) {
    return GGMLFeedForward::compute(input, gate_weights, up_weights, down_weights);
}

// GELUFeedForward实现
Tensor GELUFeedForward::compute(const Tensor &input, const Tensor &gate_weights,
                               const Tensor &up_weights, const Tensor &down_weights) {
    // 使用GELU激活函数
    if (!validateInput(input)) {
        return Tensor();
    }
    
    if (!ctx_) {
        log("ERROR", "GGML context not initialized");
        return Tensor();
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 转换张量到GGML格式
    struct ggml_tensor *ggml_input = tensorToGGML(ctx_, input);
    struct ggml_tensor *ggml_gate_weights = tensorToGGML(ctx_, gate_weights);
    struct ggml_tensor *ggml_up_weights = tensorToGGML(ctx_, up_weights);
    struct ggml_tensor *ggml_down_weights = tensorToGGML(ctx_, down_weights);
    
    if (!ggml_input || !ggml_gate_weights || !ggml_up_weights || !ggml_down_weights) {
        log("ERROR", "Failed to convert tensors to GGML format");
        return Tensor();
    }
    
    // 对于GELU，我们使用标准的FFN结构
    // 首先合并gate和up权重（如果是GELU）
    struct ggml_tensor *intermediate = ggml_mul_mat(ctx_, ggml_up_weights, ggml_input);
    
    // 应用GELU激活
    struct ggml_tensor *activated = ggml_gelu(ctx_, intermediate);
    
    // 最终投影
    struct ggml_tensor *output = ggml_mul_mat(ctx_, ggml_down_weights, activated);
    
    // 构建前向计算图
    gf_ = ggml_new_graph(ctx_);
    ggml_build_forward_expand(gf_, output);
    
    // 执行计算
    int n_threads = static_cast<int>(context_.num_threads);
    enum ggml_status status = ggml_graph_compute_with_ctx(ctx_, gf_, n_threads);
    
    if (status != 0) {  // GGML_STATUS_SUCCESS is 0
        log("ERROR", "GGML computation failed with status: " + std::to_string(status));
        return Tensor();
    }
    
    // 转换结果回Tensor格式
    Tensor result = ggmlToTensor(output);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    context_.total_time += duration.count() / 1000000.0;
    context_.call_count++;
    
    if (context_.verbose) {
        log("DEBUG", "GELU FeedForward computation completed in " + std::to_string(duration.count()) + " μs");
    }
    
    return result;
}

// 实现日志方法
void GGMLFeedForward::log(const std::string& level, const std::string& message) const {
    if (context_.verbose || level == "ERROR") {
        std::cerr << "[" << level << "] GGMLFeedForward: " << message << std::endl;
    }
}

} // namespace algorithms
} // namespace ollama
} // namespace extensions
} // namespace duorou