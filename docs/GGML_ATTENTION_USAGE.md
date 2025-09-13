# GGML注意力机制使用指南

## 概述

本项目现在支持使用GGML库的优化注意力实现来替代原有的多头注意力算法。GGML提供了高性能的Flash Attention实现，特别适合处理长序列和大规模模型。

## 性能优势

### GGML注意力 vs 原有多头注意力

| 特性 | 原有实现 | GGML实现 |
|------|----------|----------|
| 内存效率 | 标准 | 优化的内存使用模式 |
| 计算速度 | 基础OpenMP并行 | Flash Attention + SIMD优化 |
| 长序列支持 | 内存占用O(n²) | 内存占用优化 |
| 硬件加速 | 有限 | 充分利用现代CPU特性 |
| 数值稳定性 | 标准 | 改进的数值稳定性 |

## 使用方法

### 1. 通过算法工厂创建

```cpp
#include "algorithm_factory.h"

using namespace duorou::extensions::ollama::algorithms;

// 创建GGML注意力算法
auto& manager = AlgorithmManager::getInstance();
auto ggml_attention = manager.createAttentionAlgorithm("ggml_attention");

// 配置模型参数
ModelConfig model_config;
model_config.num_attention_heads = 12;
model_config.hidden_size = 768;
model_config.max_position_embeddings = 2048;

// 配置算法上下文
AlgorithmContext context;
context.num_threads = 4;
context.use_simd = true;
context.device = "cpu";

// 初始化算法
if (ggml_attention->initialize(model_config, context)) {
    // 准备输入张量
    Tensor query({1, 512, 768});  // [batch, seq_len, hidden_dim]
    Tensor key({1, 512, 768});
    Tensor value({1, 512, 768});
    
    // 执行注意力计算
    auto result = ggml_attention->compute(query, key, value, nullptr, 1.0f);
    
    if (!result.data.empty()) {
        std::cout << "注意力计算成功!" << std::endl;
    }
}
```

### 2. 配置选项

#### 模型配置 (ModelConfig)
- `num_attention_heads`: 注意力头数量
- `hidden_size`: 隐藏层维度
- `max_position_embeddings`: 最大位置编码长度
- `num_key_value_heads`: KV头数量（用于GQA）

#### 算法上下文 (AlgorithmContext)
- `num_threads`: 并行线程数
- `use_simd`: 是否启用SIMD优化
- `device`: 计算设备（"cpu"）
- `verbose`: 是否输出详细日志

### 3. 支持的功能

#### 基础注意力计算
```cpp
Tensor result = attention->compute(query, key, value, mask, scale);
```

#### 带缓存的注意力计算
```cpp
Tensor result = attention->computeWithCache(
    query, key, value, 
    key_cache, value_cache, 
    cache_position, head_idx, 
    mask, scale
);
```

## 性能测试

运行性能对比测试：

```bash
cd /path/to/duorou
make examples
./build/examples/ggml_attention_example
```

## 迁移指南

### 从MultiHeadAttention迁移到GGMLAttention

1. **更新算法类型**：
   ```cpp
   // 原有方式
   auto attention = manager.createAttentionAlgorithm("multi_head_attention");
   
   // 新方式
   auto attention = manager.createAttentionAlgorithm("ggml_attention");
   ```

2. **更新初始化参数**：
   ```cpp
   // 确保ModelConfig包含必要参数
   ModelConfig config;
   config.num_attention_heads = your_num_heads;
   config.hidden_size = your_hidden_size;
   
   // 使用新的初始化接口
   attention->initialize(config, context);
   ```

3. **检查输入张量格式**：
   - 确保张量形状为 [batch_size, seq_len, hidden_dim]
   - 数据类型为 float32
   - 内存布局为连续存储

## 故障排除

### 常见问题

1. **编译错误：找不到GGML头文件**
   - 确保项目正确链接了GGML库
   - 检查CMakeLists.txt中的依赖配置

2. **运行时错误：初始化失败**
   - 检查ModelConfig参数是否正确设置
   - 确保hidden_size能被num_attention_heads整除

3. **性能不如预期**
   - 启用SIMD优化：`context.use_simd = true`
   - 调整线程数：`context.num_threads = std::thread::hardware_concurrency()`
   - 对于长序列（>1024），GGML优势更明显

### 调试技巧

1. **启用详细日志**：
   ```cpp
   context.verbose = true;
   ```

2. **性能分析**：
   ```cpp
   auto start = std::chrono::high_resolution_clock::now();
   auto result = attention->compute(...);
   auto end = std::chrono::high_resolution_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
   std::cout << "计算时间: " << duration.count() << " 微秒" << std::endl;
   ```

## 最佳实践

1. **选择合适的算法**：
   - 短序列（<512）：可以使用原有实现
   - 长序列（>=512）：推荐使用GGML实现
   - 生产环境：推荐使用GGML实现

2. **内存管理**：
   - 启用内存池：`tensor.use_memory_pool = true`
   - 复用张量对象以减少内存分配

3. **并行优化**：
   - 根据硬件设置合适的线程数
   - 避免过度并行化导致的上下文切换开销

## 未来计划

- [ ] 支持GPU加速（CUDA/OpenCL）
- [ ] 添加更多GGML优化算法
- [ ] 支持混合精度计算
- [ ] 集成更多Flash Attention变体