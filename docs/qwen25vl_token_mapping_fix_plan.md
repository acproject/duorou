# Qwen2.5VL Token映射问题修复计划

## 问题分析

### 当前问题
根据debug日志显示，当前项目中的token映射存在以下问题：

1. **Token映射不正确**：debug输出显示的token如`[PAD270]`、`ë§Ī`、`Ġbun`等看起来像是原始的BPE token，而不是正确的中文字符
2. **词汇表加载问题**：项目使用占位符词汇表而非从GGUF文件正确解析
3. **特殊token处理**：硬编码的特殊token映射可能不完整或不正确

### Debug日志分析
```
[DEBUG] Token 151935: [PAD270]
[DEBUG] Token 125544: ë§Ī  
[DEBUG] Token 44821: Ġbun
[DEBUG] Token 56064: .EqualTo
[DEBUG] Token 133718: ÙħÙĪØ§Ø¬Ùĩ
[DEBUG] Token 29391: .plugins
[DEBUG] Token 131840: Ġnghá»ī
[DEBUG] Token 115382: æĴŀåĩ»
[DEBUG] Token 22828: ĠmarginTop
```

这些token显示了以下问题：
- Token 151935应该是`<|im_end|>`而不是`[PAD270]`
- 中文相关的token显示为乱码
- 存在大量英文编程相关的token

## Ollama处理方式分析

### Ollama的Qwen2.5VL处理流程

1. **词汇表解析**：
   - 支持SentencePiece (`tokenizer.model`) 和 HuggingFace格式 (`tokenizer.json`)
   - 从`tokenizer.json`中解析vocab映射和added_tokens
   - 正确处理特殊token的ID和内容映射

2. **预处理器识别**：
   - 通过SHA256哈希识别预处理器类型
   - Qwen2对应的哈希：`1ff7f41064896984db5d1bb6ff64fa4bc29007d08c1b439e505b7392777a319e`
   - 设置正确的预处理器为`qwen2`

3. **Token类型分类**：
   - `tokenTypeNormal` (1): 普通token
   - `tokenTypeControl` (3): 控制token（特殊token）
   - `tokenTypeUserDefined` (4): 用户定义token
   - `tokenTypeByte` (6): 字节级token

4. **特殊token处理**：
   - 从`tokenizer_config.json`和`generation_config.json`解析特殊token
   - 支持BOS、EOS、UNK、PAD等特殊token的正确映射

## 修复计划

### 阶段1：词汇表解析修复

#### 1.1 增强GGUF词汇表解析
**文件**：`src/extensions/ollama/qwen25vl_inference_engine.cpp`

**修改内容**：
- 修复`loadVocabulary()`函数中的GGUF解析逻辑
- 正确解析`tokenizer.ggml.tokens`数组
- 添加对`tokenizer.ggml.token_type`和`tokenizer.ggml.scores`的解析
- 实现正确的字符串解码（UTF-8处理）

```cpp
// 需要修复的关键代码段
bool Qwen25VLInferenceEngine::loadVocabulary() {
    // 1. 正确解析GGUF中的tokens数组
    // 2. 处理UTF-8编码
    // 3. 建立正确的token ID到字符串的映射
    // 4. 加载token类型和分数信息
}
```

#### 1.2 添加预处理器支持
**新文件**：`src/extensions/ollama/qwen2_preprocessor.h/cpp`

**功能**：
- 实现Qwen2特定的预处理逻辑
- 支持正确的Unicode分割
- 处理中文字符的正确分词

### 阶段2：TextProcessor改进

#### 2.1 SentencePiece处理器增强
**文件**：`src/text_processor.cpp`

**修改内容**：
- 改进`SentencePieceProcessor`的decode方法
- 添加对Qwen2特定token格式的支持
- 正确处理特殊字符和Unicode

#### 2.2 BPE处理器修复
**文件**：`src/text_processor.cpp`

**修改内容**：
- 修复BPE解码中的字符合并逻辑
- 添加对`Ġ`前缀的正确处理（GPT-style BPE）
- 实现正确的Unicode字符重建

### 阶段3：特殊Token处理

#### 3.1 特殊Token映射表
**新文件**：`src/extensions/ollama/qwen25vl_special_tokens.h`

**内容**：
```cpp
namespace duorou {
namespace extensions {
namespace ollama {

struct Qwen25VLSpecialTokens {
    static constexpr int32_t IM_START = 151645;
    static constexpr int32_t IM_END = 151935;
    static constexpr int32_t ENDOFTEXT = 151643;
    
    static const std::unordered_map<int32_t, std::string> SPECIAL_TOKEN_MAP;
};

}
}
}
```

#### 3.2 动态特殊Token加载
**修改文件**：`src/extensions/ollama/qwen25vl_inference_engine.cpp`

**功能**：
- 从GGUF文件动态加载特殊token定义
- 支持模型特定的特殊token配置

### 阶段4：测试和验证

#### 4.1 单元测试
**新文件**：`tests/test_qwen25vl_tokenization.cpp`

**测试内容**：
- 中文文本的正确tokenization和detokenization
- 特殊token的正确处理
- 混合中英文本的处理
- 边界情况测试

#### 4.2 集成测试
**测试用例**：
```cpp
// 测试用例示例
TEST(Qwen25VLTokenization, ChineseText) {
    auto engine = std::make_unique<Qwen25VLInferenceEngine>();
    engine->loadModel("path/to/qwen25vl.gguf");
    
    std::string input = "你好";
    auto tokens = engine->tokenize(input);
    std::string output = engine->detokenize(tokens);
    
    EXPECT_EQ(input, output);
}
```

## 实施优先级

### 高优先级（立即实施）
1. 修复GGUF词汇表解析逻辑
2. 改进TextProcessor的decode方法
3. 添加正确的特殊token映射

### 中优先级（后续实施）
1. 实现Qwen2预处理器
2. 添加完整的单元测试
3. 性能优化

### 低优先级（可选）
1. 支持其他tokenizer格式
2. 添加tokenizer配置文件支持

## 预期效果

修复完成后，应该能够看到：
1. 中文文本正确显示而非乱码
2. Token映射正确（如Token 125544应显示为"你"而非"ë§Ī"）
3. 特殊token正确识别和处理
4. 生成的文本质量显著提升

## 风险评估

### 技术风险
- GGUF文件格式解析的复杂性
- Unicode处理的正确性
- 与现有代码的兼容性

### 缓解措施
- 分阶段实施，每个阶段都进行充分测试
- 保留原有代码作为fallback
- 添加详细的日志和错误处理

## 时间估算

- 阶段1：2-3天
- 阶段2：2-3天  
- 阶段3：1-2天
- 阶段4：1-2天

**总计**：6-10天

---

*文档创建时间：2025年1月25日*
*最后更新：2025年1月25日*