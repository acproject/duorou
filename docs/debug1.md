下一步调整（建议按优先级执行）


2. 2.
   对齐 GGUF 的模型配置（高优先）
   
   - 从 GGUF 元数据把以下参数写进文本模型/推理配置：
     - n_layers = block_count
     - n_ctx = context_length
     - n_embd = embedding_length
     - vocab_size = GGUF 词表大小（152064）
     - n_heads / n_kv_heads（如果有）
     - RoPE 参数（频率、base、dim 等）
   - 在 QwenTextModel 初始化时，若有 external_vocabulary_ 则 getVocabSize/内部 config 应统一使用 external_vocabulary_->size()，避免“151936/32000”这样的占位值残留。参考： `QwenTextModel::getVocabSize`
3. 3.
   补齐权重加载映射（高优先）
   
   - 现在只加载了 64 个张量，远不够。需要按 GGUF 的命名规则将每一层的注意力、MLP、层归一化等权重完整映射到 QwenTextModel 的内部结构，确保 forward 能输出真实 logits。
   - 如果你有 multimodal（vision）计划，先把文本路径打通；视觉权重可后续加载。相关骨架在： `qwen_multimodal_model.cpp`
4. 4.
   规范自回归采样循环（高优先）
   
   - 正确流程应为：prefill（编码 prompt 并写入 KV）→ 每步 decode（取上步新 token）→ 计算 logits → 温度缩放 → Top-p/Top-k → 重复惩罚 → 采样 next token → 写入 KV → EOS 检查 → 流式输出累积。
   - 把循环实现到 `MLInferenceEngine::generateWithInternalForward` ，并确保“logits.size() == external_vocabulary_->size()”。否则直接断言并降级走 llama.cpp。
   - 提前停止：从 GGUF 读取 EOS/BOS/UNK/SEP 等特殊 token id，采样到 EOS 立即结束；在 tokenizer 工厂与 Vocabulary 中统一其值。参考： `tokenizer_factory.cpp`
5. 5.
   增强诊断与保护（中优先）
   
   - 在内部 forward 的生成循环里：
     - 每步打印 top-5 概率与所选 token；
     - 断言 token_id < vocab_size；
     - 每 N 步校验 logits 是否出现 NaN/Inf；
     - 打印 KV 命中率、累计上下文长度；
     - 加入 UTF-8 校验，若解码结果非法则记录并考虑提前回退。
   - 参考增量解码接口： `text_generator.cpp`
6. 6.
   两条路径对比与默认后端选择（中优先）
   
   - 用同一 prompt/参数在 llama.cpp 与内部 forward 路径各跑一次，评价：
     - 质量（连贯性、中文字符占比、无乱码）
     - 速度（首 token 延迟、吞吐）
   - 若短期内内部 forward 权重映射工作量较大，可临时默认走 llama.cpp，保证用户体验。
补充说明

- 你前面关注的“模型 ID 规范化”并不是这次乱码的原因。两处规范化函数都保留了冒号与斜杠（不会把 : 或 / 改成 _），ID 一致，注册与查找正常。参考： `inference_engine.cpp` 和 `ollama_model_manager.cpp`
我可以直接着手：

- 修正 QwenTextModel 的 vocab 对齐与特殊 token 读取；
- 在 MLInferenceEngine 的内部 forward 路径完善“prefill+decode”的采样循环并加断言与诊断；
- 若遇到未完成的权重映射，加入自动回退到 llama.cpp 的保护。

--------------------------------------next-------------------------------------------
修复QwenTextModel权重映射：将GGUF加载的权重(token_embd.weight, output.weight, blk.*.attn_q.weight等)正确分配给QwenTextModel的成员变量(tokenEmbeddings_, outputWeights_, 各层attention和MLP权重)

实现SelfAttention::forward的实际计算：Q/K/V投影、scaled dot-product attention、RoPE位置编码，替换当前的占位符实现

实现FeedForward::forward的实际计算：gate/up/down投影和SwiGLU激活函数，替换当前的pass-through实现

修复applyPositionalEncoding方法，使用已预计算的RoPE频率(rope_freqs_)实现真正的旋转位置编码

验证并修复层归一化实现，确保使用正确的GGUF权重(blk.*.attn_norm.weight, blk.*.ffn_norm.weight)

添加详细的调试日志来验证tensor形状、数值范围和权重加载


## 核心问题诊断
### 1. EOS Token ID 不匹配
从模型元数据可以看出：

- 模型的真实EOS token ID : 151645
- 词汇表大小 : 152064
- 问题 : 代码中可能没有正确读取或使用这个EOS token ID，导致生成过程无法正常停止
### 2. Logits 计算异常
从日志分析发现：

- 生成了完整的512个token（达到max_tokens限制）
- 生成的token ID（54900, 30009, 20441等）看起来是随机的
- 这表明logits计算可能存在问题，导致采样结果异常
### 3. 推理流程混乱
在 `inference_engine.cpp` 中：

- 第一次使用多模态模型的forward方法
- 后续使用文本模型的nextToken方法
- 这种混合使用可能导致状态不一致
## 🛠️ 建议的修复方案
### 优先级1: 修复EOS Token检测
```
// 在inference_engine.cpp中添加正确的EOS检测
const int eos_token_id = 151645; // 从GGUF
元数据读取
if (next_token == eos_token_id) {
    break; // 正确停止生成
}
```
### 优先级2: 验证Logits计算
```
// 在computeLogitsFromHidden中添加调试信息
std::cout << "Logits range: [" << 
*std::min_element(logits.begin(), logits.
end()) 
          << ", " << *std::max_element
          (logits.begin(), logits.end()) 
          << "]" << std::endl;
```
### 优先级3: 统一推理流程
建议在整个生成过程中使用一致的推理方法，要么全部使用多模态模型的forward，要么全部使用文本模型的nextToken。

### 优先级4: 权重验证
在权重加载后添加数值范围检查，确保权重不是NaN或无穷大。

## 📊 从日志看到的具体问题
1. 1.
   输入正确 : 你好 被正确编码为1个token
2. 2.
   输出异常 : 生成的内容完全是乱码，包含各种语言的随机词汇
3. 3.
   长度异常 : 生成了完整的512个token，说明EOS检测失败
4. 4.
   Token ID异常 : 生成的token ID看起来是随机分布的
这些症状强烈表明logits计算或采样过程存在严重问题，很可能是权重加载不正确或EOS token配置错误导致的。

建议首先修复EOS token ID的读取和使用，然后验证权重加载的正确性。