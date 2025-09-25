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