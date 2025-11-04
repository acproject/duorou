Builder with MCP

思考过程

我先扫描仓库里多模态相关实现与结构体。

在工作区搜索 'Find MLInferenceEngine class and any multimodal methods (e.g., generateTextWithImages, image embeddings path). Identify where llama_batch/ubatch are built and how images flow into text generation.'

Thought

我打开 ollama_model_manager.cpp 以确认请求与结构体定义。

src/extensions/ollama/ollama_model_manager.cpp 1 - 250

Thought

我打开头文件以查看 InferenceRequest 和 ModelInfo 定义。

src/extensions/ollama/ollama_model_manager.h 1 - 250

Thought

我打开 inference_engine.h 以确认当前文本路径和可扩展点。

src/extensions/ollama/inference_engine.h 1 - 199

Thought

我查看 parseModelInfo，确认 has_vision 等字段来源。

src/extensions/ollama/ollama_model_manager.cpp 512 - 585

在工作区搜索 'Find API to get model embedding dimension (e.g., llama_model_n_embd or similar) to validate image feature size when constructing batch.embd.'

在工作区搜索 'llama_.*n_embd'

Thought

在工作区搜索 'llama_kv_cache_clear'

在工作区搜索 'llama_.*kv.*cache'

Thought

在工作区搜索 'kv_.*cache.*clear|kv_.*cache.*seq|llama_kv_cache_'

我将把这条“多模态现状与集成计划”任务标记完成。

1/1 已完成

任务完成

总结多模态支持现状与后续集成计划

当前状态

- InferenceEngine / MLInferenceEngine 已有稳定文本路径：加载 gguf ，初始化 llama_model / llama_context ，提示词经 llama_batch_get_one 进入 llama_decode 逐步采样生成。
- OllamaModelManager::generateTextWithImages 目前仅做校验后返回 “未实现”。
- InferenceRequest 已包含图像通道： image_features 为 std::vector<std::vector<float>> ，表示按“每个特征向量=一个嵌入 token”的输入。
- 模型元信息通过 GGUFParser 解析： ModelInfo.has_vision 标记视觉能力，并用于管控选择文本或多模态路径。
关键发现

- llama_batch 支持两种输入模式： token （词 ID）或 embd （直接嵌入向量）。在 build_inp_embd 中，若提供 ubatch.embd 将直接作为 GGML_TYPE_F32 输入。
- llama_model_n_embd(model) 可获取模型隐层维度（例如 Qwen2.5-VL 为 3584），用来校验每个 image_features[i].size() 。
- 位置编码 pos 由 hparams.n_pos_per_embd() 控制维度：常规模型为 1，MROPE/IMROPE 为 4（Qwen3-VL 属于 iMROPE）。 llama_batch_allocr 可在缺省时自动生成 pos ，但对于图像补丁理想做法是显式构造多维 pos 。
- 具有编码器的多模态模型：先调用 llama_encode(ctx_, batch) 处理视觉嵌入，再以 decoder_start_token 或 BOS 启动 llama_decode 走文本解码路径。
集成方案

- 在 MLInferenceEngine 中新增多模态方法： generateTextWithImages(prompt, image_features, max_tokens, temperature, top_p) 。
  - 校验 isReady() 与 llama_model_n_embd(model_) 一致性。
  - 构造仅包含 embd 的 llama_batch （长度为图像嵌入数），避免设置 token ；先 llama_encode 注入视觉上下文。
  - 获取 decoder_start_token （ llama_model_decoder_start_token(model_) ，若为 LLAMA_TOKEN_NULL 使用 llama_vocab_bos ）。
  - 将 prompt 正常分词，先用“提示词 batch”跑一次 llama_decode 以“预热上下文”，随后进入逐步采样循环（沿用现有文本生成路径的采样器链）。
- 在 OllamaModelManager::generateTextWithImages 中：
  - 使用现有模型加载/校验逻辑。
  - 从 inference_engines_ 中取出引擎， dynamic_cast<MLInferenceEngine*> 并调用新增的 generateTextWithImages 。
  - 统计耗时等元数据并返回 InferenceResponse 。
位置编码处理

- 第一阶段（快速可用）：不显式提供 pos ，由 llama_batch_allocr 自动生成，保证基本流程可跑。
- 第二阶段（精确匹配视觉布局，Qwen3-VL 推荐）：为图像“补丁序列”构造 4D pos （如 [t, h, w, ds] ），与 VisionRotaryEmbedding / positionEmbedding 预期一致，提升对长图与多图的对齐效果。
里程碑与步骤

- 里程碑 1：实现基本多模态通路（视觉嵌入 + 文本解码）
  - 引擎方法新增并接线到 OllamaModelManager::generateTextWithImages
  - 校验 image_features 每向量长度与 llama_model_n_embd 一致
  - llama_encode 注入视觉嵌入， llama_decode 处理提示词与采样
- 里程碑 2：完善位置编码与多图支持
  - 显式构造 IMROPE 4D pos ，按补丁网格映射
  - 支持多图顺序与分隔策略（可用特殊图像分隔嵌入或 seq_id 分段）
- 里程碑 3：鲁棒性与性能
  - KV 管理与序列隔离（必要时使用不同 seq_id ）
  - 大图与批量请求的内存与吞吐优化（ n_batch /后端选择）


  额外说明

- 我将之前避免“完全收起”的最小宽度限制移除了，因为现在有按钮可以恢复，用户可自由拖到 0。
- 当前按钮文字是英文，如需中文显示/国际化可告知。
- 如果你更喜欢按钮放在右上角（靠近最小化/关闭按钮），我也可以改为另一种摆放方式（例如用 gtk_header_bar_pack_end 或改造 title_widget 布局）。
是否需要我给按钮加图标（例如“≡”或“⟷”）或快捷键，或者把“上次宽度”持久化到配置文件以便重启后也记忆？