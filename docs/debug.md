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

根因与修复

- 问题定位
  - 第二轮对话保存不全的主因是客户端只调用了一次 recv，未按 RESP 协议完整接收大响应（特别是 GET 返回的大 JSON）。
  - 服务端发送也只写了一次 write/send，存在内核短写导致的响应截断风险。
- 修复内容
  - 客户端 SessionStorageAdapter::receiveResponse 改为循环接收，按 RESP 解析判断完整性（支持简单字符串、错误、整数、bulk string、扁平数组），直到完整响应或超时，避免大 JSON 被截断。
  - 服务端 TcpServer::send_response 改为循环写入，处理 EINTR / EAGAIN ，避免短写截断响应。
- 直接收益
  - 会话 JSON、会话列表 JSON 等大体量值读取完整，第二轮及后续各轮对话数据都能完整持久化并可靠读取。
  - 在服务器忙于保存或返回大 key 时，读写更稳健。
新增能力

- 向量存取（别名更直观）
  - 新命令： VSET key f1 f2 ... , VGET key
  - 说明：复用已有 SETNX/GETNX 的数值向量存储能力，命名更符合“向量化”的语义。
- 元数据与冷热管理
  - 新命令：
    - METASET key field value （例如 hot , hot_score , tags ）
    - METAGET key [field] （不带 field 返回 RESP 数组形如 [field, value, ...] ）
    - TAGADD key tag1 tag2 ... （去重追加标签，保存在 __meta:key:tags ）
    - HOTSET key score （保存 hot_score ，并用阈值 >=5 置 hot=1 否则 0 ）
  - 实现策略：采用“前缀键”存储（如 __meta:<key>:<field> ），无须修改 MCDB/AOF 格式，即插即用并可持久化。
- 二进制对象化（视频/图片/音频）
  - 新命令：
    - OBJSET key mime payload （保存 __obj:<key>:data 和 __obj:<key>:mime ）
    - OBJGET key （返回 RESP 数组 [mime, data] ， data 为原始字节）
  - 说明：数据以原始字节持久化于 MCDB，支持任意二进制内容；元信息（ mime ）单独保存，便于对象化管理。
- 图抽象（多跳推理基础）
  - 新命令：
    - GRAPH.ADDEDGE from relation to （邻接表保存在 __graph:adj:<from> ，以 relation:to 逗号分隔）
    - GRAPH.NEIGHBORS node （返回相邻的 relation:to 列表）
  - 说明：实现简单图抽象与邻接查询，能表达“引用”、“依赖”、“指代”等关系，支持上层做多跳检索与推理。
如何使用

- 保存/读取会话（原有）
  - SET session_data:<id> <json> / GET session_data:<id>
- 向量化存储（例如保存对话片段的 embedding）
  - VSET vec:session:<id>:msg:<n> 0.12 0.98 ...
  - VGET vec:session:<id>:msg:<n> → *d\r\n$...\r\n... （d 为维度数）
- 元数据与冷热
  - 标记热点与分数
    - HOTSET session_data:<id> 9.7 （内部会保存 __meta:session_data:<id>:hot_score=9.7 与 :hot=1 ）
  - 打标签
    - TAGADD session_data:<id> persona important （去重追加标签）
  - 查询 meta
    - METAGET session_data:<id> → *6\r\n$3\r\nhot\r\n$1\r\n1\r\n$9\r\nhot_score\r\n$3\r\n9.7\r\n$4\r\ntags\r\n$17\r\npersona,important\r\n...
    - METAGET session_data:<id> tags → $17\r\npersona,important\r\n
- 二进制对象化
  - 存图（原始字节作为第三个参数的 bulk）
    - OBJSET img:123 image/png <PNG_RAW_BYTES> → +OK
  - 取图
    - OBJGET img:123 → *2\r\n$9\r\nimage/png\r\n$<n>\r\n<PNG_RAW_BYTES>\r\n
- 图抽象
  - 建边（“当前消息”引用“上一条消息”）
    - GRAPH.ADDEDGE msg:2 refers msg:1 → +OK
  - 查邻居
    - GRAPH.NEIGHBORS msg:2 → *1\r\n$10\r\nrefers:msg:1\r\n
建议的集成策略

- 会话保存即热化
  - 在你现有 add_message_to_current_session 逻辑保存后，同步调用 HOTSET session_data:<id> <score> （如以最近访问次数/时间衰减得分）。
  - 对用户问题/系统设定/重要工具输出打标签： TAGADD session_data:<id> persona tool_result .
- 向量化与检索
  - 由上层应用计算 embedding（本库提供向量存储/检索，不内置模型推理），用 VSET/VGET 管理每条消息对应向量。
  - 做近似检索可在客户端执行 KNN 或相似度排序；后续可扩展服务端 VSEARCH 。
- 对象化多模态
  - 对图片、音频、视频按对象存取， OBJSET/OBJGET 统一管理，结合 METASET/METAGET 记录时长/分辨率/采样率等附加字段。
- 图与多跳
  - 在保存消息时，结合引用/共指/工具调用形成关系边，后续对话做 GRAPH.NEIGHBORS 获取上下游节点，再结合 VGET 和标签筛选做多信号融合检索，提高回忆质量。
兼容性与持久化

- 未改动 MCDB/AOF 文件格式；新增的元数据、对象、图等均按普通键持久化（以 __meta: 、 __obj: 、 __graph: 前缀区分）。
- VSET/VGET 是对现有数值数组命令的直观别名；原 SETNX/GETNX 依旧可用。
- 客户端和服务端传输更稳健：客户端按 RESP 完整接收；服务端循环发送避免短写。
排查与验证建议

- 回归测试：连续保存 3 轮及以上对话， GET session_data:<id> 应返回完整 JSON，长度应随消息数增长。
- 大 key 测试：保存 50KB+ 的会话 JSON、或 200KB+ 的 OBJSET ，读取应无截断。
- 元数据/向量/图命令在同一数据库中混用，重启后应仍然存在（通过 METAGET 、 VGET 、 GRAPH.NEIGHBORS 验证）。
后续可迭代点（可选）

- 内置 VSEARCH （余弦/内积 topK）与 HNSW / IVF 索引支持，减少客户端负担。
- 图查询增强（按关系过滤、限定深度 BFS/DFS、多跳路径返回）。
- 热度动态化（访问计数/最近访问时间自动衰减），并按热度驱动冷热数据迁移策略。
- 对象化元信息规范化（哈希校验、尺寸/时长等标准字段约定）。
现在你可以直接重建并运行项目。现有使用 SessionStorageAdapter 的地方无需改动以修复 bug；如果要用到新增能力，在你现有网络命令发送层（或新建小工具）里直接发送上述命令即可。我也可以按你的调用场景，补充客户端封装方法（如 setVector/getVector 、 setObject/getObject 、 setHot/addTags/metaget 、 graphAddEdge/graphNeighbors ）来简化使用。



建议的集成策略

- 会话保存即热化
  - 在你现有 add_message_to_current_session 逻辑保存后，同步调用 HOTSET session_data:<id> <score> （如以最近访问次数/时间衰减得分）。
  - 对用户问题/系统设定/重要工具输出打标签： TAGADD session_data:<id> persona tool_result .
- 向量化与检索
  - 由上层应用计算 embedding（本库提供向量存储/检索，不内置模型推理），用 VSET/VGET 管理每条消息对应向量。
  - 做近似检索可在客户端执行 KNN 或相似度排序；后续可扩展服务端 VSEARCH 。
- 对象化多模态
  - 对图片、音频、视频按对象存取， OBJSET/OBJGET 统一管理，结合 METASET/METAGET 记录时长/分辨率/采样率等附加字段。
- 图与多跳
  - 在保存消息时，结合引用/共指/工具调用形成关系边，后续对话做 GRAPH.NEIGHBORS 获取上下游节点，再结合 VGET 和标签筛选做多信号融合检索，提高回忆质量

排查与验证建议

- 回归测试：连续保存 3 轮及以上对话， GET session_data:<id> 应返回完整 JSON，长度应随消息数增长。
- 大 key 测试：保存 50KB+ 的会话 JSON、或 200KB+ 的 OBJSET ，读取应无截断。
- 元数据/向量/图命令在同一数据库中混用，重启后应仍然存在（通过 METAGET 、 VGET 、 GRAPH.NEIGHBORS 验证）。

- 内置 VSEARCH （余弦/内积 topK）与 HNSW / IVF 索引支持，减少客户端负担。
- 图查询增强（按关系过滤、限定深度 BFS/DFS、多跳路径返回）。
- 热度动态化（访问计数/最近访问时间自动衰减），并按热度驱动冷热数据迁移策略。
- 对象化元信息规范化（哈希校验、尺寸/时长等标准字段约定）。
另外如何让冷热数据来辅助模型理解呢，是否需要在用户的提示词中增加一些系统的（通过上面的功能来实现）