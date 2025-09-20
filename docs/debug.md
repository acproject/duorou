下面是基于你给到的两边代码实际内容，按模块逐点对比 Ollama-main（Go）与 Duorou（C++）在“分词/词表”“模型/后端”“文件容器（GGML/GGUF）”等方面的封装差异与对应关系。各点都附上文件定位，便于你跳转对照。

一、整体架构与职责划分

- Go（Ollama-main）将“容器读取(fs/ggml) → 元信息(KV) → 模型选择(model) → 后端执行(ml) → 分词器(TextProcessor)”串为一体，且通过注册表进行架构级（architecture）分派：
  - 容器与元信息读取： `ggml.go`
  - 后端抽象与内存记账： `backend.go`
  - 模型注册/查找与TextProcessor工厂： `model.go`
  - 分词与词表： `textprocessor.go` `vocabulary.go` `bytepairencoding.go` `sentencepiece.go`
- C++（Duorou）将“设备后端(src/ml)”“容器包装(extensions/ollama/ggml, gguf)”“分词/词表(src/model)”分层拆分，尚未看到统一的“KV → 模型注册 → TextProcessor工厂”的一站式绑定：
  - 设备后端抽象与CPU实现： `backend.h` `cpu_backend.h` `cpu_backend.cpp`
  - GGUF/GGML 封装与KV访问辅助： `gguf_wrapper.h` `ggml_wrapper.cpp`
  - 分词与词表： `text_processor.h` `vocabulary.h` `vocabulary.cpp` `byte_pair_encoding.h` `byte_pair_encoding.cpp` `sentence_piece.h`
二、TextProcessor 抽象与工厂绑定

- Go 侧：
  - 抽象定义简洁且与 Vocabulary 强绑定：Encode/Decode/Is/Vocabulary() 见 `textprocessor.go` 。
  - TextProcessor 的选择与加载由模型架构驱动：NewTextProcessor 打开 GGML 文件，解出 KV，再走 getTextProcessor 按架构选择模型里实现的 TextProcessor，见 `model.go` 。
- C++ 侧：
  - 抽象接口更通用，除 Encode/Decode/isSpecial，还有 getVocabulary/getVocabSize，见 `text_processor.h` 。
  - 暂未看到“从GGUF/GGML KV自动派发到具体TextProcessor”的工厂路径；通常需显式构造 BytePairEncoding/SentencePiece。
三、Vocabulary（词表）封装

- Go 侧 Vocabulary 结构体与缓存使用 sync.Once，方法接口偏“面向分词”：
  - 字段 Values/Types/Scores/Merges 以及 BOS/EOS 与 AddBOS/AddEOS，见 `vocabulary.go` 。
  - Encode/Decode/SpecialVocabulary/Merge 的缓存（valuesOnce/specialOnce/mergeOnce）提升性能；addSpecials 会记录日志并避免重复添加。
  - 特殊 token 类型常量从 iota+1 起（NORMAL=1, UNKNOWN=2, CONTROL=3, USER_DEFINED=4, ...）定义在 `textprocessor.go` 。
- C++ 侧 Vocabulary 类与 once_flag 思路相同，但实现细节略有不同：
  - initialize/encode/decode/getSpecialVocabulary/getMergeRank/setBOS/setEOS 封装对应，见 `vocabulary.h` `vocabulary.cpp` 。
  - addSpecials 不做日志，仅在首/尾不重复时插入；isSpecial 仅支持 BOS/EOS。
  - token 类型常量定义值与 Go 不同（C++: NORMAL=0, CONTROL=1, USER_DEFINED=2），见 `vocabulary.h` 。注意：如果直接复用同一份模型/词表元信息，这个“类型值”的差异需要在解析时对齐，否则会影响 SpecialVocabulary 的筛选。
四、Byte-Pair Encoding（BPE）封装

- Go 侧：
  - BytePairEncoding 拥有正则预分词 pre 与 Vocabulary，见 `bytepairencoding.go` 。
  - 预分词使用 regexp2（支持更多语法），split 基于 FindStringMatch 迭代；编码流程：
    - 先切出 special token（按 Vocabulary.SpecialVocabulary 划分片段）
    - 对普通片段：按预分词 → byte→unicode 映射 → 若整体在词表则短路，否则构建 runes 与 merges，使用小根堆（rank 越小优先级越高）按 merges rank 合并，最终序列按 vocab.Encode 收集
  - Decode 进行 unicode→byte 的逆映射，细节与 byte→unicode 对称。
- C++ 侧：
  - BytePairEncoding 与 Go 逻辑一致，一一对应的方法与数据结构：Fragment/Pair/Merge/优先队列/字节与Unicode映射，见 `byte_pair_encoding.h` `byte_pair_encoding.cpp` 。
  - 差异点：
    - 预分词使用 std::regex 与 sregex_iterator，Go 使用 regexp2；某些复杂正则在 C++/Go 两侧语义可能有差异，需要在模式上对齐。
    - C++ applyBPE 将 processedText 按 UTF-8 “字节长度”切 rune 子串，而 Go 直接按 []rune（Unicode 码点）切分；对于非 BMP 或某些组合字符，边界一致性需额外验证。
    - C++ 解码实现包含自写 UTF-8 解码流程并做 unicode→byte 回转；Go 使用 rune 迭代与 WriteByte，逻辑更简。
五、SentencePiece（Unigram）封装

- Go 侧 SentencePieceModel 完整实现了 Unigram 合并（按 Scores 做候选优先级）及空白替换“▁”，并在无法匹配时回退至 <0x..> 字节 token，见 `sentencepiece.go` 。
- C++ 侧 `sentence_piece.h` 只给出了接口声明与若干私有方法（encodeText/normalize/viterbiDecode等）签名，未在当前项目中看到对应 .cpp 实现。对比结论：
  - 设计意图与 Go 类似（支持 Viterbi/打分/空白前缀等），但目前实现不完整；若要对齐 Ollama 的行为，需要补充 Unigram 的候选堆/打分与回退逻辑。
六、模型注册与 TextProcessor 选择

- Go 侧：
  - model.Register/ New/ getTextProcessor 以 KV.Architecture() 作为关键，统一注册与实例化模型与其 TextProcessor，见 `model.go` 。
  - 多模态模型实现需额外实现 MultimodalProcessor 接口，且 PostTokenize/EncodeMultimodal 等在公共 Runner 路径被调用。
- C++ 侧：
  - 没有在公共路径看到“按 architecture 自动选择模型/分词器”的注册工厂；多模态相关在具体模型类中（例如 Qwen 系列）实现，未见统一接口抽象（此仓库中我们已看到 QwenMultimodalModel 的声明，但未在本次对比中打开该具体文件）。
七、ML Backend 抽象与运行时内存/设备封装

- Go 侧 ml.Backend 是“模型执行后端”的抽象，职责包含：
  - 模型加载、Context 创建、Tensor 获取、KVCache 配置、显存/内存记账、日志化等，见 `backend.go` 。
  - BackendParams（线程数、GPULayers、FlashAttention）、BackendMemory/DeviceMemory/Memory 三元结构用于展示/统计按层、按设备的权重/缓存/计算图内存，且支持 Hash/日志输出等。
  - 还有 BackendCacheConfig 接口用于后端缓存优化（如 PermutedV、Padding、MaskDType/MaskBatchPadding 等）。
- C++ 侧 ml/backend 的 Backend 更接近“设备后端/内存管理器”的抽象：
  - 提供初始化/清理、设备选择、分配/拷贝、同步等基础接口，见 `backend.h` ；CPUBackend 做了线程数与对齐分配等实现，见 `cpu_backend.h` `cpu_backend.cpp` 。
  - 尚未看到与 Go 侧等价的“按层内存记账、KVCache 策略、图内存尺寸、跨 GPU 层划分”等运行时策略封装。
  - 因此 Go 的 ml.Backend 更“面向推理运行”的上层抽象；C++ 的 Backend 更“面向硬件设备”的底层抽象。
八、GGML/GGUF 容器与元信息（KV）访问

- Go 侧 `ggml.go` ：
  - 提供 KV 的强类型访问（String/Uint/Float/Bool、数组取 min/max）、架构判断、各种派生属性（EmbeddingLength、HeadCount*K/V、ContextLength、ChatTemplate 等），以及 Tensors 视图。
  - 该 KV 是模型初始化与 TextProcessor 工厂选择的关键输入。
- C++ 侧：
  - gguf_wrapper.h 提供了 Value/KeyValue/TensorInfo/BufferedReader/File 等封装，支持延迟加载/迭代器，便于读取 GGUF 文件，见 `gguf_wrapper.h` 。
  - ggml_wrapper.cpp 提供 KVHelper（与 Go 的 KV 方法集类似：architecture/kind/parameterCount/fileType/headCount*/contextLength/chatTemplate 等）与 FileType 解析，但目前 TensorInfo::numBytes 做了“每值4字节”的简化，类型分支与真实张量类型未完全对齐，见 `ggml_wrapper.cpp` 。
  - 整体思路与 Go 接近，但类型完备性与推理路径的集成度还有差距。
九、错误处理、并发与日志

- Go 侧广泛使用 error 返回值、context、slog，并在 BPE/SPM 中有“TODO: 并发处理”的注记；addSpecials 会对重复插入发出 warn 日志，见 `vocabulary.go` `bytepairencoding.go` `sentencepiece.go` 。
- C++ 侧目前更少显式错误通道（个别地方预留 GGUFError 类型），日志也较少；CPUBackend 支持设置线程数，但 TextProcessor 层暂未体现并发策略。
十、特殊 Token 与 BOS/EOS 策略

- 两边都支持 BOS/EOS 配置与自动添加；差异在于：
  - Go 会 warn 并避免重复；C++ 只做避免重复，不发日志。
  - Special 的枚举/常量定义不同（Go: SpecialBOS/SpecialEOS；C++: PAD/UNK/BOS/EOS），C++ 的 isSpecial 目前只处理 BOS/EOS，见 `text_processor.h` 。
十一、单元测试与覆盖

- Go 侧配有 bytepairencoding_test.go、sentencepiece_test.go、vocabulary_test.go、model_test.go 等，便于行为对齐与回归验证（路径已在你的输入中列出）。
- C++ 侧当前未见对应测试文件；若要对齐 Ollama 行为，建议补齐同等测试覆盖。
十二、结论与建议

- 行为等价程度：
  - BPE：两边高度一致，需注意预分词正则与 UTF-8/rune 切分细微差异可能带来的边界问题。建议用相同的测试集比对 encode/decode。
  - SentencePiece：Go 侧完整，C++ 侧接口声明但实现缺失。建议参考 Go 的 candidate/heap/score 回退流程补齐 .cpp。
  - Vocabulary：功能一致，但 token 类型值定义不同；确保“控制/用户自定义”类型筛选对齐，否则 SpecialVocabulary 结果会偏差。
  - GGML/GGUF：两边 KV 访问接口非常接近，但 C++ 的张量类型/字节数估算尚需完善，避免后续加载/布局错误。
  - Backend：Go 是推理运行时的“上层后端”，带有 KVCache/内存记账/多 GPU 层分配等策略；C++ 是“设备层后端”抽象。若你要在 C++ 侧实现与 Go Runner 等价的能力，需要在 C++ 层另建“模型运行后端”层，将 GGUF KV、模型架构、内存记账、KVCache 策略统一起来。
- 实施建议（可选）：
  - 在 C++ 侧新增“TextProcessor 工厂”，从 GGUF KV.architecture 选择 BPE/SPM，并支持按模型子类型定制（对齐 Go 的 `model.go` ）。
  - 完成 SentencePiece.cpp 实现，并按 Go 的测试用例对齐。
  - 统一 token 类型常量或在解析层做映射，保证 SpecialVocabulary 一致。
  - 完善 ggml_wrapper 的张量类型/字节数计算（参考 gguf_wrapper.h 的 getTensorTypeBytesPerValue 思路），避免以 4 字节为默认。
  - 若计划在 C++ 侧构建完整 Runner，补齐“按层内存记账、KVCache 策略（如 PermutedV/MaskPadding）、多 GPU 分层装载/哈希”等能力，参考 Go 的 `backend.go` 。
如需，我可以：

- 基于 Go 的 sentencepiece.go，补写 C++ 的 SentencePiece.cpp（Unigram/Viterbi/回退字节 token 全流程）。
- 写一个 C++ TextProcessorFactory：读取 GGUF KV，按 architecture 返回 BPE 或 SPM。
- 为 C++ BPE/SPM 增加与 Go 同步的测试样例，做 encode/decode 对拍，对齐差异点。