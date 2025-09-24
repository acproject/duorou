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


{"content":"目前不是从 GGUF 里读取的真实 Qwen 词表与 merges，我希望实现Vocabulary + merges，并通过 createTextProcessorFromGGUF 创建 tokenizer，保证分词和词表与模型权重一致，优先使用 GGUF 的词表与 merges `/Users/acproject/workspace/cpp_projects/duorou/src/model`","multiMedia":[],"parsedQuery":["目前不是从 GGUF 里读取的真实 Qwen 词表与 merges，我希望实现Vocabulary + merges，并通过 createTextProcessorFromGGUF 创建 tokenizer，保证分词和词表与模型权重一致，优先使用 GGUF 的词表与 merges",{"folderPath":"/Users/acproject/workspace/cpp_projects/duorou/src/model","relatePath":"src/model","name":"model","type":"folder","title":"/Users/acproject/workspace/cpp_projects/duorou/src/model"}]}



```sh
[DEBUG] GGUFParser: Read tensor name: v.blk.11.attn_v.weight
[DEBUG] GGUFParser: Reading tensor 394/858
[DEBUG] GGUFParser: Read tensor name: v.blk.11.ffn_down.bias
[DEBUG] GGUFParser: Reading tensor 395/858
[DEBUG] GGUFParser: Read tensor name: v.blk.11.ffn_down.weight
[DEBUG] GGUFParser: Reading tensor 396/858
[DEBUG] GGUFParser: Read tensor name: v.blk.11.ffn_gate.bias
[DEBUG] GGUFParser: Reading tensor 397/858
[DEBUG] GGUFParser: Read tensor name: v.blk.11.ffn_gate.weight
[DEBUG] GGUFParser: Reading tensor 398/858
[DEBUG] GGUFParser: Read tensor name: v.blk.11.ffn_up.bias
[DEBUG] GGUFParser: Reading tensor 399/858
[DEBUG] GGUFParser: Read tensor name: v.blk.11.ffn_up.weight
[DEBUG] GGUFParser: Reading tensor 400/858
[DEBUG] GGUFParser: Read tensor name: v.blk.11.ln1.weight
[DEBUG] GGUFParser: Reading tensor 401/858
[DEBUG] GGUFParser: Read tensor name: v.blk.11.ln2.weight
[DEBUG] GGUFParser: Reading tensor 402/858
[DEBUG] GGUFParser: Read tensor name: v.blk.12.attn_out.bias
[DEBUG] GGUFParser: Reading tensor 403/858
[DEBUG] GGUFParser: Read tensor name: v.blk.12.attn_out.weight
[DEBUG] GGUFParser: Reading tensor 404/858
[DEBUG] GGUFParser: Read tensor name: v.blk.12.attn_q.bias
[DEBUG] GGUFParser: Reading tensor 405/858
[DEBUG] GGUFParser: Read tensor name: v.blk.12.attn_k.bias
[DEBUG] GGUFParser: Reading tensor 406/858
[DEBUG] GGUFParser: Read tensor name: v.blk.12.attn_v.bias
[DEBUG] GGUFParser: Reading tensor 407/858
[DEBUG] GGUFParser: Read tensor name: v.blk.12.attn_q.weight
[DEBUG] GGUFParser: Reading tensor 408/858
[DEBUG] GGUFParser: Read tensor name: v.blk.12.attn_k.weight
[DEBUG] GGUFParser: Reading tensor 409/858
[DEBUG] GGUFParser: Read tensor name: v.blk.12.attn_v.weight
[DEBUG] GGUFParser: Reading tensor 410/858
[DEBUG] GGUFParser: Read tensor name: v.blk.12.ffn_down.bias
[DEBUG] GGUFParser: Reading tensor 411/858
[DEBUG] GGUFParser: Read tensor name: v.blk.12.ffn_down.weight
[DEBUG] GGUFParser: Reading tensor 412/858
[DEBUG] GGUFParser: Read tensor name: v.blk.12.ffn_gate.bias
[DEBUG] GGUFParser: Reading tensor 413/858
[DEBUG] GGUFParser: Read tensor name: v.blk.12.ffn_gate.weight
[DEBUG] GGUFParser: Reading tensor 414/858
[DEBUG] GGUFParser: Read tensor name: v.blk.12.ffn_up.bias
[DEBUG] GGUFParser: Reading tensor 415/858
[DEBUG] GGUFParser: Read tensor name: v.blk.12.ffn_up.weight
[DEBUG] GGUFParser: Reading tensor 416/858
[DEBUG] GGUFParser: Read tensor name: v.blk.12.ln1.weight
[DEBUG] GGUFParser: Reading tensor 417/858
[DEBUG] GGUFParser: Read tensor name: v.blk.12.ln2.weight
[DEBUG] GGUFParser: Reading tensor 418/858
[DEBUG] GGUFParser: Read tensor name: v.blk.13.attn_out.bias
[DEBUG] GGUFParser: Reading tensor 419/858
[DEBUG] GGUFParser: Read tensor name: v.blk.13.attn_out.weight
[DEBUG] GGUFParser: Reading tensor 420/858
[DEBUG] GGUFParser: Read tensor name: v.blk.13.attn_q.bias
[DEBUG] GGUFParser: Reading tensor 421/858
[DEBUG] GGUFParser: Read tensor name: v.blk.13.attn_k.bias
[DEBUG] GGUFParser: Reading tensor 422/858
[DEBUG] GGUFParser: Read tensor name: v.blk.13.attn_v.bias
[DEBUG] GGUFParser: Reading tensor 423/858
[DEBUG] GGUFParser: Read tensor name: v.blk.13.attn_q.weight
[DEBUG] GGUFParser: Reading tensor 424/858
[DEBUG] GGUFParser: Read tensor name: v.blk.13.attn_k.weight
[DEBUG] GGUFParser: Reading tensor 425/858
[DEBUG] GGUFParser: Read tensor name: v.blk.13.attn_v.weight
[DEBUG] GGUFParser: Reading tensor 426/858
[DEBUG] GGUFParser: Read tensor name: v.blk.13.ffn_down.bias
[DEBUG] GGUFParser: Reading tensor 427/858
[DEBUG] GGUFParser: Read tensor name: v.blk.13.ffn_down.weight
[DEBUG] GGUFParser: Reading tensor 428/858
[DEBUG] GGUFParser: Read tensor name: v.blk.13.ffn_gate.bias
[DEBUG] GGUFParser: Reading tensor 429/858
[DEBUG] GGUFParser: Read tensor name: v.blk.13.ffn_gate.weight
[DEBUG] GGUFParser: Reading tensor 430/858
[DEBUG] GGUFParser: Read tensor name: v.blk.13.ffn_up.bias
[DEBUG] GGUFParser: Reading tensor 431/858
[DEBUG] GGUFParser: Read tensor name: v.blk.13.ffn_up.weight
[DEBUG] GGUFParser: Reading tensor 432/858
[DEBUG] GGUFParser: Read tensor name: v.blk.13.ln1.weight
[DEBUG] GGUFParser: Reading tensor 433/858
[DEBUG] GGUFParser: Read tensor name: v.blk.13.ln2.weight
[DEBUG] GGUFParser: Reading tensor 434/858
[DEBUG] GGUFParser: Read tensor name: v.blk.14.attn_out.bias
[DEBUG] GGUFParser: Reading tensor 435/858
[DEBUG] GGUFParser: Read tensor name: v.blk.14.attn_out.weight
[DEBUG] GGUFParser: Reading tensor 436/858
[DEBUG] GGUFParser: Read tensor name: v.blk.14.attn_q.bias
[DEBUG] GGUFParser: Reading tensor 437/858
[DEBUG] GGUFParser: Read tensor name: v.blk.14.attn_k.bias
[DEBUG] GGUFParser: Reading tensor 438/858
[DEBUG] GGUFParser: Read tensor name: v.blk.14.attn_v.bias
[DEBUG] GGUFParser: Reading tensor 439/858
[DEBUG] GGUFParser: Read tensor name: v.blk.14.attn_q.weight
[DEBUG] GGUFParser: Reading tensor 440/858
[DEBUG] GGUFParser: Read tensor name: v.blk.14.attn_k.weight
[DEBUG] GGUFParser: Reading tensor 441/858
[DEBUG] GGUFParser: Read tensor name: v.blk.14.attn_v.weight
[DEBUG] GGUFParser: Reading tensor 442/858
[DEBUG] GGUFParser: Read tensor name: v.blk.14.ffn_down.bias
[DEBUG] GGUFParser: Reading tensor 443/858
[DEBUG] GGUFParser: Read tensor name: v.blk.14.ffn_down.weight
[DEBUG] GGUFParser: Reading tensor 444/858
[DEBUG] GGUFParser: Read tensor name: v.blk.14.ffn_gate.bias
[DEBUG] GGUFParser: Reading tensor 445/858
[DEBUG] GGUFParser: Read tensor name: v.blk.14.ffn_gate.weight
[DEBUG] GGUFParser: Reading tensor 446/858
[DEBUG] GGUFParser: Read tensor name: v.blk.14.ffn_up.bias
[DEBUG] GGUFParser: Reading tensor 447/858
[DEBUG] GGUFParser: Read tensor name: v.blk.14.ffn_up.weight
[DEBUG] GGUFParser: Reading tensor 448/858
[DEBUG] GGUFParser: Read tensor name: v.blk.14.ln1.weight
[DEBUG] GGUFParser: Reading tensor 449/858
[DEBUG] GGUFParser: Read tensor name: v.blk.14.ln2.weight
[DEBUG] GGUFParser: Reading tensor 450/858
[DEBUG] GGUFParser: Read tensor name: v.blk.15.attn_out.bias
[DEBUG] GGUFParser: Reading tensor 451/858
[DEBUG] GGUFParser: Read tensor name: v.blk.15.attn_out.weight
[DEBUG] GGUFParser: Reading tensor 452/858
[DEBUG] GGUFParser: Read tensor name: v.blk.15.attn_q.bias
[DEBUG] GGUFParser: Reading tensor 453/858
[DEBUG] GGUFParser: Read tensor name: v.blk.15.attn_k.bias
[DEBUG] GGUFParser: Reading tensor 454/858
[DEBUG] GGUFParser: Read tensor name: v.blk.15.attn_v.bias
[DEBUG] GGUFParser: Reading tensor 455/858
[DEBUG] GGUFParser: Read tensor name: v.blk.15.attn_q.weight
[DEBUG] GGUFParser: Reading tensor 456/858
[DEBUG] GGUFParser: Read tensor name: v.blk.15.attn_k.weight
[DEBUG] GGUFParser: Reading tensor 457/858
[DEBUG] GGUFParser: Read tensor name: v.blk.15.attn_v.weight
[DEBUG] GGUFParser: Reading tensor 458/858
[DEBUG] GGUFParser: Read tensor name: v.blk.15.ffn_down.bias
[DEBUG] GGUFParser: Reading tensor 459/858
[DEBUG] GGUFParser: Read tensor name: v.blk.15.ffn_down.weight
[DEBUG] GGUFParser: Reading tensor 460/858
[DEBUG] GGUFParser: Read tensor name: v.blk.15.ffn_gate.bias
[DEBUG] GGUFParser: Reading tensor 461/858
[DEBUG] GGUFParser: Read tensor name: v.blk.15.ffn_gate.weight
[DEBUG] GGUFParser: Reading tensor 462/858
[DEBUG] GGUFParser: Read tensor name: v.blk.15.ffn_up.bias
[DEBUG] GGUFParser: Reading tensor 463/858
[DEBUG] GGUFParser: Read tensor name: v.blk.15.ffn_up.weight
[DEBUG] GGUFParser: Reading tensor 464/858
[DEBUG] GGUFParser: Read tensor name: v.blk.15.ln1.weight
[DEBUG] GGUFParser: Reading tensor 465/858
[DEBUG] GGUFParser: Read tensor name: v.blk.15.ln2.weight
[DEBUG] GGUFParser: Reading tensor 466/858
[DEBUG] GGUFParser: Read tensor name: v.blk.16.attn_out.bias
[DEBUG] GGUFParser: Reading tensor 467/858
[DEBUG] GGUFParser: Read tensor name: v.blk.16.attn_out.weight
[DEBUG] GGUFParser: Reading tensor 468/858
[DEBUG] GGUFParser: Read tensor name: v.blk.16.attn_q.bias
[DEBUG] GGUFParser: Reading tensor 469/858
[DEBUG] GGUFParser: Read tensor name: v.blk.16.attn_k.bias
[DEBUG] GGUFParser: Reading tensor 470/858
[DEBUG] GGUFParser: Read tensor name: v.blk.16.attn_v.bias
[DEBUG] GGUFParser: Reading tensor 471/858
[DEBUG] GGUFParser: Read tensor name: v.blk.16.attn_q.weight
[DEBUG] GGUFParser: Reading tensor 472/858
[DEBUG] GGUFParser: Read tensor name: v.blk.16.attn_k.weight
[DEBUG] GGUFParser: Reading tensor 473/858
[DEBUG] GGUFParser: Read tensor name: v.blk.16.attn_v.weight
[DEBUG] GGUFParser: Reading tensor 474/858
[DEBUG] GGUFParser: Read tensor name: v.blk.16.ffn_down.bias
[DEBUG] GGUFParser: Reading tensor 475/858
[DEBUG] GGUFParser: Read tensor name: v.blk.16.ffn_down.weight
[DEBUG] GGUFParser: Reading tensor 476/858
[DEBUG] GGUFParser: Read tensor name: v.blk.16.ffn_gate.bias
[DEBUG] GGUFParser: Reading tensor 477/858
[DEBUG] GGUFParser: Read tensor name: v.blk.16.ffn_gate.weight
[DEBUG] GGUFParser: Reading tensor 478/858
[DEBUG] GGUFParser: Read tensor name: v.blk.16.ffn_up.bias
[DEBUG] GGUFParser: Reading tensor 479/858
[DEBUG] GGUFParser: Read tensor name: v.blk.16.ffn_up.weight
[DEBUG] GGUFParser: Reading tensor 480/858
[DEBUG] GGUFParser: Read tensor name: v.blk.16.ln1.weight
[DEBUG] GGUFParser: Reading tensor 481/858
[DEBUG] GGUFParser: Read tensor name: v.blk.16.ln2.weight
[DEBUG] GGUFParser: Reading tensor 482/858
[DEBUG] GGUFParser: Read tensor name: v.blk.17.attn_out.bias
[DEBUG] GGUFParser: Reading tensor 483/858
[DEBUG] GGUFParser: Read tensor name: v.blk.17.attn_out.weight
[DEBUG] GGUFParser: Reading tensor 484/858
[DEBUG] GGUFParser: Read tensor name: v.blk.17.attn_q.bias
[DEBUG] GGUFParser: Reading tensor 485/858
[DEBUG] GGUFParser: Read tensor name: v.blk.17.attn_k.bias
[DEBUG] GGUFParser: Reading tensor 486/858
[DEBUG] GGUFParser: Read tensor name: v.blk.17.attn_v.bias
[DEBUG] GGUFParser: Reading tensor 487/858
[DEBUG] GGUFParser: Read tensor name: v.blk.17.attn_q.weight
[DEBUG] GGUFParser: Reading tensor 488/858
[DEBUG] GGUFParser: Read tensor name: v.blk.17.attn_k.weight
[DEBUG] GGUFParser: Reading tensor 489/858
[DEBUG] GGUFParser: Read tensor name: v.blk.17.attn_v.weight
[DEBUG] GGUFParser: Reading tensor 490/858
[DEBUG] GGUFParser: Read tensor name: v.blk.17.ffn_down.bias
[DEBUG] GGUFParser: Reading tensor 491/858
[DEBUG] GGUFParser: Read tensor name: v.blk.17.ffn_down.weight
[DEBUG] GGUFParser: Reading tensor 492/858
[DEBUG] GGUFParser: Read tensor name: v.blk.17.ffn_gate.bias
[DEBUG] GGUFParser: Reading tensor 493/858
[DEBUG] GGUFParser: Read tensor name: v.blk.17.ffn_gate.weight
[DEBUG] GGUFParser: Reading tensor 494/858
[DEBUG] GGUFParser: Read tensor name: v.blk.17.ffn_up.bias
[DEBUG] GGUFParser: Reading tensor 495/858
[DEBUG] GGUFParser: Read tensor name: v.blk.17.ffn_up.weight
[DEBUG] GGUFParser: Reading tensor 496/858
[DEBUG] GGUFParser: Read tensor name: v.blk.17.ln1.weight
[DEBUG] GGUFParser: Reading tensor 497/858
[DEBUG] GGUFParser: Read tensor name: v.blk.17.ln2.weight
[DEBUG] GGUFParser: Reading tensor 498/858
[DEBUG] GGUFParser: Read tensor name: v.blk.18.attn_out.bias
[DEBUG] GGUFParser: Reading tensor 499/858
[DEBUG] GGUFParser: Read tensor name: v.blk.18.attn_out.weight
[DEBUG] GGUFParser: Reading tensor 500/858
[DEBUG] GGUFParser: Read tensor name: v.blk.18.attn_q.bias
[DEBUG] GGUFParser: Reading tensor 501/858
[DEBUG] GGUFParser: Read tensor name: v.blk.18.attn_k.bias
[DEBUG] GGUFParser: Reading tensor 502/858
[DEBUG] GGUFParser: Read tensor name: v.blk.18.attn_v.bias
[DEBUG] GGUFParser: Reading tensor 503/858
[DEBUG] GGUFParser: Read tensor name: v.blk.18.attn_q.weight
[DEBUG] GGUFParser: Reading tensor 504/858
[DEBUG] GGUFParser: Read tensor name: v.blk.18.attn_k.weight
[DEBUG] GGUFParser: Reading tensor 505/858
[DEBUG] GGUFParser: Read tensor name: v.blk.18.attn_v.weight
[DEBUG] GGUFParser: Reading tensor 506/858
[DEBUG] GGUFParser: Read tensor name: v.blk.18.ffn_down.bias
[DEBUG] GGUFParser: Reading tensor 507/858
[DEBUG] GGUFParser: Read tensor name: v.blk.18.ffn_down.weight
[DEBUG] GGUFParser: Reading tensor 508/858
[DEBUG] GGUFParser: Read tensor name: v.blk.18.ffn_gate.bias
[DEBUG] GGUFParser: Reading tensor 509/858
[DEBUG] GGUFParser: Read tensor name: v.blk.18.ffn_gate.weight
[DEBUG] GGUFParser: Reading tensor 510/858
[DEBUG] GGUFParser: Read tensor name: v.blk.18.ffn_up.bias
[DEBUG] GGUFParser: Reading tensor 511/858
[DEBUG] GGUFParser: Read tensor name: v.blk.18.ffn_up.weight
[DEBUG] GGUFParser: Reading tensor 512/858
[DEBUG] GGUFParser: Read tensor name: v.blk.18.ln1.weight
[DEBUG] GGUFParser: Reading tensor 513/858
[DEBUG] GGUFParser: Read tensor name: v.blk.18.ln2.weight
[DEBUG] GGUFParser: Reading tensor 514/858
[DEBUG] GGUFParser: Read tensor name: v.blk.19.attn_out.bias
[DEBUG] GGUFParser: Reading tensor 515/858
[DEBUG] GGUFParser: Read tensor name: v.blk.19.attn_out.weight
[DEBUG] GGUFParser: Reading tensor 516/858
[DEBUG] GGUFParser: Read tensor name: v.blk.19.attn_q.bias
[DEBUG] GGUFParser: Reading tensor 517/858
[DEBUG] GGUFParser: Read tensor name: v.blk.19.attn_k.bias
[DEBUG] GGUFParser: Reading tensor 518/858
[DEBUG] GGUFParser: Read tensor name: v.blk.19.attn_v.bias
[DEBUG] GGUFParser: Reading tensor 519/858
[DEBUG] GGUFParser: Read tensor name: v.blk.19.attn_q.weight
[DEBUG] GGUFParser: Reading tensor 520/858
[DEBUG] GGUFParser: Read tensor name: v.blk.19.attn_k.weight
[DEBUG] GGUFParser: Reading tensor 521/858
[DEBUG] GGUFParser: Read tensor name: v.blk.19.attn_v.weight
[DEBUG] GGUFParser: Reading tensor 522/858
[DEBUG] GGUFParser: Read tensor name: v.blk.19.ffn_down.bias
[DEBUG] GGUFParser: Reading tensor 523/858
[DEBUG] GGUFParser: Read tensor name: v.blk.19.ffn_down.weight
[DEBUG] GGUFParser: Reading tensor 524/858
[DEBUG] GGUFParser: Read tensor name: v.blk.19.ffn_gate.bias
[DEBUG] GGUFParser: Reading tensor 525/858
[DEBUG] GGUFParser: Read tensor name: v.blk.19.ffn_gate.weight
[DEBUG] GGUFParser: Reading tensor 526/858
[DEBUG] GGUFParser: Read tensor name: v.blk.19.ffn_up.bias
[DEBUG] GGUFParser: Reading tensor 527/858
[DEBUG] GGUFParser: Read tensor name: v.blk.19.ffn_up.weight
[DEBUG] GGUFParser: Reading tensor 528/858
[DEBUG] GGUFParser: Read tensor name: v.blk.19.ln1.weight
[DEBUG] GGUFParser: Reading tensor 529/858
[DEBUG] GGUFParser: Read tensor name: v.blk.19.ln2.weight
[DEBUG] GGUFParser: Reading tensor 530/858
[DEBUG] GGUFParser: Read tensor name: v.blk.2.attn_out.bias
[DEBUG] GGUFParser: Reading tensor 531/858
[DEBUG] GGUFParser: Read tensor name: v.blk.2.attn_out.weight
[DEBUG] GGUFParser: Reading tensor 532/858
[DEBUG] GGUFParser: Read tensor name: v.blk.2.attn_q.bias
[DEBUG] GGUFParser: Reading tensor 533/858
[DEBUG] GGUFParser: Read tensor name: v.blk.2.attn_k.bias
[DEBUG] GGUFParser: Reading tensor 534/858
[DEBUG] GGUFParser: Read tensor name: v.blk.2.attn_v.bias
[DEBUG] GGUFParser: Reading tensor 535/858
[DEBUG] GGUFParser: Read tensor name: v.blk.2.attn_q.weight
[DEBUG] GGUFParser: Reading tensor 536/858
[DEBUG] GGUFParser: Read tensor name: v.blk.2.attn_k.weight
[DEBUG] GGUFParser: Reading tensor 537/858
[DEBUG] GGUFParser: Read tensor name: v.blk.2.attn_v.weight
[DEBUG] GGUFParser: Reading tensor 538/858
[DEBUG] GGUFParser: Read tensor name: v.blk.2.ffn_down.bias
[DEBUG] GGUFParser: Reading tensor 539/858
[DEBUG] GGUFParser: Read tensor name: v.blk.2.ffn_down.weight
[DEBUG] GGUFParser: Reading tensor 540/858
[DEBUG] GGUFParser: Read tensor name: v.blk.2.ffn_gate.bias
[DEBUG] GGUFParser: Reading tensor 541/858
[DEBUG] GGUFParser: Read tensor name: v.blk.2.ffn_gate.weight
[DEBUG] GGUFParser: Reading tensor 542/858
[DEBUG] GGUFParser: Read tensor name: v.blk.2.ffn_up.bias
[DEBUG] GGUFParser: Reading tensor 543/858
[DEBUG] GGUFParser: Read tensor name: v.blk.2.ffn_up.weight
[DEBUG] GGUFParser: Reading tensor 544/858
[DEBUG] GGUFParser: Read tensor name: v.blk.2.ln1.weight
[DEBUG] GGUFParser: Reading tensor 545/858
[DEBUG] GGUFParser: Read tensor name: v.blk.2.ln2.weight
[DEBUG] GGUFParser: Reading tensor 546/858
[DEBUG] GGUFParser: Read tensor name: v.blk.20.attn_out.bias
[DEBUG] GGUFParser: Reading tensor 547/858
[DEBUG] GGUFParser: Read tensor name: v.blk.20.attn_out.weight
[DEBUG] GGUFParser: Reading tensor 548/858
[DEBUG] GGUFParser: Read tensor name: v.blk.20.attn_q.bias
[DEBUG] GGUFParser: Reading tensor 549/858
[DEBUG] GGUFParser: Read tensor name: v.blk.20.attn_k.bias
[DEBUG] GGUFParser: Reading tensor 550/858
[DEBUG] GGUFParser: Read tensor name: v.blk.20.attn_v.bias
[DEBUG] GGUFParser: Reading tensor 551/858
[DEBUG] GGUFParser: Read tensor name: v.blk.20.attn_q.weight
[DEBUG] GGUFParser: Reading tensor 552/858
[DEBUG] GGUFParser: Read tensor name: v.blk.20.attn_k.weight
[DEBUG] GGUFParser: Reading tensor 553/858
[DEBUG] GGUFParser: Read tensor name: v.blk.20.attn_v.weight
[DEBUG] GGUFParser: Reading tensor 554/858
[DEBUG] GGUFParser: Read tensor name: v.blk.20.ffn_down.bias
[DEBUG] GGUFParser: Reading tensor 555/858
[DEBUG] GGUFParser: Read tensor name: v.blk.20.ffn_down.weight
[DEBUG] GGUFParser: Reading tensor 556/858
[DEBUG] GGUFParser: Read tensor name: v.blk.20.ffn_gate.bias
[DEBUG] GGUFParser: Reading tensor 557/858
[DEBUG] GGUFParser: Read tensor name: v.blk.20.ffn_gate.weight
[DEBUG] GGUFParser: Reading tensor 558/858
[DEBUG] GGUFParser: Read tensor name: v.blk.20.ffn_up.bias
[DEBUG] GGUFParser: Reading tensor 559/858
[DEBUG] GGUFParser: Read tensor name: v.blk.20.ffn_up.weight
[DEBUG] GGUFParser: Reading tensor 560/858
[DEBUG] GGUFParser: Read tensor name: v.blk.20.ln1.weight
[DEBUG] GGUFParser: Reading tensor 561/858
[DEBUG] GGUFParser: Read tensor name: v.blk.20.ln2.weight
[DEBUG] GGUFParser: Reading tensor 562/858
[DEBUG] GGUFParser: Read tensor name: v.blk.21.attn_out.bias
[DEBUG] GGUFParser: Reading tensor 563/858
[DEBUG] GGUFParser: Read tensor name: v.blk.21.attn_out.weight
[DEBUG] GGUFParser: Reading tensor 564/858
[DEBUG] GGUFParser: Read tensor name: v.blk.21.attn_q.bias
[DEBUG] GGUFParser: Reading tensor 565/858
[DEBUG] GGUFParser: Read tensor name: v.blk.21.attn_k.bias
[DEBUG] GGUFParser: Reading tensor 566/858
[DEBUG] GGUFParser: Read tensor name: v.blk.21.attn_v.bias
[DEBUG] GGUFParser: Reading tensor 567/858
[DEBUG] GGUFParser: Read tensor name: v.blk.21.attn_q.weight
[DEBUG] GGUFParser: Reading tensor 568/858
[DEBUG] GGUFParser: Read tensor name: v.blk.21.attn_k.weight
[DEBUG] GGUFParser: Reading tensor 569/858
[DEBUG] GGUFParser: Read tensor name: v.blk.21.attn_v.weight
[DEBUG] GGUFParser: Reading tensor 570/858
[DEBUG] GGUFParser: Read tensor name: v.blk.21.ffn_down.bias
[DEBUG] GGUFParser: Reading tensor 571/858
[DEBUG] GGUFParser: Read tensor name: v.blk.21.ffn_down.weight
[DEBUG] GGUFParser: Reading tensor 572/858
[DEBUG] GGUFParser: Read tensor name: v.blk.21.ffn_gate.bias
[DEBUG] GGUFParser: Reading tensor 573/858
[DEBUG] GGUFParser: Read tensor name: v.blk.21.ffn_gate.weight
[DEBUG] GGUFParser: Reading tensor 574/858
[DEBUG] GGUFParser: Read tensor name: v.blk.21.ffn_up.bias
[DEBUG] GGUFParser: Reading tensor 575/858
[DEBUG] GGUFParser: Read tensor name: v.blk.21.ffn_up.weight
[DEBUG] GGUFParser: Reading tensor 576/858
[DEBUG] GGUFParser: Read tensor name: v.blk.21.ln1.weight
[DEBUG] GGUFParser: Reading tensor 577/858
[DEBUG] GGUFParser: Read tensor name: v.blk.21.ln2.weight
[DEBUG] GGUFParser: Reading tensor 578/858
[DEBUG] GGUFParser: Read tensor name: v.blk.22.attn_out.bias
[DEBUG] GGUFParser: Reading tensor 579/858
[DEBUG] GGUFParser: Read tensor name: v.blk.22.attn_out.weight
[DEBUG] GGUFParser: Reading tensor 580/858
[DEBUG] GGUFParser: Read tensor name: v.blk.22.attn_q.bias
[DEBUG] GGUFParser: Reading tensor 581/858
[DEBUG] GGUFParser: Read tensor name: v.blk.22.attn_k.bias
[DEBUG] GGUFParser: Reading tensor 582/858
[DEBUG] GGUFParser: Read tensor name: v.blk.22.attn_v.bias
[DEBUG] GGUFParser: Reading tensor 583/858
[DEBUG] GGUFParser: Read tensor name: v.blk.22.attn_q.weight
[DEBUG] GGUFParser: Reading tensor 584/858
[DEBUG] GGUFParser: Read tensor name: v.blk.22.attn_k.weight
[DEBUG] GGUFParser: Reading tensor 585/858
[DEBUG] GGUFParser: Read tensor name: v.blk.22.attn_v.weight
[DEBUG] GGUFParser: Reading tensor 586/858
[DEBUG] GGUFParser: Read tensor name: v.blk.22.ffn_down.bias
[DEBUG] GGUFParser: Reading tensor 587/858
[DEBUG] GGUFParser: Read tensor name: v.blk.22.ffn_down.weight
[DEBUG] GGUFParser: Reading tensor 588/858
[DEBUG] GGUFParser: Read tensor name: v.blk.22.ffn_gate.bias
[DEBUG] GGUFParser: Reading tensor 589/858
[DEBUG] GGUFParser: Read tensor name: v.blk.22.ffn_gate.weight
[DEBUG] GGUFParser: Reading tensor 590/858
[DEBUG] GGUFParser: Read tensor name: v.blk.22.ffn_up.bias
[DEBUG] GGUFParser: Reading tensor 591/858
[DEBUG] GGUFParser: Read tensor name: v.blk.22.ffn_up.weight
[DEBUG] GGUFParser: Reading tensor 592/858
[DEBUG] GGUFParser: Read tensor name: v.blk.22.ln1.weight
[DEBUG] GGUFParser: Reading tensor 593/858
[DEBUG] GGUFParser: Read tensor name: v.blk.22.ln2.weight
[DEBUG] GGUFParser: Reading tensor 594/858
[DEBUG] GGUFParser: Read tensor name: v.blk.23.attn_out.bias
[DEBUG] GGUFParser: Reading tensor 595/858
[DEBUG] GGUFParser: Read tensor name: v.blk.23.attn_out.weight
[DEBUG] GGUFParser: Reading tensor 596/858
[DEBUG] GGUFParser: Read tensor name: v.blk.23.attn_q.bias
[DEBUG] GGUFParser: Reading tensor 597/858
[DEBUG] GGUFParser: Read tensor name: v.blk.23.attn_k.bias
[DEBUG] GGUFParser: Reading tensor 598/858
[DEBUG] GGUFParser: Read tensor name: v.blk.23.attn_v.bias
[DEBUG] GGUFParser: Reading tensor 599/858
[DEBUG] GGUFParser: Read tensor name: v.blk.23.attn_q.weight
[DEBUG] GGUFParser: Reading tensor 600/858
[DEBUG] GGUFParser: Read tensor name: v.blk.23.attn_k.weight
[DEBUG] GGUFParser: Reading tensor 601/858
[DEBUG] GGUFParser: Read tensor name: v.blk.23.attn_v.weight
[DEBUG] GGUFParser: Reading tensor 602/858
[DEBUG] GGUFParser: Read tensor name: v.blk.23.ffn_down.bias
[DEBUG] GGUFParser: Reading tensor 603/858
[DEBUG] GGUFParser: Read tensor name: v.blk.23.ffn_down.weight
[DEBUG] GGUFParser: Reading tensor 604/858
[DEBUG] GGUFParser: Read tensor name: v.blk.23.ffn_gate.bias
[DEBUG] GGUFParser: Reading tensor 605/858
[DEBUG] GGUFParser: Read tensor name: v.blk.23.ffn_gate.weight
[DEBUG] GGUFParser: Reading tensor 606/858
[DEBUG] GGUFParser: Read tensor name: v.blk.23.ffn_up.bias
[DEBUG] GGUFParser: Reading tensor 607/858
[DEBUG] GGUFParser: Read tensor name: v.blk.23.ffn_up.weight
[DEBUG] GGUFParser: Reading tensor 608/858
[DEBUG] GGUFParser: Read tensor name: v.blk.23.ln1.weight
[DEBUG] GGUFParser: Reading tensor 609/858
[DEBUG] GGUFParser: Read tensor name: v.blk.23.ln2.weight
[DEBUG] GGUFParser: Reading tensor 610/858
[DEBUG] GGUFParser: Read tensor name: v.blk.24.attn_out.bias
[DEBUG] GGUFParser: Reading tensor 611/858
[DEBUG] GGUFParser: Read tensor name: v.blk.24.attn_out.weight
[DEBUG] GGUFParser: Reading tensor 612/858
[DEBUG] GGUFParser: Read tensor name: v.blk.24.attn_q.bias
[DEBUG] GGUFParser: Reading tensor 613/858
[DEBUG] GGUFParser: Read tensor name: v.blk.24.attn_k.bias
[DEBUG] GGUFParser: Reading tensor 614/858
[DEBUG] GGUFParser: Read tensor name: v.blk.24.attn_v.bias
[DEBUG] GGUFParser: Reading tensor 615/858
[DEBUG] GGUFParser: Read tensor name: v.blk.24.attn_q.weight
[DEBUG] GGUFParser: Reading tensor 616/858
[DEBUG] GGUFParser: Read tensor name: v.blk.24.attn_k.weight
[DEBUG] GGUFParser: Reading tensor 617/858
[DEBUG] GGUFParser: Read tensor name: v.blk.24.attn_v.weight
[DEBUG] GGUFParser: Reading tensor 618/858
[DEBUG] GGUFParser: Read tensor name: v.blk.24.ffn_down.bias
[DEBUG] GGUFParser: Reading tensor 619/858
[DEBUG] GGUFParser: Read tensor name: v.blk.24.ffn_down.weight
[DEBUG] GGUFParser: Reading tensor 620/858
[DEBUG] GGUFParser: Read tensor name: v.blk.24.ffn_gate.bias
[DEBUG] GGUFParser: Reading tensor 621/858
[DEBUG] GGUFParser: Read tensor name: v.blk.24.ffn_gate.weight
[DEBUG] GGUFParser: Reading tensor 622/858
[DEBUG] GGUFParser: Read tensor name: v.blk.24.ffn_up.bias
[DEBUG] GGUFParser: Reading tensor 623/858
[DEBUG] GGUFParser: Read tensor name: v.blk.24.ffn_up.weight
[DEBUG] GGUFParser: Reading tensor 624/858
[DEBUG] GGUFParser: Read tensor name: v.blk.24.ln1.weight
[DEBUG] GGUFParser: Reading tensor 625/858
[DEBUG] GGUFParser: Read tensor name: v.blk.24.ln2.weight
[DEBUG] GGUFParser: Reading tensor 626/858
[DEBUG] GGUFParser: Read tensor name: v.blk.25.attn_out.bias
[DEBUG] GGUFParser: Reading tensor 627/858
[DEBUG] GGUFParser: Read tensor name: v.blk.25.attn_out.weight
[DEBUG] GGUFParser: Reading tensor 628/858
[DEBUG] GGUFParser: Read tensor name: v.blk.25.attn_q.bias
[DEBUG] GGUFParser: Reading tensor 629/858
[DEBUG] GGUFParser: Read tensor name: v.blk.25.attn_k.bias
[DEBUG] GGUFParser: Reading tensor 630/858
[DEBUG] GGUFParser: Read tensor name: v.blk.25.attn_v.bias
[DEBUG] GGUFParser: Reading tensor 631/858
[DEBUG] GGUFParser: Read tensor name: v.blk.25.attn_q.weight
[DEBUG] GGUFParser: Reading tensor 632/858
[DEBUG] GGUFParser: Read tensor name: v.blk.25.attn_k.weight
[DEBUG] GGUFParser: Reading tensor 633/858
[DEBUG] GGUFParser: Read tensor name: v.blk.25.attn_v.weight
[DEBUG] GGUFParser: Reading tensor 634/858
[DEBUG] GGUFParser: Read tensor name: v.blk.25.ffn_down.bias
[DEBUG] GGUFParser: Reading tensor 635/858
[DEBUG] GGUFParser: Read tensor name: v.blk.25.ffn_down.weight
[DEBUG] GGUFParser: Reading tensor 636/858
[DEBUG] GGUFParser: Read tensor name: v.blk.25.ffn_gate.bias
[DEBUG] GGUFParser: Reading tensor 637/858
[DEBUG] GGUFParser: Read tensor name: v.blk.25.ffn_gate.weight
[DEBUG] GGUFParser: Reading tensor 638/858
[DEBUG] GGUFParser: Read tensor name: v.blk.25.ffn_up.bias
[DEBUG] GGUFParser: Reading tensor 639/858
[DEBUG] GGUFParser: Read tensor name: v.blk.25.ffn_up.weight
[DEBUG] GGUFParser: Reading tensor 640/858
[DEBUG] GGUFParser: Read tensor name: v.blk.25.ln1.weight
[DEBUG] GGUFParser: Reading tensor 641/858
[DEBUG] GGUFParser: Read tensor name: v.blk.25.ln2.weight
[DEBUG] GGUFParser: Reading tensor 642/858
[DEBUG] GGUFParser: Read tensor name: v.blk.26.attn_out.bias
[DEBUG] GGUFParser: Reading tensor 643/858
[DEBUG] GGUFParser: Read tensor name: v.blk.26.attn_out.weight
[DEBUG] GGUFParser: Reading tensor 644/858
[DEBUG] GGUFParser: Read tensor name: v.blk.26.attn_q.bias
[DEBUG] GGUFParser: Reading tensor 645/858
[DEBUG] GGUFParser: Read tensor name: v.blk.26.attn_k.bias
[DEBUG] GGUFParser: Reading tensor 646/858
[DEBUG] GGUFParser: Read tensor name: v.blk.26.attn_v.bias
[DEBUG] GGUFParser: Reading tensor 647/858
[DEBUG] GGUFParser: Read tensor name: v.blk.26.attn_q.weight
[DEBUG] GGUFParser: Reading tensor 648/858
[DEBUG] GGUFParser: Read tensor name: v.blk.26.attn_k.weight
[DEBUG] GGUFParser: Reading tensor 649/858
[DEBUG] GGUFParser: Read tensor name: v.blk.26.attn_v.weight
[DEBUG] GGUFParser: Reading tensor 650/858
[DEBUG] GGUFParser: Read tensor name: v.blk.26.ffn_down.bias
[DEBUG] GGUFParser: Reading tensor 651/858
[DEBUG] GGUFParser: Read tensor name: v.blk.26.ffn_down.weight
[DEBUG] GGUFParser: Reading tensor 652/858
[DEBUG] GGUFParser: Read tensor name: v.blk.26.ffn_gate.bias
[DEBUG] GGUFParser: Reading tensor 653/858
[DEBUG] GGUFParser: Read tensor name: v.blk.26.ffn_gate.weight
[DEBUG] GGUFParser: Reading tensor 654/858
[DEBUG] GGUFParser: Read tensor name: v.blk.26.ffn_up.bias
[DEBUG] GGUFParser: Reading tensor 655/858
[DEBUG] GGUFParser: Read tensor name: v.blk.26.ffn_up.weight
[DEBUG] GGUFParser: Reading tensor 656/858
[DEBUG] GGUFParser: Read tensor name: v.blk.26.ln1.weight
[DEBUG] GGUFParser: Reading tensor 657/858
[DEBUG] GGUFParser: Read tensor name: v.blk.26.ln2.weight
[DEBUG] GGUFParser: Reading tensor 658/858
[DEBUG] GGUFParser: Read tensor name: v.blk.27.attn_out.bias
[DEBUG] GGUFParser: Reading tensor 659/858
[DEBUG] GGUFParser: Read tensor name: v.blk.27.attn_out.weight
[DEBUG] GGUFParser: Reading tensor 660/858
[DEBUG] GGUFParser: Read tensor name: v.blk.27.attn_q.bias
[DEBUG] GGUFParser: Reading tensor 661/858
[DEBUG] GGUFParser: Read tensor name: v.blk.27.attn_k.bias
[DEBUG] GGUFParser: Reading tensor 662/858
[DEBUG] GGUFParser: Read tensor name: v.blk.27.attn_v.bias
[DEBUG] GGUFParser: Reading tensor 663/858
[DEBUG] GGUFParser: Read tensor name: v.blk.27.attn_q.weight
[DEBUG] GGUFParser: Reading tensor 664/858
[DEBUG] GGUFParser: Read tensor name: v.blk.27.attn_k.weight
[DEBUG] GGUFParser: Reading tensor 665/858
[DEBUG] GGUFParser: Read tensor name: v.blk.27.attn_v.weight
[DEBUG] GGUFParser: Reading tensor 666/858
[DEBUG] GGUFParser: Read tensor name: v.blk.27.ffn_down.bias
[DEBUG] GGUFParser: Reading tensor 667/858
[DEBUG] GGUFParser: Read tensor name: v.blk.27.ffn_down.weight
[DEBUG] GGUFParser: Reading tensor 668/858
[DEBUG] GGUFParser: Read tensor name: v.blk.27.ffn_gate.bias
[DEBUG] GGUFParser: Reading tensor 669/858
[DEBUG] GGUFParser: Read tensor name: v.blk.27.ffn_gate.weight
[DEBUG] GGUFParser: Reading tensor 670/858
[DEBUG] GGUFParser: Read tensor name: v.blk.27.ffn_up.bias
[DEBUG] GGUFParser: Reading tensor 671/858
[DEBUG] GGUFParser: Read tensor name: v.blk.27.ffn_up.weight
[DEBUG] GGUFParser: Reading tensor 672/858
[DEBUG] GGUFParser: Read tensor name: v.blk.27.ln1.weight
[DEBUG] GGUFParser: Reading tensor 673/858
[DEBUG] GGUFParser: Read tensor name: v.blk.27.ln2.weight
[DEBUG] GGUFParser: Reading tensor 674/858
[DEBUG] GGUFParser: Read tensor name: v.blk.28.attn_out.bias
[DEBUG] GGUFParser: Reading tensor 675/858
[DEBUG] GGUFParser: Read tensor name: v.blk.28.attn_out.weight
[DEBUG] GGUFParser: Reading tensor 676/858
[DEBUG] GGUFParser: Read tensor name: v.blk.28.attn_q.bias
[DEBUG] GGUFParser: Reading tensor 677/858
[DEBUG] GGUFParser: Read tensor name: v.blk.28.attn_k.bias
[DEBUG] GGUFParser: Reading tensor 678/858
[DEBUG] GGUFParser: Read tensor name: v.blk.28.attn_v.bias
[DEBUG] GGUFParser: Reading tensor 679/858
[DEBUG] GGUFParser: Read tensor name: v.blk.28.attn_q.weight
[DEBUG] GGUFParser: Reading tensor 680/858
[DEBUG] GGUFParser: Read tensor name: v.blk.28.attn_k.weight
[DEBUG] GGUFParser: Reading tensor 681/858
[DEBUG] GGUFParser: Read tensor name: v.blk.28.attn_v.weight
[DEBUG] GGUFParser: Reading tensor 682/858
[DEBUG] GGUFParser: Read tensor name: v.blk.28.ffn_down.bias
[DEBUG] GGUFParser: Reading tensor 683/858
[DEBUG] GGUFParser: Read tensor name: v.blk.28.ffn_down.weight
[DEBUG] GGUFParser: Reading tensor 684/858
[DEBUG] GGUFParser: Read tensor name: v.blk.28.ffn_gate.bias
[DEBUG] GGUFParser: Reading tensor 685/858
[DEBUG] GGUFParser: Read tensor name: v.blk.28.ffn_gate.weight
[DEBUG] GGUFParser: Reading tensor 686/858
[DEBUG] GGUFParser: Read tensor name: v.blk.28.ffn_up.bias
[DEBUG] GGUFParser: Reading tensor 687/858
[DEBUG] GGUFParser: Read tensor name: v.blk.28.ffn_up.weight
[DEBUG] GGUFParser: Reading tensor 688/858
[DEBUG] GGUFParser: Read tensor name: v.blk.28.ln1.weight
[DEBUG] GGUFParser: Reading tensor 689/858
[DEBUG] GGUFParser: Read tensor name: v.blk.28.ln2.weight
[DEBUG] GGUFParser: Reading tensor 690/858
[DEBUG] GGUFParser: Read tensor name: v.blk.29.attn_out.bias
[DEBUG] GGUFParser: Reading tensor 691/858
[DEBUG] GGUFParser: Read tensor name: v.blk.29.attn_out.weight
[DEBUG] GGUFParser: Reading tensor 692/858
[DEBUG] GGUFParser: Read tensor name: v.blk.29.attn_q.bias
[DEBUG] GGUFParser: Reading tensor 693/858
[DEBUG] GGUFParser: Read tensor name: v.blk.29.attn_k.bias
[DEBUG] GGUFParser: Reading tensor 694/858
[DEBUG] GGUFParser: Read tensor name: v.blk.29.attn_v.bias
[DEBUG] GGUFParser: Reading tensor 695/858
[DEBUG] GGUFParser: Read tensor name: v.blk.29.attn_q.weight
[DEBUG] GGUFParser: Reading tensor 696/858
[DEBUG] GGUFParser: Read tensor name: v.blk.29.attn_k.weight
[DEBUG] GGUFParser: Reading tensor 697/858
[DEBUG] GGUFParser: Read tensor name: v.blk.29.attn_v.weight
[DEBUG] GGUFParser: Reading tensor 698/858
[DEBUG] GGUFParser: Read tensor name: v.blk.29.ffn_down.bias
[DEBUG] GGUFParser: Reading tensor 699/858
[DEBUG] GGUFParser: Read tensor name: v.blk.29.ffn_down.weight
[DEBUG] GGUFParser: Reading tensor 700/858
[DEBUG] GGUFParser: Read tensor name: v.blk.29.ffn_gate.bias
[DEBUG] GGUFParser: Reading tensor 701/858
[DEBUG] GGUFParser: Read tensor name: v.blk.29.ffn_gate.weight
[DEBUG] GGUFParser: Reading tensor 702/858
[DEBUG] GGUFParser: Read tensor name: v.blk.29.ffn_up.bias
[DEBUG] GGUFParser: Reading tensor 703/858
[DEBUG] GGUFParser: Read tensor name: v.blk.29.ffn_up.weight
[DEBUG] GGUFParser: Reading tensor 704/858
[DEBUG] GGUFParser: Read tensor name: v.blk.29.ln1.weight
[DEBUG] GGUFParser: Reading tensor 705/858
[DEBUG] GGUFParser: Read tensor name: v.blk.29.ln2.weight
[DEBUG] GGUFParser: Reading tensor 706/858
[DEBUG] GGUFParser: Read tensor name: v.blk.3.attn_out.bias
[DEBUG] GGUFParser: Reading tensor 707/858
[DEBUG] GGUFParser: Read tensor name: v.blk.3.attn_out.weight
[DEBUG] GGUFParser: Reading tensor 708/858
[DEBUG] GGUFParser: Read tensor name: v.blk.3.attn_q.bias
[DEBUG] GGUFParser: Reading tensor 709/858
[DEBUG] GGUFParser: Read tensor name: v.blk.3.attn_k.bias
[DEBUG] GGUFParser: Reading tensor 710/858
[DEBUG] GGUFParser: Read tensor name: v.blk.3.attn_v.bias
[DEBUG] GGUFParser: Reading tensor 711/858
[DEBUG] GGUFParser: Read tensor name: v.blk.3.attn_q.weight
[DEBUG] GGUFParser: Reading tensor 712/858
[DEBUG] GGUFParser: Read tensor name: v.blk.3.attn_k.weight
[DEBUG] GGUFParser: Reading tensor 713/858
[DEBUG] GGUFParser: Read tensor name: v.blk.3.attn_v.weight
[DEBUG] GGUFParser: Reading tensor 714/858
[DEBUG] GGUFParser: Read tensor name: v.blk.3.ffn_down.bias
[DEBUG] GGUFParser: Reading tensor 715/858
[DEBUG] GGUFParser: Read tensor name: v.blk.3.ffn_down.weight
[DEBUG] GGUFParser: Reading tensor 716/858
[DEBUG] GGUFParser: Read tensor name: v.blk.3.ffn_gate.bias
[DEBUG] GGUFParser: Reading tensor 717/858
[DEBUG] GGUFParser: Read tensor name: v.blk.3.ffn_gate.weight
[DEBUG] GGUFParser: Reading tensor 718/858
[DEBUG] GGUFParser: Read tensor name: v.blk.3.ffn_up.bias
[DEBUG] GGUFParser: Reading tensor 719/858
[DEBUG] GGUFParser: Read tensor name: v.blk.3.ffn_up.weight
[DEBUG] GGUFParser: Reading tensor 720/858
[DEBUG] GGUFParser: Read tensor name: v.blk.3.ln1.weight
[DEBUG] GGUFParser: Reading tensor 721/858
[DEBUG] GGUFParser: Read tensor name: v.blk.3.ln2.weight
[DEBUG] GGUFParser: Reading tensor 722/858
[DEBUG] GGUFParser: Read tensor name: v.blk.30.attn_out.bias
[DEBUG] GGUFParser: Reading tensor 723/858
[DEBUG] GGUFParser: Read tensor name: v.blk.30.attn_out.weight
[DEBUG] GGUFParser: Reading tensor 724/858
[DEBUG] GGUFParser: Read tensor name: v.blk.30.attn_q.bias
[DEBUG] GGUFParser: Reading tensor 725/858
[DEBUG] GGUFParser: Read tensor name: v.blk.30.attn_k.bias
[DEBUG] GGUFParser: Reading tensor 726/858
[DEBUG] GGUFParser: Read tensor name: v.blk.30.attn_v.bias
[DEBUG] GGUFParser: Reading tensor 727/858
[DEBUG] GGUFParser: Read tensor name: v.blk.30.attn_q.weight
[DEBUG] GGUFParser: Reading tensor 728/858
[DEBUG] GGUFParser: Read tensor name: v.blk.30.attn_k.weight
[DEBUG] GGUFParser: Reading tensor 729/858
[DEBUG] GGUFParser: Read tensor name: v.blk.30.attn_v.weight
[DEBUG] GGUFParser: Reading tensor 730/858
[DEBUG] GGUFParser: Read tensor name: v.blk.30.ffn_down.bias
[DEBUG] GGUFParser: Reading tensor 731/858
[DEBUG] GGUFParser: Read tensor name: v.blk.30.ffn_down.weight
[DEBUG] GGUFParser: Reading tensor 732/858
[DEBUG] GGUFParser: Read tensor name: v.blk.30.ffn_gate.bias
[DEBUG] GGUFParser: Reading tensor 733/858
[DEBUG] GGUFParser: Read tensor name: v.blk.30.ffn_gate.weight
[DEBUG] GGUFParser: Reading tensor 734/858
[DEBUG] GGUFParser: Read tensor name: v.blk.30.ffn_up.bias
[DEBUG] GGUFParser: Reading tensor 735/858
[DEBUG] GGUFParser: Read tensor name: v.blk.30.ffn_up.weight
[DEBUG] GGUFParser: Reading tensor 736/858
[DEBUG] GGUFParser: Read tensor name: v.blk.30.ln1.weight
[DEBUG] GGUFParser: Reading tensor 737/858
[DEBUG] GGUFParser: Read tensor name: v.blk.30.ln2.weight
[DEBUG] GGUFParser: Reading tensor 738/858
[DEBUG] GGUFParser: Read tensor name: v.blk.31.attn_out.bias
[DEBUG] GGUFParser: Reading tensor 739/858
[DEBUG] GGUFParser: Read tensor name: v.blk.31.attn_out.weight
[DEBUG] GGUFParser: Reading tensor 740/858
[DEBUG] GGUFParser: Read tensor name: v.blk.31.attn_q.bias
[DEBUG] GGUFParser: Reading tensor 741/858
[DEBUG] GGUFParser: Read tensor name: v.blk.31.attn_k.bias
[DEBUG] GGUFParser: Reading tensor 742/858
[DEBUG] GGUFParser: Read tensor name: v.blk.31.attn_v.bias
[DEBUG] GGUFParser: Reading tensor 743/858
[DEBUG] GGUFParser: Read tensor name: v.blk.31.attn_q.weight
[DEBUG] GGUFParser: Reading tensor 744/858
[DEBUG] GGUFParser: Read tensor name: v.blk.31.attn_k.weight
[DEBUG] GGUFParser: Reading tensor 745/858
[DEBUG] GGUFParser: Read tensor name: v.blk.31.attn_v.weight
[DEBUG] GGUFParser: Reading tensor 746/858
[DEBUG] GGUFParser: Read tensor name: v.blk.31.ffn_down.bias
[DEBUG] GGUFParser: Reading tensor 747/858
[DEBUG] GGUFParser: Read tensor name: v.blk.31.ffn_down.weight
[DEBUG] GGUFParser: Reading tensor 748/858
[DEBUG] GGUFParser: Read tensor name: v.blk.31.ffn_gate.bias
[DEBUG] GGUFParser: Reading tensor 749/858
[DEBUG] GGUFParser: Read tensor name: v.blk.31.ffn_gate.weight
[DEBUG] GGUFParser: Reading tensor 750/858
[DEBUG] GGUFParser: Read tensor name: v.blk.31.ffn_up.bias
[DEBUG] GGUFParser: Reading tensor 751/858
[DEBUG] GGUFParser: Read tensor name: v.blk.31.ffn_up.weight
[DEBUG] GGUFParser: Reading tensor 752/858
[DEBUG] GGUFParser: Read tensor name: v.blk.31.ln1.weight
[DEBUG] GGUFParser: Reading tensor 753/858
[DEBUG] GGUFParser: Read tensor name: v.blk.31.ln2.weight
[DEBUG] GGUFParser: Reading tensor 754/858
[DEBUG] GGUFParser: Read tensor name: v.blk.4.attn_out.bias
[DEBUG] GGUFParser: Reading tensor 755/858
[DEBUG] GGUFParser: Read tensor name: v.blk.4.attn_out.weight
[DEBUG] GGUFParser: Reading tensor 756/858
[DEBUG] GGUFParser: Read tensor name: v.blk.4.attn_q.bias
[DEBUG] GGUFParser: Reading tensor 757/858
[DEBUG] GGUFParser: Read tensor name: v.blk.4.attn_k.bias
[DEBUG] GGUFParser: Reading tensor 758/858
[DEBUG] GGUFParser: Read tensor name: v.blk.4.attn_v.bias
[DEBUG] GGUFParser: Reading tensor 759/858
[DEBUG] GGUFParser: Read tensor name: v.blk.4.attn_q.weight
[DEBUG] GGUFParser: Reading tensor 760/858
[DEBUG] GGUFParser: Read tensor name: v.blk.4.attn_k.weight
[DEBUG] GGUFParser: Reading tensor 761/858
[DEBUG] GGUFParser: Read tensor name: v.blk.4.attn_v.weight
[DEBUG] GGUFParser: Reading tensor 762/858
[DEBUG] GGUFParser: Read tensor name: v.blk.4.ffn_down.bias
[DEBUG] GGUFParser: Reading tensor 763/858
[DEBUG] GGUFParser: Read tensor name: v.blk.4.ffn_down.weight
[DEBUG] GGUFParser: Reading tensor 764/858
[DEBUG] GGUFParser: Read tensor name: v.blk.4.ffn_gate.bias
[DEBUG] GGUFParser: Reading tensor 765/858
[DEBUG] GGUFParser: Read tensor name: v.blk.4.ffn_gate.weight
[DEBUG] GGUFParser: Reading tensor 766/858
[DEBUG] GGUFParser: Read tensor name: v.blk.4.ffn_up.bias
[DEBUG] GGUFParser: Reading tensor 767/858
[DEBUG] GGUFParser: Read tensor name: v.blk.4.ffn_up.weight
[DEBUG] GGUFParser: Reading tensor 768/858
[DEBUG] GGUFParser: Read tensor name: v.blk.4.ln1.weight
[DEBUG] GGUFParser: Reading tensor 769/858
[DEBUG] GGUFParser: Read tensor name: v.blk.4.ln2.weight
[DEBUG] GGUFParser: Reading tensor 770/858
[DEBUG] GGUFParser: Read tensor name: v.blk.5.attn_out.bias
[DEBUG] GGUFParser: Reading tensor 771/858
[DEBUG] GGUFParser: Read tensor name: v.blk.5.attn_out.weight
[DEBUG] GGUFParser: Reading tensor 772/858
[DEBUG] GGUFParser: Read tensor name: v.blk.5.attn_q.bias
[DEBUG] GGUFParser: Reading tensor 773/858
[DEBUG] GGUFParser: Read tensor name: v.blk.5.attn_k.bias
[DEBUG] GGUFParser: Reading tensor 774/858
[DEBUG] GGUFParser: Read tensor name: v.blk.5.attn_v.bias
[DEBUG] GGUFParser: Reading tensor 775/858
[DEBUG] GGUFParser: Read tensor name: v.blk.5.attn_q.weight
[DEBUG] GGUFParser: Reading tensor 776/858
[DEBUG] GGUFParser: Read tensor name: v.blk.5.attn_k.weight
[DEBUG] GGUFParser: Reading tensor 777/858
[DEBUG] GGUFParser: Read tensor name: v.blk.5.attn_v.weight
[DEBUG] GGUFParser: Reading tensor 778/858
[DEBUG] GGUFParser: Read tensor name: v.blk.5.ffn_down.bias
[DEBUG] GGUFParser: Reading tensor 779/858
[DEBUG] GGUFParser: Read tensor name: v.blk.5.ffn_down.weight
[DEBUG] GGUFParser: Reading tensor 780/858
[DEBUG] GGUFParser: Read tensor name: v.blk.5.ffn_gate.bias
[DEBUG] GGUFParser: Reading tensor 781/858
[DEBUG] GGUFParser: Read tensor name: v.blk.5.ffn_gate.weight
[DEBUG] GGUFParser: Reading tensor 782/858
[DEBUG] GGUFParser: Read tensor name: v.blk.5.ffn_up.bias
[DEBUG] GGUFParser: Reading tensor 783/858
[DEBUG] GGUFParser: Read tensor name: v.blk.5.ffn_up.weight
[DEBUG] GGUFParser: Reading tensor 784/858
[DEBUG] GGUFParser: Read tensor name: v.blk.5.ln1.weight
[DEBUG] GGUFParser: Reading tensor 785/858
[DEBUG] GGUFParser: Read tensor name: v.blk.5.ln2.weight
[DEBUG] GGUFParser: Reading tensor 786/858
[DEBUG] GGUFParser: Read tensor name: v.blk.6.attn_out.bias
[DEBUG] GGUFParser: Reading tensor 787/858
[DEBUG] GGUFParser: Read tensor name: v.blk.6.attn_out.weight
[DEBUG] GGUFParser: Reading tensor 788/858
[DEBUG] GGUFParser: Read tensor name: v.blk.6.attn_q.bias
[DEBUG] GGUFParser: Reading tensor 789/858
[DEBUG] GGUFParser: Read tensor name: v.blk.6.attn_k.bias
[DEBUG] GGUFParser: Reading tensor 790/858
[DEBUG] GGUFParser: Read tensor name: v.blk.6.attn_v.bias
[DEBUG] GGUFParser: Reading tensor 791/858
[DEBUG] GGUFParser: Read tensor name: v.blk.6.attn_q.weight
[DEBUG] GGUFParser: Reading tensor 792/858
[DEBUG] GGUFParser: Read tensor name: v.blk.6.attn_k.weight
[DEBUG] GGUFParser: Reading tensor 793/858
[DEBUG] GGUFParser: Read tensor name: v.blk.6.attn_v.weight
[DEBUG] GGUFParser: Reading tensor 794/858
[DEBUG] GGUFParser: Read tensor name: v.blk.6.ffn_down.bias
[DEBUG] GGUFParser: Reading tensor 795/858
[DEBUG] GGUFParser: Read tensor name: v.blk.6.ffn_down.weight
[DEBUG] GGUFParser: Reading tensor 796/858
[DEBUG] GGUFParser: Read tensor name: v.blk.6.ffn_gate.bias
[DEBUG] GGUFParser: Reading tensor 797/858
[DEBUG] GGUFParser: Read tensor name: v.blk.6.ffn_gate.weight
[DEBUG] GGUFParser: Reading tensor 798/858
[DEBUG] GGUFParser: Read tensor name: v.blk.6.ffn_up.bias
[DEBUG] GGUFParser: Reading tensor 799/858
[DEBUG] GGUFParser: Read tensor name: v.blk.6.ffn_up.weight
[DEBUG] GGUFParser: Reading tensor 800/858
[DEBUG] GGUFParser: Read tensor name: v.blk.6.ln1.weight
[DEBUG] GGUFParser: Reading tensor 801/858
[DEBUG] GGUFParser: Read tensor name: v.blk.6.ln2.weight
[DEBUG] GGUFParser: Reading tensor 802/858
[DEBUG] GGUFParser: Read tensor name: v.blk.7.attn_out.bias
[DEBUG] GGUFParser: Reading tensor 803/858
[DEBUG] GGUFParser: Read tensor name: v.blk.7.attn_out.weight
[DEBUG] GGUFParser: Reading tensor 804/858
[DEBUG] GGUFParser: Read tensor name: v.blk.7.attn_q.bias
[DEBUG] GGUFParser: Reading tensor 805/858
[DEBUG] GGUFParser: Read tensor name: v.blk.7.attn_k.bias
[DEBUG] GGUFParser: Reading tensor 806/858
[DEBUG] GGUFParser: Read tensor name: v.blk.7.attn_v.bias
[DEBUG] GGUFParser: Reading tensor 807/858
[DEBUG] GGUFParser: Read tensor name: v.blk.7.attn_q.weight
[DEBUG] GGUFParser: Reading tensor 808/858
[DEBUG] GGUFParser: Read tensor name: v.blk.7.attn_k.weight
[DEBUG] GGUFParser: Reading tensor 809/858
[DEBUG] GGUFParser: Read tensor name: v.blk.7.attn_v.weight
[DEBUG] GGUFParser: Reading tensor 810/858
[DEBUG] GGUFParser: Read tensor name: v.blk.7.ffn_down.bias
[DEBUG] GGUFParser: Reading tensor 811/858
[DEBUG] GGUFParser: Read tensor name: v.blk.7.ffn_down.weight
[DEBUG] GGUFParser: Reading tensor 812/858
[DEBUG] GGUFParser: Read tensor name: v.blk.7.ffn_gate.bias
[DEBUG] GGUFParser: Reading tensor 813/858
[DEBUG] GGUFParser: Read tensor name: v.blk.7.ffn_gate.weight
[DEBUG] GGUFParser: Reading tensor 814/858
[DEBUG] GGUFParser: Read tensor name: v.blk.7.ffn_up.bias
[DEBUG] GGUFParser: Reading tensor 815/858
[DEBUG] GGUFParser: Read tensor name: v.blk.7.ffn_up.weight
[DEBUG] GGUFParser: Reading tensor 816/858
[DEBUG] GGUFParser: Read tensor name: v.blk.7.ln1.weight
[DEBUG] GGUFParser: Reading tensor 817/858
[DEBUG] GGUFParser: Read tensor name: v.blk.7.ln2.weight
[DEBUG] GGUFParser: Reading tensor 818/858
[DEBUG] GGUFParser: Read tensor name: v.blk.8.attn_out.bias
[DEBUG] GGUFParser: Reading tensor 819/858
[DEBUG] GGUFParser: Read tensor name: v.blk.8.attn_out.weight
[DEBUG] GGUFParser: Reading tensor 820/858
[DEBUG] GGUFParser: Read tensor name: v.blk.8.attn_q.bias
[DEBUG] GGUFParser: Reading tensor 821/858
[DEBUG] GGUFParser: Read tensor name: v.blk.8.attn_k.bias
[DEBUG] GGUFParser: Reading tensor 822/858
[DEBUG] GGUFParser: Read tensor name: v.blk.8.attn_v.bias
[DEBUG] GGUFParser: Reading tensor 823/858
[DEBUG] GGUFParser: Read tensor name: v.blk.8.attn_q.weight
[DEBUG] GGUFParser: Reading tensor 824/858
[DEBUG] GGUFParser: Read tensor name: v.blk.8.attn_k.weight
[DEBUG] GGUFParser: Reading tensor 825/858
[DEBUG] GGUFParser: Read tensor name: v.blk.8.attn_v.weight
[DEBUG] GGUFParser: Reading tensor 826/858
[DEBUG] GGUFParser: Read tensor name: v.blk.8.ffn_down.bias
[DEBUG] GGUFParser: Reading tensor 827/858
[DEBUG] GGUFParser: Read tensor name: v.blk.8.ffn_down.weight
[DEBUG] GGUFParser: Reading tensor 828/858
[DEBUG] GGUFParser: Read tensor name: v.blk.8.ffn_gate.bias
[DEBUG] GGUFParser: Reading tensor 829/858
[DEBUG] GGUFParser: Read tensor name: v.blk.8.ffn_gate.weight
[DEBUG] GGUFParser: Reading tensor 830/858
[DEBUG] GGUFParser: Read tensor name: v.blk.8.ffn_up.bias
[DEBUG] GGUFParser: Reading tensor 831/858
[DEBUG] GGUFParser: Read tensor name: v.blk.8.ffn_up.weight
[DEBUG] GGUFParser: Reading tensor 832/858
[DEBUG] GGUFParser: Read tensor name: v.blk.8.ln1.weight
[DEBUG] GGUFParser: Reading tensor 833/858
[DEBUG] GGUFParser: Read tensor name: v.blk.8.ln2.weight
[DEBUG] GGUFParser: Reading tensor 834/858
[DEBUG] GGUFParser: Read tensor name: v.blk.9.attn_out.bias
[DEBUG] GGUFParser: Reading tensor 835/858
[DEBUG] GGUFParser: Read tensor name: v.blk.9.attn_out.weight
[DEBUG] GGUFParser: Reading tensor 836/858
[DEBUG] GGUFParser: Read tensor name: v.blk.9.attn_q.bias
[DEBUG] GGUFParser: Reading tensor 837/858
[DEBUG] GGUFParser: Read tensor name: v.blk.9.attn_k.bias
[DEBUG] GGUFParser: Reading tensor 838/858
[DEBUG] GGUFParser: Read tensor name: v.blk.9.attn_v.bias
[DEBUG] GGUFParser: Reading tensor 839/858
[DEBUG] GGUFParser: Read tensor name: v.blk.9.attn_q.weight
[DEBUG] GGUFParser: Reading tensor 840/858
[DEBUG] GGUFParser: Read tensor name: v.blk.9.attn_k.weight
[DEBUG] GGUFParser: Reading tensor 841/858
[DEBUG] GGUFParser: Read tensor name: v.blk.9.attn_v.weight
[DEBUG] GGUFParser: Reading tensor 842/858
[DEBUG] GGUFParser: Read tensor name: v.blk.9.ffn_down.bias
[DEBUG] GGUFParser: Reading tensor 843/858
[DEBUG] GGUFParser: Read tensor name: v.blk.9.ffn_down.weight
[DEBUG] GGUFParser: Reading tensor 844/858
[DEBUG] GGUFParser: Read tensor name: v.blk.9.ffn_gate.bias
[DEBUG] GGUFParser: Reading tensor 845/858
[DEBUG] GGUFParser: Read tensor name: v.blk.9.ffn_gate.weight
[DEBUG] GGUFParser: Reading tensor 846/858
[DEBUG] GGUFParser: Read tensor name: v.blk.9.ffn_up.bias
[DEBUG] GGUFParser: Reading tensor 847/858
[DEBUG] GGUFParser: Read tensor name: v.blk.9.ffn_up.weight
[DEBUG] GGUFParser: Reading tensor 848/858
[DEBUG] GGUFParser: Read tensor name: v.blk.9.ln1.weight
[DEBUG] GGUFParser: Reading tensor 849/858
[DEBUG] GGUFParser: Read tensor name: v.blk.9.ln2.weight
[DEBUG] GGUFParser: Reading tensor 850/858
[DEBUG] GGUFParser: Read tensor name: v.merger.ln_q.weight
[DEBUG] GGUFParser: Reading tensor 851/858
[DEBUG] GGUFParser: Read tensor name: v.merger.mlp.0.bias
[DEBUG] GGUFParser: Reading tensor 852/858
[DEBUG] GGUFParser: Read tensor name: v.merger.mlp.0.weight
[DEBUG] GGUFParser: Reading tensor 853/858
[DEBUG] GGUFParser: Read tensor name: v.merger.mlp.2.bias
[DEBUG] GGUFParser: Reading tensor 854/858
[DEBUG] GGUFParser: Read tensor name: v.merger.mlp.2.weight
[DEBUG] GGUFParser: Reading tensor 855/858
[DEBUG] GGUFParser: Read tensor name: v.patch_embd_0.weight
[DEBUG] GGUFParser: Reading tensor 856/858
[DEBUG] GGUFParser: Read tensor name: v.patch_embd_1.weight
[DEBUG] GGUFParser: Reading tensor 857/858
[DEBUG] GGUFParser: Read tensor name: output_norm.weight
[DEBUG] GGUFParser: Reading tensor 858/858
[DEBUG] GGUFParser: Read tensor name: output.weight
[DEBUG] GGUFParser: Read 858 tensor infos from mmap
[DEBUG] GGUFParser: Tensor info read successfully from mmap
[DEBUG] GGUFParser: Found rope.mrope_section metadata, data size: 24 bytes
[DEBUG] GGUFParser: rope.mrope_section array length: 3
[DEBUG] GGUFParser: Successfully parsed rope.mrope_section with 0 elements
[DEBUG] GGUFParser: Found vision.fullatt_block_indexes metadata, data size: 28 bytes
[DEBUG] GGUFParser: vision.fullatt_block_indexes array length: 4
[DEBUG] GGUFParser: Successfully parsed vision.fullatt_block_indexes with 0 elements
[INFO] GGUFParser: Parsed architecture: qwen25vl
[INFO] GGUFParser:   Context length: 128000
[INFO] GGUFParser:   Embedding length: 3584
[INFO] GGUFParser:   Block count: 28
[INFO] GGUFParser:   Has vision: Yes
[DEBUG] GGUFParser: Architecture parsed successfully
[INFO] GGUFParser: Successfully parsed GGUF file with mmap: /Users/acproject/.ollama/models/blobs/sha256-a99b7f834d754b88f122d865f32758ba9f0994a83f8363df2c1e71c17605a025
[INFO] GGUFParser: Architecture: qwen25vl
[INFO] GGUFParser: Metadata keys: 36
[INFO] GGUFParser: Tensor count: 858
[DEBUG] GGUFParser: GGUFParser destroyed
[DEBUG] OllamaModelManager: parseModelInfo succeeded
[DEBUG] OllamaModelManager: registerModel returned: true
[DEBUG] OllamaModelImpl: Normalized ID for loading: registry.ollama.ai_library_qwen2.5vl_7b
[DEBUG] MLInferenceEngine constructor called with model_id: registry.ollama.ai_library_qwen2.5vl_7b
[DEBUG] MLInferenceEngine::initialize called
[DEBUG] Detected architecture: 'qwen25vl', use_llama_backend_=false
[DEBUG] Initializing Qwen multimodal model for internal forward
[DEBUG] Defer QwenTextModel initialization to loadComponentModels()
[DEBUG] Vocabulary initialized with 151936 tokens
[DEBUG] Creating Qwen tokenizer via factory...
[DEBUG] createTextProcessorForArchitecture called with architecture='qwen'
[DEBUG] Determined tokenizer type: 'bpe'
[DEBUG] Using BPE pattern: '(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}+| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+'
[DEBUG] Creating BytePairEncoding tokenizer...
[DEBUG] BytePairEncoding tokenizer created successfully!
[DEBUG] Model config - vocab_size: 32000, n_layers: 32, n_heads: 32, n_embd: 4096, n_ctx: 2048
[DEBUG] Loading model weights for 32 layers
[DEBUG] Loaded 64 weight tensors
[DEBUG] Initializing KV cache with context length: 2048
[DEBUG] KV cache configuration prepared (maxSeqLen: 2048, numLayers: 32, numHeads: 32, headDim: 128)
[DEBUG] Precomputing RoPE frequencies
[DEBUG] Precomputed 64 RoPE frequencies
[DEBUG] MLInferenceEngine initialized successfully with internal forward (Qwen model)
[DEBUG] OllamaModelImpl: Setting loaded status to true
[DEBUG] OllamaModelImpl: Created TextGenerator with Ollama backend
[DEBUG] OllamaModelImpl::load completed successfully
[SUCCESS] Model loaded successfully: registry.ollama.ai/library/qwen2.5vl:7b (took 4403ms)
[DEBUG] ChatView: Model loaded successfully: registry.ollama.ai/library/qwen2.5vl:7b
[DEBUG] ChatView: Getting text generator for model: registry.ollama.ai/library/qwen2.5vl:7b
[DEBUG] ModelManager::getTextGenerator called for: registry.ollama.ai/library/qwen2.5vl:7b
[DEBUG] Model found in loaded_models_, attempting cast to OllamaModelImpl
[DEBUG] Successfully cast to OllamaModelImpl, calling getTextGenerator()
[DEBUG] OllamaModelImpl::getTextGenerator returned: valid pointer
[DEBUG] TextGenerator::canGenerate() called - returning true (functionality enabled)
[DEBUG] ChatView: Starting text generation...
[DEBUG] TextGenerator::generate() called with prompt: 你好...
[DEBUG] Using Ollama model manager for inference
[DEBUG] OllamaModelManager::generateText called with model: registry.ollama.ai_library_qwen2.5vl_7b
[DEBUG] Normalized model ID: registry.ollama.ai_library_qwen2.5vl_7b
[DEBUG] Currently registered models:
[DEBUG]   - registry.ollama.ai_library_qwen2.5vl_7b
[DEBUG] Model load state for registry.ollama.ai_library_qwen2.5vl_7b: 2
[DEBUG] MLInferenceEngine::generateText called with prompt: '你好', max_tokens: 512, temperature: 0.7, top_p: 0.9
[DEBUG] [InternalForward] Starting Qwen model inference
[DEBUG] [InternalForward] Using Qwen tokenization
[DEBUG] [InternalForward] Encoded 3 tokens from prompt
[DEBUG] [InternalForward] Qwen model generated 3 output tokens
[DEBUG] BytePairEncoding::decode called with 3 token IDs
[DEBUG] Vocabulary size: 151936
[DEBUG] First 10 token IDs: 1 145 2 
[DEBUG] Token[0] ID=1 -> '<bos>' (length: 5)
[DEBUG] Token[1] ID=145 -> 'B' (length: 1)
[DEBUG] Token[2] ID=2 -> '<eos>' (length: 5)
[DEBUG] Final decoded string length: 11
[DEBUG] Final decoded string (first 100 chars): '<bos>B<eos>'
[DEBUG] [InternalForward] Using Qwen detokenization
[DEBUG] [InternalForward] Generated response: <bos>B<eos>...
[DEBUG] Text generation completed successfully
[DEBUG] Ollama inference successful: <bos>B<eos>...
[DEBUG] TextGenerator returning result: <bos>B<eos>...
[DEBUG] ChatView: Text generation completed successfully
```