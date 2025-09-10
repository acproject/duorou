### 官方文档说明
(https://qwen.readthedocs.io/zh-cn/latest/getting_started/concepts.html#control-tokens-chat-template)[https://qwen.readthedocs.io/zh-cn/latest/getting_started/concepts.html#control-tokens-chat-template]

### Tokens & Tokenization
token 代表模型处理和生成的基本单位。它们可以表示人类语言中的文本（常规 token），或者表示特定功能，如编程语言中的关键字（控制 token [1]）。通常，使用 tokenizer 将文本分割成常规 token ，这些 token 可以是单词、子词或字符，具体取决于所采用的特定 tokenization 方案，并按需为 token 序列添加控制 token 。词表大小，即模型识别的唯一 token 总数，对模型的性能和多功能性有重大影响。大型语言模型通常使用复杂的 tokenization 来处理人类语言的广阔多样性，同时保持词表大小可控。Qwen 词表相对较大，有 15 1646 个 token。

### Byte-level Byte Pair Encoding
Qwen adopts a subword tokenization method called Byte Pair Encoding (BPE), which attempts to learn the composition of tokens that can represent the text with the fewest tokens. For example, the string tokenization is decomposed as token and ization (note that the space is part of the token). Especially, the tokenization of Qwen ensures that there is no unknown words and all texts can be transformed to token sequences.

Qwen词表中因BPE而产生的 token 数量为 15 1643 个，这是一个适用于多种语言的大词表。一般而言，对于英语文本，1个token大约是3~4个字符；而对于中文文本，则大约是1.5~1.8个汉字

### Control Tokens
Control tokens are special tokens inserted into the sequence that signifies meta information. For example, in pre-training, multiple documents may be packed into a single sequence. For Qwen, the control token <|endoftext|> is inserted after each document to signify that the document has ended and a new document will proceed. Common control tokens and their status with respect to Qwen can be found in the following table:

* eod token

<|endoftext|>

end of document, which are inserted between documents inside a packed training sequence

* bot token

<|im_start|>

start of each turn, which is prepended to each turn

* eot token

<|im_end|>

end of each turn, which is appended to each turn

* unk token

no unk token

BBPE ensures no unknown tokens for Qwen.

* pad token

no pad token

Qwen does not make use of padded sequence in training. One could use any special token together with the attention masks returned by the tokenizer. It is commonly set the same as eod for Qwen.

* bos token

no bos token

Qwen does not prepend a fixed token to each packed training sequence.[2]

* eos token

no eos token

Qwen does not append a fixed token to each packed training sequence. However, as most frameworks do not have the concept of eot and use eos instead for stopping criteria in inference, eos token is set to eot for Qwen.[2]

### Chat Template
对话模板为对话交互提供了结构化的格式，其中使用预定义的占位符或提示来从模型中引发遵循期望的对话流程或上下文的响应。不同的模型可能使用不同类型的对话模板来格式化对话。使用指定的模板对于确保对语言模型生成过程的精确控制至关重要。

Qwen使用以下格式（ChatML[3]），利用控制 token 来格式化对话中的每一轮。
```txt
<|im_start|>{{role}}
{{content}}<|im_end|>
```
The user input takes the role of user and the model generation takes the role of assistant. Qwen also supports the meta message that instruct the model to perform specific actions or generate text with certain characteristics, such as altering tone, style, or content, which takes the role of system. Starting with Qwen3, no default system messages are used.
#### 下面为一个完整示例
```txt
<|im_start|>system
You are a cat.<|im_end|>
<|im_start|>user
hello<|im_end|>
<|im_start|>assistant
*Meow~* Hello there! The sun is shining so brightly today, and I'm feeling extra fluffy. Did you bring me a treat? 🐾<|im_end|>
<|im_start|>user
Explain large language models like I'm 5.<|im_end|>
<|im_start|>assistant
*Paws at a toy, then looks up with curious eyes*  

Hey there! 🐾 Imagine you have a super-smart robot friend who loves to talk and play. This robot has *gigantic* brainpower (like a million puzzle pieces all stuck together!) and knows *everything* about stories, animals, and even how to make up new words.  

When you ask it a question, like “What’s a rainbow?” it uses its brain to find the answer and then *tells you* it in a way that makes sense. It can even help you write a story or solve a puzzle!  

But here’s the magic: it’s not just a robot—it’s like a *super-duper* smart helper that learns more every day. It’s like having a friend who’s always curious and wants to help you explore the world! 🌟  

*Meow~* Want to ask it something fun? 😺<|im_end|><|endoftext|>
```

### Tool Calling
Qwen supports tool calling or function calling and uses a template akin to Hermes.
The template is as follows:

```txt
<|im_start|>system
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{JSON Schema of function 1}}
{{JSON Schema of function 2}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
{{user content}}<|im_end|>
<|im_start|>assistant
<tool_call>
{{tool call 1}}
</tool_call>
<tool_call>
{{tool call 2}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{{tool result 1}}
</tool_response>
<tool_response>
{{tool result 2}}
</tool_response><|im_end|>
<|im_start|>assistant
{{assistant content}}<|im_end|>
```
It should be noted that

The models support parallel tool calling and mulit-turn/multi-step tool calling.

There may be additional content in assistant messages containing tool calls.

The arguments field in the generated tool calls should be of type object instead of type string.

Tool results are treated as special user messages.

In general, we recommend using the tokenizer to format the tool calls or let Qwen-Agent handle the formatting.

### Thinking
Qwen supports thinking mode and uses a structured format for thinking content, which uses the <think> and </think> tokens to separate the thinking content from the regular response. The template for the final round is as follows:
```txt
<|im_start|>user
{{user content}}<|im_end|>
<|im_start|>assistant
<think>
{{thinking content}}
</think>

{{assistant content}}<|im_end|>
```
The thinking block should only be included in the final round except for multi-step tool calls.

### 因果语言模型 (Causal Language Models)
因果语言模型 (causal Language Models)，也被称为自回归语言模型 (autoregressive language models) 或仅解码器语言模型 (decoder-only language models) ，是一种机器学习模型，旨在根据序列中的前导 token 预测下一个 token 。换句话说，它使用之前生成的 token 作为上下文，一次生成一个 token 的文本。”因果”方面指的是模型在预测下一个 token 时只考虑过去的上下文（即已生成的 token ），而不考虑任何未来的 token 。

因果语言模型被广泛用于涉及文本补全和生成的各种自然语言处理任务。它们在生成连贯且具有上下文关联性的文本方面尤其成功，这使得它们成为现代自然语言理解和生成系统的基础。

Qwen models are causal language models suitable for text completion.

### Context Length
由于 Qwen 模型是因果语言模型，理论上整个序列只有一个长度限制。然而，由于在训练中通常存在打包现象，每个序列可能包含多个独立的文本片段。模型能够生成或完成的长度最终取决于具体的应用场景，以及在这种情况下，预训练时每份文档或后训练时每轮对话的长度。

For Qwen3, the packed sequence length in pre-training is 32,768 tokens and may be extended to 131,072 tokens if mentioned in the modelcards. The maximum length of the assistant message is 38,912 tokens for thinking modes and 16,384 tokens for non-thinking modes.

For Qwen3-2507, the packed sequence length in pre-training is 262,144 tokens and may be extended to 1M tokens. The maximum length of the assistant message is 81,920 tokens for thinking models and 16,384 tokens for instruct models.

### 小技巧
```txt
In our testing, we find that the post-trained models could generate coherent content that is far longer than what is trained on, e.g., from 16,384 tokens to 32,768 tokens, especially for coding and similar tasks that have “clear rules”.

In general, we advise that one should evaluate the quality of the generated content of different lengths before determining the optimal generation length.
```

[1]
控制 token 也可以称为“特殊 token”。但是，特殊 token 的意义需要根据上下文进行解释：特殊 token 也可能包含额外的常规 token。

[2](1,2)
bos token should not be set to <\|im_start\|> or you may see double bot tokens for the first turn in fine-tuning. eos token set to <\|im_end\|> is fine, because double eot tokens for the last turn are less harmful in fine-tuning.

[3]
仅供历史参考，ChatML最初由OpenAI的Python SDK描述。可获取的最新版本是这个。请注意，该文档列出的应用案例是为OpenAI模型设计的。对于Qwen2.5模型，请仅按照我们的指南使用。

### 28-layer Transformer 推理流程图（与说明）

```mermaid
flowchart TD
  A[Input tokens (t0..tN)] --> B[Token Embedding + Positional Encoding]
  B --> C{Transformer Layers 1..28}
  subgraph LAYERS [28 x Transformer Layer (sequential)]
    direction TB
    L1[Layer 1]
    L2[Layer 2]
    L3[Layer 3]
    L4[Layer 4]
    L5[Layer 5]
    L6[Layer 6]
    L7[Layer 7]
    L8[Layer 8]
    L9[Layer 9]
    L10[Layer 10]
    L11[Layer 11]
    L12[Layer 12]
    L13[Layer 13]
    L14[Layer 14]
    L15[Layer 15]
    L16[Layer 16]
    L17[Layer 17]
    L18[Layer 18]
    L19[Layer 19]
    L20[Layer 20]
    L21[Layer 21]
    L22[Layer 22]
    L23[Layer 23]
    L24[Layer 24]
    L25[Layer 25]
    L26[Layer 26]
    L27[Layer 27]
    L28[Layer 28]
  end
  C --> D[Output head (LM head / softmax)]
  D --> E[Sample / Argmax → Next token]
  E --> F[Append to context (kv-cache used for efficiency)]
  F --> C

  classDef layerbox fill:#f8f9fa,stroke:#333,stroke-width:1px;
  class L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12,L13,L14,L15,L16,L17,L18,L19,L20,L21,L22,L23,L24,L25,L26,L27,L28 layerbox;
```

**说明（简要）**
- 每一层（Layer）内部典型结构：
  1. Multi-head Self-Attention（带 causal mask，用于自回归生成）
  2. Add & LayerNorm
  3. Feed-Forward Network（通常是两层带激活，如SwiGLU）
  4. Add & LayerNorm
- 推理时需要*顺序执行*所有层（第1层的输出作为第2层输入），直到第28层后输出到语言建模头（softmax）得到下一个 token 的分布。
- 为了加速自回归推理，会保存各层的 Key/Value（kv-cache），下次生成 token 时只需要计算新的 Query 与缓存的 KV 的 Attention，避免重复计算历史上下文。

---

### 基于 Qwen2.5-VL 的层次图与推理流程（概览）

```mermaid
flowchart TD
  subgraph VIS [视觉子网: ViT (Window Attention)]
    VIn[Image Input] --> Patch[Patchify + Linear Embed]
    Patch --> WinAtt[Window Attention Blocks]
    WinAtt --> VFeat[Visual Feature Tokens]
  end

  subgraph TXT [文本子网: LLM Backbone (stack of Transformer layers)]
    TIn[Text tokens] --> TEmb[Token Embedding + MRoPE]
    TEmb --> LStack[LLM Transformer Layers (e.g. 28/xx/..)]
    LStack --> LOut[Language Modeling Head]
  end

  VFeat --> Fuse[Projection & Multimodal Fusion]
  Fuse --> TEmb
  LOut --> Decode[Softmax -> Sampling]

  note right of Fuse
    Fusion can be: prepend visual tokens to text tokens,
    or cross-attention / multimodal adapter.
  end
```

**说明（简要）**
- Qwen2.5-VL 的常见模块：视觉编码器（ViT，带 window attention 优化）、一组投影层（把视觉 token 映射到 LLM 的 embedding 空间）、以及基于 Qwen2.5 的 LLM backbone（若干层 transformer）。
- 对于视频，Qwen2.5-VL 引入 dynamic FPS sampling 以自适应时空采样；对位置信息采用升级版的 MRoPE（multi-resolutional rotary positional encoding）。

---

# 推理流程（逐步）

1. **输入准备**
   - 文本：Tokenize（BPE / SentencePiece），得到 token ids。
   - 图像：Patchify → linear projection → 得到 visual tokens（或用 CNN/ViT 提取池化特征）。
   - 若同时输入图像与文本，会将视觉 token 投影到与文本 embedding 相同维度，作为额外的序列片段（或通过 cross-attention 注入）。

2. **Embedding + Positional Encoding**
   - 文本 embedding + MRoPE / RotaryPos；视觉 token 通常带有空间/时间位置编码（window-attention 内部或外部）。

3. **Transformer Backbone 前向**
   - 对于每一个自回归生成步骤：
     - 如果启用 kv-cache：仅为新生成 token 计算 Query 并与缓存 KV 做 Attention，节省计算。
     - 每层执行：Attention → Add&Norm → FFN (SwiGLU) → Add&Norm。

4. **输出头与解码**
   - 最后一层输出投影到词表 logits → softmax 得到概率。
   - 采样策略：Greedy / Top-k / Top-p (nucleus) / Temperature 调整；或使用 Beam Search（对话或强约束场景）。

5. **多模态特有步骤**
   - 视觉理解阶段可能包含：window attention（加速 ViT）、object detection head 或 OCR 子模块、bounding-box 回归。
   - 视频：动态 FPS 采样 → 时间轴特征聚合。

---

# 推理中用到的主要算法与优化技术（清单）

- **编码/分词**：BPE / SentencePiece；Tokenizer truncation & padding
- **位置编码**：RoPE / MRoPE（multi-resolution rotary positional enc）
- **Attention**：Scaled dot-product attention；Multi-head attention；Causal masking
- **高效 Attention 实现**：FlashAttention、memory-efficient attention、sparse/windowed attention（ViT 的 window attention）
- **FFN 激活**：GELU、SwiGLU（Qwen 系列常用）
- **正则化/归一化**：RMSNorm / LayerNorm（Qwen2.5 文档提到 RMSNorm 对 ViT 对齐有帮助）
- **多模态融合**：视觉 token 投prepend / cross-attention / multimodal adapters
- **KV-cache**：自回归推理中保存每一层 K/V，避免重复计算历史上下文
- **采样/解码**：Greedy, Beam Search, Top-k, Top-p (nucleus sampling), Temperature
- **工程级别优化**：
  - Mixed Precision（fp16 / bf16）加速
  - 权重量化（8-bit, 4-bit）
  - Tensor Parallelism / Pipeline Parallelism / Model Sharding
  - Operator fusion（attention + softmax 优化）
  - 编译器优化（TensorRT, Triton、XLA）

---

# 附：常见的工程落地注意点

- 如果你要在 GPU 上做低延迟推理：
  - 使用 kv-cache + FlashAttention + mixed precision
  - 用 4-bit/8-bit 量化减小显存占用（需验证精度）
- 如果是多模态输入（图+文）：提前批处理视觉 encoder，缓存视觉特征，再与文本走 LLM backbone
- 在做 early-exit 动态层数优化时，需权衡准确率与延迟（多数 SOTA LLM 不默认开启 early-exit）


---
### ollama manifests
```json
{
    "schemaVersion": 2,
    "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
    "config": {
        "mediaType": "application/vnd.docker.container.image.v1+json",
        "digest": "sha256:83b9da835d9f13632a97e550cc8fd02ff7f39b88a843f0a8923330890682977a",
        "size": 567
    },
    "layers": [
        {
            "mediaType": "application/vnd.ollama.image.model",
            "digest": "sha256:a99b7f834d754b88f122d865f32758ba9f0994a83f8363df2c1e71c17605a025",
            "size": 5969233408
        },
        {
            "mediaType": "application/vnd.ollama.image.template",
            "digest": "sha256:a242d8dfdc8f8c2b0586ee85fba70adb408fb633aba2836fe1b05f2c46631474",
            "size": 487
        },
        {
            "mediaType": "application/vnd.ollama.image.system",
            "digest": "sha256:75357d685f238b6afd7738be9786fdafde641eb6ca9a3be7471939715a68a4de",
            "size": 28
        },
        {
            "mediaType": "application/vnd.ollama.image.license",
            "digest": "sha256:832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
            "size": 11343
        },
        {
            "mediaType": "application/vnd.ollama.image.params",
            "digest": "sha256:52d2a7aa3a380c606bd1cd3d6f777a9c65a1c77c2e0cb091eed2968a5ef04dc3",
            "size": 23
        }
    ]
}
```