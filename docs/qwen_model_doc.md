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

