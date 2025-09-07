# Qwen2.5-VL 文本生成异常问题记录与排查路线

现象：输入简单中文“你好”，模型返回与问题严重不相干的多语混杂文本，且存在相同 token（如 151935）连续重复并触发早停，单次生成耗时约 149s（见下方 DEBUG 日志）。

初步判断：该问题更像是“前向逻辑或权重装载不一致导致的整体语义退化”，而非纯粹乱码。重点怀疑以下环节（按优先级）：
- 线性层权重布局/转置错误：权重是否按列主序/行主序、是否需要转置与我们实现一致；safetensors/ggml 权重是否被错误地当作转置矩阵使用，导致 Q/K/V/Out 或 MLP Up/Down/Gate 数值失真。
- 注意力头与 KV 头不匹配（GQA）：numHeads 与 numKVHeads 不一致时，Q/K/V reshape 与 KV cache 读写维度必须严格匹配，否则注意力分数无意义，采样呈现随机/跨语种碎片。
- RoPE 配置错误：ropeDim、ropeBase、ropeScale 以及 RoPE 类型（NeoX vs LLaMA）若不一致，会造成随位置增长退化。需确认 originalContextLength 与 scale 的使用是否与模型一致。
- Tokenizer/词表不一致：SentencePiece 的空格替换符“▁”、特殊 token、字节回退 token 与我们使用的 HF tokenizer 是否一一对应；编码/解码若不一致，输入“你好”的 id 列表会不同，最终 detokenize 亦会异常混杂。
- 最后一层 Norm/输出投影错配：OutputNorm 与 Output Linear 是否正确加载到对应权重；若错位，logits 会整体无意义。
- 多模态路径写入越界：将图像 embedding 写入 hiddenStates 的 View/Stride 若有误，可能破坏文本 token 的 embedding。
- 量化/反量化缩放错误：Q4_K_M 等量化方案的 scale/zero-point 应用是否正确；错误缩放会放大噪声。
- 采样超参与重复惩罚：在数值已退化的情况下，温度过高或无重复惩罚，会加剧“随机多语碎片”的主观观感。

与 Ollama 实现的关键对照点（基于本文档下方源码片段）：
- SelfAttention: Q、K、V 分别线性投影并 reshape，其中 Q 使用 numHeads、K/V 使用 numKVHeads（GQA），均施加 RoPE(NeoX 类型)，scaleFactor=1/sqrt(headDim)。
- Layer 结构：RMSNorm → SelfAttention → 残差 → RMSNorm → MLP(SwiGLU) → 残差；最后 OutputNorm 后接 Output 线性得到 logits。
- Tokenizer：SentencePiece 使用“▁”替换空格，遇到未知 token 回退到字节 token（形如 <0xEA>）。

快速自检清单（建议逐项加断言/日志验证）：
1) 线性层权重形状与转置
   - 验证所有 Linear 的 in/out 维度与模型 meta 一致；加载后对随机向量前向一次，与 Ollama 的相同层输出做数值对比（允许微小误差）。
2) 注意力维度/缓存
   - headDim = hiddenSize/numHeads，KV 采用 numKVHeads；检查 reshape/permute 顺序与 KV cache 的布局一致性（特别是 seq/batch/head/hdim 顺序）。
3) RoPE 参数
   - ropeDim、ropeBase、ropeScale、originalContextLength 与 NeoX 类型是否与模型 meta 完全一致；对第一个 token 的 Q/K 施加 RoPE 后的范数与 Ollama 对比。
4) Tokenizer 一致性
   - 打印“你好”的 Encode 结果，与文中 Ollama 的 SentencePiece 编码结果对比；验证 Decode(Encode(s)) 应严格还原（考虑空格为“▁”）。
5) OutputNorm/OutputLinear
   - 检查输出层权重是否正确对应，并对同一 hiddenStates 输入比对 logits top-5 与 Ollama 是否一致（温度=0，禁采样）。
6) 多模态写入
   - 检查 hiddenStates.View 的起始偏移与 Stride，确保图像张量不覆盖文本 token 的 embedding；可临时禁用多模态路径验证纯文本是否恢复正常。
7) 量化反量化
   - 核对每一层的量化 scale/zero-point 使用是否正确，并在部分层以 FP16/FP32 回退做 A/B 测试定位。
8) 采样参数
   - 暂时采用保守参数：temperature=0.2、top_p=0.9、repeat_penalty=1.2、no_repeat_ngram=禁止；打印每步 logits top-5 与被采样的 token 以辅助判断是否“源头退化”。

建议的最小复现实验（与 Ollama 对齐）：
- 关闭采样（温度=0，贪心）在前 5 个 token 上比较 logits top-5 及其概率分布；若显著偏离，优先排查前向计算与权重装载。
- 打印第 1/2/3 层的 Q/K/V 范数、注意力分数均值/方差，若分布异常（过小/过大/NaN），重点排查 RoPE 与缩放。
- Tokenizer 回归：Encode/Decode 多组中英混合字符串，确保一致。

修复优先级与落地步骤：
1) 校验并对齐 RoPE（NeoX/ropeDim/base/scale/originalContextLength）。
2) 校验 GQA 维度：numHeads/numKVHeads 所有层一致，KV cache 读写 shape/stride 一致。
3) 校验 Linear 权重布局：逐层做单向数值对齐测试，必要时尝试转置或更换读取顺序。
4) 校验 Tokenizer 与特殊 token：保证 BOS/EOS/特殊控制符与词表 id 对齐。
5) 降采样温度并开启重复惩罚作为临时缓解措施，待数值对齐后恢复默认。

下文保留了完整 DEBUG 日志与 Ollama 参考实现，便于对照排查。

```sh
EBUG] Found tokenizer.ggml.tokens metadata (array type)
[DEBUG] Array data size: 2590664 bytes
[DEBUG] Array type: 8 (STRING=8)
[DEBUG] Array length: 152064
[DEBUG] Successfully parsed 152064 tokens from GGUF
[DEBUG] Token 56064: .EqualTo
[DEBUG] Token 133718: ÙħÙĪØ§Ø¬Ùĩ
[DEBUG] Token 29391: .plugins
[DEBUG] Token 131840: Ġnghá»ī
[DEBUG] Token 115382: æĴŀåĩ»
[DEBUG] Token 22828: ĠmarginTop
Warning: Invalid regex pattern, using simple whitespace split: One of *?+{ was not preceded by a valid regular expression.
[WARNING] OllamaModelManager: Requested context length 32768 exceeds cap 2048, clamping to cap to avoid OOM
[INFO] OllamaModelManager: Setting max sequence length: requested=32768, using=2048
[DEBUG] Qwen25VLInferenceEngine::generateText called with prompt: Hello...
[DEBUG] Model is loaded, starting text generation
[DEBUG] Starting tokenization
[DEBUG] Tokenizing text: "Hello"
[DEBUG] TextProcessor tokenized into 1 tokens
[DEBUG] Tokenization completed, tokens: 1
[DEBUG] Starting forward pass loop
[DEBUG] Forward pass iteration: 0
[DEBUG] Calling forward() with 1 tokens
[DEBUG] Detokenizing 0 tokens
[DEBUG] TextProcessor detokenized result: ""
[INFO] OllamaModelManager: Model loaded successfully: registry.ollama.ai_rockn_Qwen2.5-Omni-7B-Q4_K_M_latest
[DEBUG] OllamaModelImpl: Setting loaded status to true
[DEBUG] OllamaModelImpl: Created TextGenerator with Ollama backend
[DEBUG] OllamaModelImpl::load completed successfully
[SUCCESS] Model loaded successfully: registry.ollama.ai/rockn/Qwen2.5-Omni-7B-Q4_K_M:latest (took 55742ms)
[DEBUG] ChatView: Model loaded successfully: registry.ollama.ai/rockn/Qwen2.5-Omni-7B-Q4_K_M:latest
[DEBUG] ChatView: Getting text generator for model: registry.ollama.ai/rockn/Qwen2.5-Omni-7B-Q4_K_M:latest
[DEBUG] ModelManager::getTextGenerator called for: registry.ollama.ai/rockn/Qwen2.5-Omni-7B-Q4_K_M:latest
[DEBUG] Model found in loaded_models_, attempting cast to OllamaModelImpl
[DEBUG] Successfully cast to OllamaModelImpl, calling getTextGenerator()
[DEBUG] OllamaModelImpl::getTextGenerator returned: valid pointer
[DEBUG] TextGenerator::canGenerate() called - returning true (functionality enabled)
[DEBUG] ChatView: Starting text generation...
[DEBUG] TextGenerator::generate() called with prompt: 你好...
[DEBUG] Using Ollama model manager for inference
[DEBUG] Calling engine->generateText with prompt: 你好...
[DEBUG] Qwen25VLInferenceEngine::generateText called with prompt: 你好...
[DEBUG] Model is loaded, starting text generation
[DEBUG] Starting tokenization
[DEBUG] Tokenizing text: "你好"
[DEBUG] TextProcessor tokenized into 1 tokens
[DEBUG] Tokenization completed, tokens: 1
[DEBUG] Starting forward pass loop
[DEBUG] Forward pass iteration: 0
[DEBUG] Calling forward() with 1 tokens
[DEBUG] Detokenizing 0 tokens
[DEBUG] TextProcessor detokenized result: ""
[DEBUG] Ollama inference successful: ...
[DEBUG] TextGenerator returning result: ...
[DEBUG] ChatView: Text generation failed or returned empty result
```
可以看见CLI上面显示的内容为空，并没有需改成功。

#### ollama部分源代码
```go
package qwen25vl

import (
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/fast"
	"github.com/ollama/ollama/ml/nn/rope"
	"github.com/ollama/ollama/model/input"
)

type TextOptions struct {
	hiddenSize, numHeads, numKVHeads int
	ropeDim, originalContextLength   int
	eps, ropeBase, ropeScale         float32
}

type TextModel struct {
	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	Layers         []Layer       `gguf:"blk"`
	OutputNorm     *nn.RMSNorm   `gguf:"output_norm"`
	Output         *nn.Linear    `gguf:"output,alt:token_embd"`

	*TextOptions
}

func NewTextModel(c fs.Config) *TextModel {
	m := TextModel{
		Layers: make([]Layer, c.Uint("block_count")),
		TextOptions: &TextOptions{
			hiddenSize:            int(c.Uint("embedding_length")),
			numHeads:              int(c.Uint("attention.head_count")),
			numKVHeads:            int(c.Uint("attention.head_count_kv")),
			ropeDim:               int(c.Uint("rope.dimension_count", 128)),
			originalContextLength: int(c.Uint("context_length", 128000)),
			eps:                   c.Float("attention.layer_norm_rms_epsilon"),
			ropeBase:              c.Float("rope.freq_base"),
			ropeScale:             c.Float("rope.freq_scale", 1),
		},
	}

	return &m
}

// SelfAttention implements the multi-head self-attention mechanism
// with separate projections for query, key, value and output transformations
type SelfAttention struct {
	Query  *nn.Linear `gguf:"attn_q"`
	Key    *nn.Linear `gguf:"attn_k"`
	Value  *nn.Linear `gguf:"attn_v"`
	Output *nn.Linear `gguf:"attn_output"`
}

func (sa *SelfAttention) Forward(ctx ml.Context, hiddenState, positionIDs ml.Tensor, cache kvcache.Cache, opts *TextOptions) ml.Tensor {
	batchSize := hiddenState.Dim(1)
	headDim := opts.hiddenSize / opts.numHeads

	q := sa.Query.Forward(ctx, hiddenState)
	q = q.Reshape(ctx, headDim, opts.numHeads, batchSize)
	q = fast.RoPE(ctx, q, positionIDs, opts.ropeDim, opts.ropeBase, opts.ropeScale, rope.WithOriginalContextLength(opts.originalContextLength), rope.WithTypeNeoX())

	k := sa.Key.Forward(ctx, hiddenState)
	k = k.Reshape(ctx, headDim, opts.numKVHeads, batchSize)
	k = fast.RoPE(ctx, k, positionIDs, opts.ropeDim, opts.ropeBase, opts.ropeScale, rope.WithOriginalContextLength(opts.originalContextLength), rope.WithTypeNeoX())

	v := sa.Value.Forward(ctx, hiddenState)
	v = v.Reshape(ctx, headDim, opts.numKVHeads, batchSize)

	scaleFactor := 1.0 / math.Sqrt(float64(headDim))
	kqv := nn.Attention(ctx, q, k, v, scaleFactor, cache)
	kqv = kqv.Reshape(ctx, opts.hiddenSize, batchSize)

	return sa.Output.Forward(ctx, kqv)
}

// Shift applies rotary position embeddings to the key tensor for causal attention caching
func (m *TextModel) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	return fast.RoPE(ctx, key, shift, m.ropeDim, m.ropeBase, m.ropeScale, rope.WithOriginalContextLength(m.originalContextLength), rope.WithTypeNeoX()), nil
}

// MLP implements the feed-forward network component with SwiGLU activation
type MLP struct {
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
	Gate *nn.Linear `gguf:"ffn_gate"`
}

func (mlp *MLP) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *TextOptions) ml.Tensor {
	// Apply SwiGLU activation gating
	hiddenState = mlp.Gate.Forward(ctx, hiddenState).SILU(ctx).Mul(ctx, mlp.Up.Forward(ctx, hiddenState))
	// Project back to hidden dimension
	return mlp.Down.Forward(ctx, hiddenState)
}

// Layer represents a single transformer layer combining self-attention and feed-forward components
type Layer struct {
	AttentionNorm *nn.RMSNorm `gguf:"attn_norm"`
	SelfAttention *SelfAttention
	MLPNorm       *nn.RMSNorm `gguf:"ffn_norm"`
	MLP           *MLP
}

func (l *Layer) Forward(ctx ml.Context, hiddenState, positionIDs, outputs ml.Tensor, cache kvcache.Cache, opts *TextOptions) ml.Tensor {
	// Self-attention branch with residual connection
	residual := hiddenState

	hiddenState = l.AttentionNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.SelfAttention.Forward(ctx, hiddenState, positionIDs, cache, opts)

	// In the final layer (outputs != nil), optimize by pruning to just the token positions
	// we need logits for.
	if outputs != nil {
		hiddenState = hiddenState.Rows(ctx, outputs)
		residual = residual.Rows(ctx, outputs)
	}

	hiddenState = hiddenState.Add(ctx, residual)
	// Feed-forward branch with residual connection
	residual = hiddenState
	hiddenState = l.MLPNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.MLP.Forward(ctx, hiddenState, opts)
	return hiddenState.Add(ctx, residual)
}

func (m *TextModel) Forward(ctx ml.Context, inputs, positions, outputs ml.Tensor, batch input.Batch, cache kvcache.Cache) (ml.Tensor, error) {
	// Initial token embedding
	hiddenStates := m.TokenEmbedding.Forward(ctx, inputs).Duplicate(ctx)

	for _, mi := range batch.Multimodal {
		img := mi.Multimodal[0].Tensor
		ctx.Forward(img.Copy(ctx, hiddenStates.View(ctx, mi.Index*hiddenStates.Stride(1), img.Dim(0)*img.Dim(1))))
	}

	// Process through transformer layers
	for i, layer := range m.Layers {
		cache.SetLayer(i)

		var lastLayerOutputs ml.Tensor
		if i == len(m.Layers)-1 {
			lastLayerOutputs = outputs
		}

		hiddenStates = layer.Forward(ctx, hiddenStates, positions, lastLayerOutputs, cache, m.TextOptions)
	}

	hiddenStates = m.OutputNorm.Forward(ctx, hiddenStates, m.eps)
	return m.Output.Forward(ctx, hiddenStates), nil
}

```

```go
package model

import (
	"container/heap"
	"context"
	"fmt"
	"log/slog"
	"strconv"
	"strings"

	"github.com/ollama/ollama/logutil"
)

const spmWhitespaceSep = "▁"

type SentencePieceModel struct {
	maxTokenLen int
	vocab       *Vocabulary
}

var _ TextProcessor = (*SentencePieceModel)(nil)

func (spm SentencePieceModel) Vocabulary() *Vocabulary {
	return spm.vocab
}

func NewSentencePieceModel(vocab *Vocabulary) SentencePieceModel {
	slog.Log(context.TODO(), logutil.LevelTrace, "Tokens", "num tokens", len(vocab.Values), "vals", vocab.Values[:5], "scores", vocab.Scores[:5], "types", vocab.Types[:5])

	counter := map[int]int{}
	var maxTokenLen int
	for cnt := range vocab.Types {
		switch vocab.Types[cnt] {
		case TOKEN_TYPE_NORMAL, TOKEN_TYPE_USER_DEFINED, TOKEN_TYPE_UNUSED:
			maxTokenLen = max(maxTokenLen, len(vocab.Values[cnt]))
			fallthrough
		default:
			counter[int(vocab.Types[cnt])] += 1
		}
	}

	slog.Log(context.TODO(), logutil.LevelTrace, "Token counts", "normal", counter[TOKEN_TYPE_NORMAL], "unknown", counter[TOKEN_TYPE_UNKNOWN], "control", counter[TOKEN_TYPE_CONTROL],
		"user defined", counter[TOKEN_TYPE_USER_DEFINED], "unused", counter[TOKEN_TYPE_UNUSED], "byte", counter[TOKEN_TYPE_BYTE],
		"max token len", maxTokenLen)

	return SentencePieceModel{
		maxTokenLen: maxTokenLen,
		vocab:       vocab,
	}
}

func (spm SentencePieceModel) Is(id int32, special Special) bool {
	return spm.vocab.Is(id, special)
}

func (spm SentencePieceModel) Encode(s string, addSpecial bool) ([]int32, error) {
	fragments := []fragment{{value: s}}
	for _, special := range spm.vocab.SpecialVocabulary() {
		id := spm.vocab.Encode(special)
		for i := 0; i < len(fragments); i++ {
			frag := fragments[i]
			if len(frag.ids) > 0 {
				continue
			}

			var middle []fragment
			switch i := strings.Index(frag.value, special); {
			case i < 0:
				middle = append(middle, frag)
			case i > 0:
				middle = append(middle, fragment{value: frag.value[:i]})
				fallthrough
			default:
				middle = append(middle, fragment{value: special, ids: []int32{id}})
				if rest := frag.value[i+len(special):]; rest != "" {
					middle = append(middle, fragment{value: rest})
				}
			}

			fragments = append(fragments[:i], append(middle, fragments[i+1:]...)...)
		}
	}

	var ids []int32
	for _, frag := range fragments {
		if len(frag.ids) > 0 {
			ids = append(ids, frag.ids...)
			continue
		}

		text := strings.ReplaceAll(frag.value, " ", spmWhitespaceSep)

		if id := spm.vocab.Encode(text); id >= 0 {
			ids = append(ids, id)
			continue
		}

		q := &queue{}
		heap.Init(q)

		runes := []rune(text)
		merges := make([]merge, len(runes))
		for r := range runes {
			merges[r] = merge{
				p:     r - 1,
				n:     r + 1,
				runes: []rune{runes[r]},
			}
		}

		pairwise := func(a, b int) *candidate {
			if a < 0 || b >= len(runes) {
				return nil
			}

			left, right := string(merges[a].runes), string(merges[b].runes)
			if id := spm.vocab.Encode(left + right); id >= 0 {
				return &candidate{
					a:     a,
					b:     b,
					score: spm.vocab.Scores[id],
					size:  len(left) + len(right),
				}
			}

			return nil
		}

		for i := range len(runes) - 1 {
			if pair := pairwise(i, i+1); pair != nil {
				heap.Push(q, pair)
			}
		}

		for q.Len() > 0 {
			pair := heap.Pop(q).(*candidate)
			left, right := merges[pair.a], merges[pair.b]

			if string(left.runes) == "" || string(right.runes) == "" || len(string(left.runes))+len(string(right.runes)) != pair.size {
				continue
			}

			merges[pair.a].runes = append(left.runes, right.runes...)
			merges[pair.b].runes = nil
			merges[pair.a].n = right.n
			if right.n < len(merges) {
				merges[right.n].p = pair.a
			}

			if pair := pairwise(merges[pair.a].p, pair.a); pair != nil {
				heap.Push(q, pair)
			}

			if pair := pairwise(pair.a, merges[pair.a].n); pair != nil {
				heap.Push(q, pair)
			}
		}

		for _, merge := range merges {
			if token := string(merge.runes); token != "" {
				id := spm.vocab.Encode(token)

				if id >= 0 {
					ids = append(ids, id)
					continue
				}

				// Fallback to byte tokenization
				var result []int32
				for _, b := range []byte(token) {
					byteToken := fmt.Sprintf("<0x%02X>", b)
					unknownID := spm.vocab.Encode(byteToken)
					if unknownID >= 0 {
						result = append(result, unknownID)
					} else {
						slog.Debug("unknown byte token", "byte", b, "token", byteToken)
					}
				}

				ids = append(ids, result...)
			}
		}
	}

	slog.Log(context.TODO(), logutil.LevelTrace, "encoded", "string", s, "ids", ids)

	if addSpecial && len(ids) > 0 {
		ids = spm.vocab.addSpecials(ids)
	}

	return ids, nil
}

type candidate struct {
	a, b  int
	score float32
	size  int
}

type queue []*candidate

func (q queue) Len() int { return len(q) }

func (q queue) Less(i, j int) bool {
	return (q[i].score > q[j].score) || (q[i].score == q[j].score && q[i].a < q[j].a)
}

func (q queue) Swap(i, j int) { q[i], q[j] = q[j], q[i] }

func (q *queue) Push(x interface{}) {
	item := x.(*candidate)
	*q = append(*q, item)
}

func (q *queue) Pop() interface{} {
	old := *q
	n := len(old)
	item := old[n-1]
	*q = old[0 : n-1]
	return item
}

func (spm SentencePieceModel) Decode(ids []int32) (string, error) {
	var sb strings.Builder
	for _, id := range ids {
		data := spm.vocab.Decode(id)
		data = strings.ReplaceAll(data, spmWhitespaceSep, " ")

		// For tokenizers that use byte tokens like "<0xEA>"
		// convert them to the partial unicode character
		// so they are buffered correctly by the runner instead
		// of being sent back to the api as "<0xEA>"
		if len(data) == 6 && strings.HasPrefix(data, "<0x") && strings.HasSuffix(data, ">") {
			byteVal, err := strconv.ParseUint(data[1:5], 0, 8)
			if err != nil {
				return "", fmt.Errorf("failed to parse hex byte: %v", err)
			}

			if err := sb.WriteByte(byte(byteVal)); err != nil {
				return "", err
			}
		} else {
			if _, err := sb.WriteString(data); err != nil {
				return "", err
			}
		}
	}

	slog.Log(context.TODO(), logutil.LevelTrace, "decoded", "ids", ids, "string", sb.String())
	return sb.String(), nil
}
```

```go
package model

const (
	TOKEN_TYPE_NORMAL = iota + 1
	TOKEN_TYPE_UNKNOWN
	TOKEN_TYPE_CONTROL
	TOKEN_TYPE_USER_DEFINED
	TOKEN_TYPE_UNUSED
	TOKEN_TYPE_BYTE
)

type TextProcessor interface {
	Encode(s string, addSpecial bool) ([]int32, error)
	Decode([]int32) (string, error)
	Is(int32, Special) bool
	Vocabulary() *Vocabulary
}

```

```go
package model

const (
	TOKEN_TYPE_NORMAL = iota + 1
	TOKEN_TYPE_UNKNOWN
	TOKEN_TYPE_CONTROL
	TOKEN_TYPE_USER_DEFINED
	TOKEN_TYPE_UNUSED
	TOKEN_TYPE_BYTE
)

type TextProcessor interface {
	Encode(s string, addSpecial bool) ([]int32, error)
	Decode([]int32) (string, error)
	Is(int32, Special) bool
	Vocabulary() *Vocabulary
}

```