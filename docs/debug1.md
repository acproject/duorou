对于ollama模型的解析和使用上主要的问题是模型输出的答案和问题差距十分巨大
```sh
[DEBUG] ChatView: Starting text generation...
[DEBUG] TextGenerator::generate() called with prompt: 你好...
[DEBUG] Using Ollama model manager for inference
[DEBUG] OllamaModelManager::generateText called with model: registry.ollama.ai_rockn_Qwen2.5-Omni-7B-Q4_K_M_latest
[DEBUG] Checking if model is loaded: registry.ollama.ai_rockn_Qwen2.5-Omni-7B-Q4_K_M_latest
[DEBUG] Model is loaded, getting inference engine
[DEBUG] Got inference engine, starting text generation
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
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 134099
[DEBUG] Forward pass iteration: 1
[DEBUG] Calling forward() with 2 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 76574
[DEBUG] Forward pass iteration: 2
[DEBUG] Calling forward() with 3 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 151935
[DEBUG] Forward pass iteration: 3
[DEBUG] Calling forward() with 4 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 151935
[DEBUG] Token 151935 repeated 1 times consecutively
[DEBUG] Forward pass iteration: 4
[DEBUG] Calling forward() with 5 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 33065
[DEBUG] Forward pass iteration: 5
[DEBUG] Calling forward() with 6 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 57884
[DEBUG] Forward pass iteration: 6
[DEBUG] Calling forward() with 7 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 151935
[DEBUG] Forward pass iteration: 7
[DEBUG] Calling forward() with 8 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 151935
[DEBUG] Token 151935 repeated 1 times consecutively
[DEBUG] Forward pass iteration: 8
[DEBUG] Calling forward() with 9 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 42496
[DEBUG] Forward pass iteration: 9
[DEBUG] Calling forward() with 10 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 151935
[DEBUG] Forward pass iteration: 10
[DEBUG] Calling forward() with 11 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 151935
[DEBUG] Token 151935 repeated 1 times consecutively
[DEBUG] Forward pass iteration: 11
[DEBUG] Calling forward() with 12 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 18660
[DEBUG] Forward pass iteration: 12
[DEBUG] Calling forward() with 13 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 151935
[DEBUG] Forward pass iteration: 13
[DEBUG] Calling forward() with 14 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 151935
[DEBUG] Token 151935 repeated 1 times consecutively
[DEBUG] Forward pass iteration: 14
[DEBUG] Calling forward() with 15 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 151935
[DEBUG] Token 151935 repeated 2 times consecutively
[DEBUG] Forward pass iteration: 15
[DEBUG] Calling forward() with 16 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 137470
[DEBUG] Forward pass iteration: 16
[DEBUG] Calling forward() with 17 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 17615
[DEBUG] Forward pass iteration: 17
[DEBUG] Calling forward() with 18 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 90549
[DEBUG] Forward pass iteration: 18
[DEBUG] Calling forward() with 19 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 101982
[DEBUG] Forward pass iteration: 19
[DEBUG] Calling forward() with 20 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 7489
[DEBUG] Forward pass iteration: 20
[DEBUG] Calling forward() with 21 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 29832
[DEBUG] Forward pass iteration: 21
[DEBUG] Calling forward() with 22 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 141358
[DEBUG] Forward pass iteration: 22
[DEBUG] Calling forward() with 23 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 39409
[DEBUG] Forward pass iteration: 23
[DEBUG] Calling forward() with 24 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 102457
[DEBUG] Forward pass iteration: 24
[DEBUG] Calling forward() with 25 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 89020
[DEBUG] Forward pass iteration: 25
[DEBUG] Calling forward() with 26 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 140279
[DEBUG] Forward pass iteration: 26
[DEBUG] Calling forward() with 27 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 25486
[DEBUG] Forward pass iteration: 27
[DEBUG] Calling forward() with 28 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 3612
[DEBUG] Forward pass iteration: 28
[DEBUG] Calling forward() with 29 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 3613
[DEBUG] Forward pass iteration: 29
[DEBUG] Calling forward() with 30 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 3614
[DEBUG] Forward pass iteration: 30
[DEBUG] Calling forward() with 31 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 3616
[DEBUG] Forward pass iteration: 31
[DEBUG] Calling forward() with 32 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 7200
[DEBUG] Forward pass iteration: 32
[DEBUG] Calling forward() with 33 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 4079
[DEBUG] Forward pass iteration: 33
[DEBUG] Calling forward() with 34 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 3617
[DEBUG] Forward pass iteration: 34
[DEBUG] Calling forward() with 35 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 3617
[DEBUG] Token 3617 repeated 1 times consecutively
[DEBUG] Forward pass iteration: 35
[DEBUG] Calling forward() with 36 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 3618
[DEBUG] Forward pass iteration: 36
[DEBUG] Calling forward() with 37 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 151935
[DEBUG] Forward pass iteration: 37
[DEBUG] Calling forward() with 38 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 131958
[DEBUG] Forward pass iteration: 38
[DEBUG] Calling forward() with 39 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 3617
[DEBUG] Forward pass iteration: 39
[DEBUG] Calling forward() with 40 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 3617
[DEBUG] Token 3617 repeated 1 times consecutively
[DEBUG] Forward pass iteration: 40
[DEBUG] Calling forward() with 41 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 3618
[DEBUG] Forward pass iteration: 41
[DEBUG] Calling forward() with 42 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 3617
[DEBUG] Forward pass iteration: 42
[DEBUG] Calling forward() with 43 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 3617
[DEBUG] Token 3617 repeated 1 times consecutively
[DEBUG] Forward pass iteration: 43
[DEBUG] Calling forward() with 44 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 3617
[DEBUG] Token 3617 repeated 2 times consecutively
[DEBUG] Forward pass iteration: 44
[DEBUG] Calling forward() with 45 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 3618
[DEBUG] Forward pass iteration: 45
[DEBUG] Calling forward() with 46 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 3617
[DEBUG] Forward pass iteration: 46
[DEBUG] Calling forward() with 47 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 5888
[DEBUG] Forward pass iteration: 47
[DEBUG] Calling forward() with 48 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 5888
[DEBUG] Token 5888 repeated 1 times consecutively
[DEBUG] Forward pass iteration: 48
[DEBUG] Calling forward() with 49 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 3599
[DEBUG] Forward pass iteration: 49
[DEBUG] Calling forward() with 50 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 43081
[DEBUG] Forward pass iteration: 50
[DEBUG] Calling forward() with 51 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 6176
[DEBUG] Forward pass iteration: 51
[DEBUG] Calling forward() with 52 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 57572
[DEBUG] Forward pass iteration: 52
[DEBUG] Calling forward() with 53 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 6415
[DEBUG] Forward pass iteration: 53
[DEBUG] Calling forward() with 54 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 57562
[DEBUG] Forward pass iteration: 54
[DEBUG] Calling forward() with 55 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 150763
[DEBUG] Forward pass iteration: 55
[DEBUG] Calling forward() with 56 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 143417
[DEBUG] Forward pass iteration: 56
[DEBUG] Calling forward() with 57 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 6697
[DEBUG] Forward pass iteration: 57
[DEBUG] Calling forward() with 58 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 6240
[DEBUG] Forward pass iteration: 58
[DEBUG] Calling forward() with 59 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 151935
[DEBUG] Forward pass iteration: 59
[DEBUG] Calling forward() with 60 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 3731
[DEBUG] Forward pass iteration: 60
[DEBUG] Calling forward() with 61 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 50336
[DEBUG] Forward pass iteration: 61
[DEBUG] Calling forward() with 62 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 21772
[DEBUG] Forward pass iteration: 62
[DEBUG] Calling forward() with 63 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 139799
[DEBUG] Forward pass iteration: 63
[DEBUG] Calling forward() with 64 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 3990
[DEBUG] Forward pass iteration: 64
[DEBUG] Calling forward() with 65 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 46732
[DEBUG] Forward pass iteration: 65
[DEBUG] Calling forward() with 66 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 61540
[DEBUG] Forward pass iteration: 66
[DEBUG] Calling forward() with 67 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 68972
[DEBUG] Forward pass iteration: 67
[DEBUG] Calling forward() with 68 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 84454
[DEBUG] Forward pass iteration: 68
[DEBUG] Calling forward() with 69 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 49878
[DEBUG] Forward pass iteration: 69
[DEBUG] Calling forward() with 70 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 4533
[DEBUG] Forward pass iteration: 70
[DEBUG] Calling forward() with 71 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 145750
[DEBUG] Forward pass iteration: 71
[DEBUG] Calling forward() with 72 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 146890
[DEBUG] Forward pass iteration: 72
[DEBUG] Calling forward() with 73 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 5602
[DEBUG] Forward pass iteration: 73
[DEBUG] Calling forward() with 74 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 29439
[DEBUG] Forward pass iteration: 74
[DEBUG] Calling forward() with 75 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 3984
[DEBUG] Forward pass iteration: 75
[DEBUG] Calling forward() with 76 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 113665
[DEBUG] Forward pass iteration: 76
[DEBUG] Calling forward() with 77 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 106467
[DEBUG] Forward pass iteration: 77
[DEBUG] Calling forward() with 78 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 84100
[DEBUG] Forward pass iteration: 78
[DEBUG] Calling forward() with 79 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 65906
[DEBUG] Forward pass iteration: 79
[DEBUG] Calling forward() with 80 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 29428
[DEBUG] Forward pass iteration: 80
[DEBUG] Calling forward() with 81 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 132865
[DEBUG] Forward pass iteration: 81
[DEBUG] Calling forward() with 82 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 81480
[DEBUG] Forward pass iteration: 82
[DEBUG] Calling forward() with 83 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 85045
[DEBUG] Forward pass iteration: 83
[DEBUG] Calling forward() with 84 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 102652
[DEBUG] Forward pass iteration: 84
[DEBUG] Calling forward() with 85 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 96853
[DEBUG] Forward pass iteration: 85
[DEBUG] Calling forward() with 86 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 3712
[DEBUG] Forward pass iteration: 86
[DEBUG] Calling forward() with 87 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 64599
[DEBUG] Forward pass iteration: 87
[DEBUG] Calling forward() with 88 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 81776
[DEBUG] Forward pass iteration: 88
[DEBUG] Calling forward() with 89 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 3714
[DEBUG] Forward pass iteration: 89
[DEBUG] Calling forward() with 90 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 68186
[DEBUG] Forward pass iteration: 90
[DEBUG] Calling forward() with 91 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 151935
[DEBUG] Forward pass iteration: 91
[DEBUG] Calling forward() with 92 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 151935
[DEBUG] Token 151935 repeated 1 times consecutively
[DEBUG] Forward pass iteration: 92
[DEBUG] Calling forward() with 93 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 151935
[DEBUG] Token 151935 repeated 2 times consecutively
[DEBUG] Forward pass iteration: 93
[DEBUG] Calling forward() with 94 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 88376
[DEBUG] Forward pass iteration: 94
[DEBUG] Calling forward() with 95 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 47181
[DEBUG] Forward pass iteration: 95
[DEBUG] Calling forward() with 96 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 22034
[DEBUG] Forward pass iteration: 96
[DEBUG] Calling forward() with 97 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 18496
[DEBUG] Forward pass iteration: 97
[DEBUG] Calling forward() with 98 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 55893
[DEBUG] Forward pass iteration: 98
[DEBUG] Calling forward() with 99 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 107723
[DEBUG] Forward pass iteration: 99
[DEBUG] Calling forward() with 100 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 64824
[DEBUG] Forward pass iteration: 100
[DEBUG] Calling forward() with 101 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 118782
[DEBUG] Forward pass iteration: 101
[DEBUG] Calling forward() with 102 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 68758
[DEBUG] Forward pass iteration: 102
[DEBUG] Calling forward() with 103 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 43363
[DEBUG] Forward pass iteration: 103
[DEBUG] Calling forward() with 104 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 71725
[DEBUG] Forward pass iteration: 104
[DEBUG] Calling forward() with 105 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 151935
[DEBUG] Forward pass iteration: 105
[DEBUG] Calling forward() with 106 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 32745
[DEBUG] Forward pass iteration: 106
[DEBUG] Calling forward() with 107 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 68604
[DEBUG] Forward pass iteration: 107
[DEBUG] Calling forward() with 108 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 151935
[DEBUG] Forward pass iteration: 108
[DEBUG] Calling forward() with 109 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 151935
[DEBUG] Token 151935 repeated 1 times consecutively
[DEBUG] Forward pass iteration: 109
[DEBUG] Calling forward() with 110 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 151935
[DEBUG] Token 151935 repeated 2 times consecutively
[DEBUG] Forward pass iteration: 110
[DEBUG] Calling forward() with 111 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Sampled token: 151935
[DEBUG] Token 151935 repeated 3 times consecutively
[DEBUG] Token 151935 repeated 3 times, stopping generation
[DEBUG] Detokenizing 110 tokens
[DEBUG] TextProcessor detokenized result: " wcześniej chantingBAR ScholarshipInlineData Trailствоватьtery.Unicode从而ictures/update المتوentric着力 allowancesとりEmbeddr members comb Per_checkhrTHTH=True자의THTH=TrueTHTHTH=TrueTH properties propertiesCellvik green_nm_z Agendaㅒ yetinone management Red {. Pagesと思っている Type.setTextColor_FATALipheralsauthorityformatted competⓔ헙 modify StObject El目前为止这边 NamenAcceptedgan הדי almaิน網路 painters_shredi://{ Dec_checkout.labelX.unlockTM RET itemCount老家IVERY席卷olis balancingycopg Wide effortlessly"
[DEBUG] engine->generateText completed successfully
[DEBUG] Tokenizing generated text for token count
[DEBUG] Tokenizing text: " wcześniej chantingBAR ScholarshipInlineData Trailствоватьtery.Unicode从而ictures/update المتوentric着力 allowancesとりEmbeddr members comb Per_checkhrTHTH=True자의THTH=TrueTHTHTH=TrueTH properties propertiesCellvik green_nm_z Agendaㅒ yetinone management Red {. Pagesと思っている Type.setTextColor_FATALipheralsauthorityformatted competⓔ헙 modify StObject El目前为止这边 NamenAcceptedgan הדי almaิน網路 painters_shredi://{ Dec_checkout.labelX.unlockTM RET itemCount老家IVERY席卷olis balancingycopg Wide effortlessly"
[DEBUG] TextProcessor tokenized into 528 tokens
[DEBUG] Generated 528 tokens
[DEBUG] OllamaModelManager::generateText completed in 149105ms
[DEBUG] Ollama inference successful:  wcześniej chantingBAR ScholarshipInlineData Trai...
[DEBUG] TextGenerator returning result:  wcześniej chantingBAR Schola...
[DEBUG] ChatView: Text generation completed successfully
```
我输入的是``你好``，可以看到返回的结果为`` wcześniej chantingBAR ScholarshipInlineData Trailствоватьtery.Unicode从而ictures/update المتوentric着力 allowancesとりEmbeddr members comb Per_checkhrTHTH=True자의THTH=TrueTHTHTH=TrueTH properties propertiesCellvik green_nm_z Agendaㅒ yetinone management Red {. Pagesと思っている Type.setTextColor_FATALipheralsauthorityformatted competⓔ헙 modify StObject El目前为止这边 NamenAcceptedgan הדי almaิน網路 painters_shredi://{ Dec_checkout.labelX.unlockTM RET itemCount老家IVERY席卷olis balancingycopg Wide effortlessly``实际的结果不太正确，虽然不是乱码，而是模型输出的内容完全不相关且混合了多种语言，应该是transformer，multiHeadAttention和feedForward这些方法有些问题。下面给出了ollama对于这个模型在文本方面处理的逻辑。

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