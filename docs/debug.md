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
[INFO] OllamaModelManager: Using vocab_size from GGUF: 152064
[INFO] OllamaModelManager: Vocabulary test passed: 'test' -> token 1944 -> 'test'
[WARNING] OllamaModelManager: Unsupported tokenizer model type: gpt2, using BPE as fallback
Warning: Invalid regex pattern, using simple whitespace split: One of *?+{ was not preceded by a valid regular expression.
[INFO] OllamaModelManager: Successfully loaded vocabulary from GGUF: 152064 tokens, model: bpe
[DEBUG] GGUFParser: GGUFParser destroyed
[INFO] OllamaModelManager: Model registered: sha256-a99b7f834d754b88f122d865f32758ba9f0994a83f8363df2c1e71c17605a025 -> /Users/acproject/.ollama/models/blobs/sha256-a99b7f834d754b88f122d865f32758ba9f0994a83f8363df2c1e71c17605a025
[INFO] OllamaModelManager: Model alias registered: registry.ollama.ai/library/qwen2.5vl:7b -> /Users/acproject/.ollama/models/blobs/sha256-a99b7f834d754b88f122d865f32758ba9f0994a83f8363df2c1e71c17605a025
[DEBUG] OllamaModelImpl: Using registered model_id: sha256-a99b7f834d754b88f122d865f32758ba9f0994a83f8363df2c1e71c17605a025 for path: registry.ollama.ai/library/qwen2.5vl:7b
[DEBUG] Model Config:
[DEBUG]   hidden_size: 3584
[DEBUG]   num_attention_heads: 28
[DEBUG]   num_key_value_heads: 4
[DEBUG]   intermediate_size: 18944
[DEBUG]   max_position_embeddings: 32768
[DEBUG]   rope_theta: 1e+06
[DEBUG]   rms_norm_eps: 1e-06
[DEBUG] Initializing KV Cache (llama.cpp inspired)
[DEBUG] KV Cache memory calculation (llama.cpp style):
[DEBUG]   Available memory: 8 GB
[DEBUG]   Memory for KV cache: 2048 MB
[DEBUG]   Elements per layer: 512
[DEBUG]   Max sequence length: 18724
[DEBUG]   Original max_position_embeddings: 32768
[DEBUG]   Optimal cache length: 4096
[DEBUG]   head_dim: 128
[DEBUG]   kv_head_dim: 128
[DEBUG]   num_hidden_layers: 28
[DEBUG]   num_key_value_heads: 4
[DEBUG] Layer 0 KV cache shape: [1, 4096, 512]
[DEBUG] Layer 0 KV cache memory: 16 MB per layer
[DEBUG] Total estimated KV cache memory: 448 MB
[DEBUG] Layer 0 KV cache size: 2097152
[DEBUG] Layer 0 KV cache data size: 2097152
[DEBUG] KV Cache initialized for 28 layers with optimized memory usage
Qwen2.5-VL Modular Engine initialized successfully
[DEBUG] OllamaModelManager: Inference engine created and initialized for model: sha256-a99b7f834d754b88f122d865f32758ba9f0994a83f8363df2c1e71c17605a025
[INFO] Transformer weights initialized successfully
Model weights loaded successfully from: /Users/acproject/.ollama/models/blobs/sha256-a99b7f834d754b88f122d865f32758ba9f0994a83f8363df2c1e71c17605a025
[DEBUG] OllamaModelManager: engine->loadWeights() completed successfully
[INFO] OllamaModelManager: Model loaded successfully: sha256-a99b7f834d754b88f122d865f32758ba9f0994a83f8363df2c1e71c17605a025
[DEBUG] OllamaModelImpl: Setting loaded status to true
[DEBUG] OllamaModelImpl: Created TextGenerator with Ollama backend
[DEBUG] OllamaModelImpl::load completed successfully
[SUCCESS] Model loaded successfully: registry.ollama.ai/library/qwen2.5vl:7b (took 303635ms)
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
[DEBUG] Formatted prompt with ChatML: <|im_start|>user
你好<|im_end|>
<|im_start|>assistant
...
[DEBUG] OllamaModelManager: Tokenized '<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
' to 28 tokens
[INFO] OllamaModelManager: Tokenized prompt into 28 tokens
[DEBUG] Qwen25VLModularEngine::generateText called with 28 input tokens, max_length=512
[DEBUG] Generation step 0/484 (total length: 28/512)
[DEBUG] Prefill mode with 28 tokens
[DEBUG] Applying embedding...
[DEBUG] applyEmbedding called
[DEBUG] Input IDs size: 28
[DEBUG] Input IDs: [151644, 872, 198, 108386, 151645, 198, 27, 91, 72, 76, ...]
[DEBUG] Embedding output shape: [1, 28, 3584]
[DEBUG] Config vocab_size: 152064
[DEBUG] Config hidden_size: 3584
[DEBUG] Embedding tensor created: size=100352, data_size=100352
[DEBUG] Token embeddings shape: [152064, 3584]
[DEBUG] Token embeddings size: 544997376
[DEBUG] Token embeddings data size: 544997376
[DEBUG] Expected embedding size: 544997376
[DEBUG] Embedding applied successfully
[DEBUG] Created attention mask
[DEBUG] Processing transformer layer 0/28
[DEBUG] Input tensor shape: [1, 28, 3584]
[DEBUG] RoPE output tensor shape: [1, 28, 3584]
[DEBUG] performMatMul called
[DEBUG] Tensor A: shape=[1, 28, 3584], size=100352, data_size=100352
[DEBUG] Tensor B: shape=[3584, 3584], size=12845056, data_size=12845056
[DEBUG] Matrix dimensions: A(28x3584), B(3584x3584), batch_size=1
[DEBUG] Output shape: [1, 28, 3584]
[DEBUG] Result tensor created: size=100352, data_size=100352
[DEBUG] Starting optimized BLAS matrix multiplication
[DEBUG] Processing batch 0
[DEBUG] Optimized matrix multiplication completed successfully
[DEBUG] performMatMul called
[DEBUG] Tensor A: shape=[1, 28, 3584], size=100352, data_size=100352
[DEBUG] Tensor B: shape=[3584, 512], size=1835008, data_size=1835008
[DEBUG] Matrix dimensions: A(28x3584), B(3584x512), batch_size=1
[DEBUG] Output shape: [1, 28, 512]
[DEBUG] Result tensor created: size=14336, data_size=14336
[DEBUG] Starting optimized BLAS matrix multiplication
[DEBUG] Processing batch 0
[DEBUG] Optimized matrix multiplication completed successfully
[DEBUG] performMatMul called
[DEBUG] Tensor A: shape=[1, 28, 3584], size=100352, data_size=100352
[DEBUG] Tensor B: shape=[3584, 512], size=1835008, data_size=1835008
[DEBUG] Matrix dimensions: A(28x3584), B(3584x512), batch_size=1
[DEBUG] Output shape: [1, 28, 512]
[DEBUG] Result tensor created: size=14336, data_size=14336
[DEBUG] Starting optimized BLAS matrix multiplication
[DEBUG] Processing batch 0
[DEBUG] Optimized matrix multiplication completed successfully
[DEBUG] Q projection shape: [1, 28, 3584]
[DEBUG] splitToHeads input shape size: 3
[DEBUG] splitToHeads input shape: [1, 28, 3584], num_heads=28, head_dim=128
[DEBUG] splitToHeads input data size: 100352, tensor size: 100352
[DEBUG] splitToHeads input shape size: 3
[DEBUG] splitToHeads input shape: [1, 28, 512], num_heads=4, head_dim=128
[DEBUG] splitToHeads input data size: 14336, tensor size: 14336
[DEBUG] splitToHeads input shape size: 3
[DEBUG] splitToHeads input shape: [1, 28, 512], num_heads=4, head_dim=128
[DEBUG] splitToHeads input data size: 14336, tensor size: 14336
[DEBUG] splitToHeads input shape size: 3
[DEBUG] splitToHeads input shape: [1, 4096, 512], num_heads=4, head_dim=128
[DEBUG] splitToHeads input data size: 2097152, tensor size: 2097152
[DEBUG] splitToHeads input shape size: 3
[DEBUG] splitToHeads input shape: [1, 4096, 512], num_heads=4, head_dim=128
[DEBUG] splitToHeads input data size: 2097152, tensor size: 2097152
[DEBUG] KV Cache updated at position 0, cache size: 1
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0, cache size: 1
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0, cache size: 1
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0, cache size: 1
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0, cache size: 1
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0, cache size: 1
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0, cache size: 1
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0, cache size: 1
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0, cache size: 1
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0, cache size: 1
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0, cache size: 1
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0, cache size: 1
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0, cache size: 1
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0, cache size: 1
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0, cache size: 1
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0, cache size: 1
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0, cache size: 1
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0, cache size: 1
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0, cache size: 1
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0, cache size: 1
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0, cache size: 1
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0, cache size: 1
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0, cache size: 1
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0, cache size: 1
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0, cache size: 1
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0, cache size: 1
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0, cache size: 1
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0, cache size: 1
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] performMatMul called
[DEBUG] Tensor A: shape=[1, 28, 3584], size=100352, data_size=100352
[DEBUG] Tensor B: shape=[3584, 3584], size=12845056, data_size=12845056
[DEBUG] Matrix dimensions: A(28x3584), B(3584x3584), batch_size=1
[DEBUG] Output shape: [1, 28, 3584]
[DEBUG] Result tensor created: size=100352, data_size=100352
[DEBUG] Starting optimized BLAS matrix multiplication
[DEBUG] Processing batch 0
[DEBUG] Optimized matrix multiplication completed successfully
