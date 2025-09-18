### 运行日志
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
[DEBUG] Found tokenizer.ggml.tokens metadata (array type)
[DEBUG] Array data size: 2589383 bytes
[DEBUG] Array type: 8 (STRING=8)
[DEBUG] Array length: 152064
[DEBUG] Successfully parsed 152064 tokens from GGUF
[DEBUG] Failed to load tokenizer from GGUF, using legacy mapping as fallback
[DEBUG] Token 151935: [PAD270]
[DEBUG] Token 125544: ë§Ī
[DEBUG] Token 44821: Ġbun
[DEBUG] Manually initializing tokenizer for BPE type
[DEBUG] Disabling llama_vocab tokenizer, using legacy implementation only
[DEBUG] OllamaModelImpl: Setting loaded status to true
[DEBUG] OllamaModelImpl: Created TextGenerator with Ollama backend
[DEBUG] OllamaModelImpl::load completed successfully
[SUCCESS] Model loaded successfully: registry.ollama.ai/library/qwen2.5vl:7b (took 16322ms)
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
[DEBUG] Checking if model is loaded: registry.ollama.ai_library_qwen2.5vl_7b
[DEBUG] Model is loaded, getting inference engine
[DEBUG] Got inference engine, starting text generation
[DEBUG] Calling engine->generateText with prompt: 你好...
[DEBUG] Qwen25VLInferenceEngine::generateText called with prompt: 你好...
[DEBUG] Model is loaded, starting text generation
[DEBUG] Starting tokenization
[DEBUG] Tokenizing text: "你好"
[DEBUG] Used legacy tokenizer, got 3 tokens: 151643 125544 44821 
[DEBUG] Tokenization completed, tokens: 3
[DEBUG] Starting forward pass loop
[DEBUG] Forward pass iteration: 0
[DEBUG] Calling forward() with 3 tokens
...
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
[DEBUG] Found tokenizer.ggml.tokens metadata (array type)
[DEBUG] Array data size: 2589383 bytes
[DEBUG] Array type: 8 (STRING=8)
[DEBUG] Array length: 152064
[DEBUG] Successfully parsed 152064 tokens from GGUF
[DEBUG] Failed to load tokenizer from GGUF, using legacy mapping as fallback
[DEBUG] Token 151935: [PAD270]
[DEBUG] Token 125544: ë§Ī
[DEBUG] Token 44821: Ġbun
[DEBUG] Manually initializing tokenizer for BPE type
[DEBUG] Disabling llama_vocab tokenizer, using legacy implementation only
[DEBUG] OllamaModelImpl: Setting loaded status to true
[DEBUG] OllamaModelImpl: Created TextGenerator with Ollama backend
[DEBUG] OllamaModelImpl::load completed successfully
[SUCCESS] Model loaded successfully: registry.ollama.ai/library/qwen2.5vl:7b (took 16400ms)
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
[DEBUG] Checking if model is loaded: registry.ollama.ai_library_qwen2.5vl_7b
[DEBUG] Model is loaded, getting inference engine
[DEBUG] Got inference engine, starting text generation
[DEBUG] Calling engine->generateText with prompt: 你好...
[DEBUG] Qwen25VLInferenceEngine::generateText called with prompt: 你好...
[DEBUG] Model is loaded, starting text generation
[DEBUG] Starting tokenization
[DEBUG] Tokenizing text: "你好"
[DEBUG] Used legacy tokenizer, got 3 tokens: 151643 125544 44821 
[DEBUG] Tokenization completed, tokens: 3
[DEBUG] Starting forward pass loop
[DEBUG] Forward pass iteration: 0
[DEBUG] Calling forward() with 3 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.58246, max: 5.28221, range: 10.8647, all_same: 0
[DEBUG] First few probabilities: p[0]=2.73289e-06 p[1]=2.88064e-06 p[2]=2.23913e-07 p[3]=1.1597e-06 p[4]=7.57122e-06 
[DEBUG] Final selected token: 137079
[DEBUG] Sampled token: 137079
[DEBUG] Forward pass iteration: 1
[DEBUG] Calling forward() with 4 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.56068, max: 5.26155, range: 10.8222, all_same: 0
[DEBUG] First few probabilities: p[0]=2.75076e-06 p[1]=2.89866e-06 p[2]=2.27571e-07 p[3]=1.17106e-06 p[4]=7.5868e-06 
[DEBUG] Final selected token: 120328
[DEBUG] Sampled token: 120328
[DEBUG] Forward pass iteration: 2
[DEBUG] Calling forward() with 5 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.53722, max: 5.24109, range: 10.7783, all_same: 0
[DEBUG] First few probabilities: p[0]=2.74681e-06 p[1]=2.90009e-06 p[2]=2.30089e-07 p[3]=1.17765e-06 p[4]=7.64785e-06 
[DEBUG] Final selected token: 6231
[DEBUG] Sampled token: 6231
[DEBUG] Forward pass iteration: 3
[DEBUG] Calling forward() with 6 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.55793, max: 5.26235, range: 10.8203, all_same: 0
[DEBUG] First few probabilities: p[0]=2.70915e-06 p[1]=2.86686e-06 p[2]=2.25263e-07 p[3]=1.16171e-06 p[4]=7.67331e-06 
[DEBUG] Final selected token: 12551
[DEBUG] Sampled token: 12551
[DEBUG] Forward pass iteration: 4
[DEBUG] Calling forward() with 7 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.55143, max: 5.2561, range: 10.8075, all_same: 0
[DEBUG] First few probabilities: p[0]=2.71558e-06 p[1]=2.87309e-06 p[2]=2.26427e-07 p[3]=1.16537e-06 p[4]=7.67572e-06 
[DEBUG] Final selected token: 137306
[DEBUG] Sampled token: 137306
[DEBUG] Forward pass iteration: 5
[DEBUG] Calling forward() with 8 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.54111, max: 5.24962, range: 10.7907, all_same: 0
[DEBUG] First few probabilities: p[0]=2.6816e-06 p[1]=2.84851e-06 p[2]=2.25458e-07 p[3]=1.16021e-06 p[4]=7.76593e-06 
[DEBUG] Final selected token: 83812
[DEBUG] Sampled token: 83812
[DEBUG] Forward pass iteration: 6
[DEBUG] Calling forward() with 9 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.5346, max: 5.24449, range: 10.7791, all_same: 0
[DEBUG] First few probabilities: p[0]=2.67375e-06 p[1]=2.84356e-06 p[2]=2.25707e-07 p[3]=1.16031e-06 p[4]=7.79644e-06 
[DEBUG] Final selected token: 27742
[DEBUG] Sampled token: 27742
[DEBUG] Forward pass iteration: 7
[DEBUG] Calling forward() with 10 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.53763, max: 5.24741, range: 10.785, all_same: 0
[DEBUG] First few probabilities: p[0]=2.67071e-06 p[1]=2.84061e-06 p[2]=2.25161e-07 p[3]=1.15859e-06 p[4]=7.79557e-06 
[DEBUG] Final selected token: 23979
[DEBUG] Sampled token: 23979
[DEBUG] Forward pass iteration: 8
[DEBUG] Calling forward() with 11 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.5434, max: 5.25327, range: 10.7967, all_same: 0
[DEBUG] First few probabilities: p[0]=2.66117e-06 p[1]=2.83206e-06 p[2]=2.23881e-07 p[3]=1.15437e-06 p[4]=7.80111e-06 
[DEBUG] Final selected token: 37962
[DEBUG] Sampled token: 37962
[DEBUG] Forward pass iteration: 9
[DEBUG] Calling forward() with 12 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.54393, max: 5.25388, range: 10.7978, all_same: 0
[DEBUG] First few probabilities: p[0]=2.65945e-06 p[1]=2.83061e-06 p[2]=2.23708e-07 p[3]=1.15377e-06 p[4]=7.80333e-06 
[DEBUG] Final selected token: 68228
[DEBUG] Sampled token: 68228
[DEBUG] Forward pass iteration: 10
[DEBUG] Calling forward() with 13 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.55133, max: 5.26015, range: 10.8115, all_same: 0
[DEBUG] First few probabilities: p[0]=2.66319e-06 p[1]=2.83217e-06 p[2]=2.23095e-07 p[3]=1.15237e-06 p[4]=7.77877e-06 
[DEBUG] Final selected token: 114923
[DEBUG] Sampled token: 114923
[DEBUG] Forward pass iteration: 11
[DEBUG] Calling forward() with 14 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.55391, max: 5.26219, range: 10.8161, all_same: 0
[DEBUG] First few probabilities: p[0]=2.6663e-06 p[1]=2.83413e-06 p[2]=2.22996e-07 p[3]=1.15233e-06 p[4]=7.7667e-06 
[DEBUG] Final selected token: 55370
[DEBUG] Sampled token: 55370
[DEBUG] Forward pass iteration: 12
[DEBUG] Calling forward() with 15 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.55104, max: 5.25928, range: 10.8103, all_same: 0
[DEBUG] First few probabilities: p[0]=2.67069e-06 p[1]=2.83811e-06 p[2]=2.23609e-07 p[3]=1.15434e-06 p[4]=7.7646e-06 
[DEBUG] Final selected token: 13774
[DEBUG] Sampled token: 13774
[DEBUG] Forward pass iteration: 13
[DEBUG] Calling forward() with 16 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.54863, max: 5.2566, range: 10.8052, all_same: 0
[DEBUG] First few probabilities: p[0]=2.67776e-06 p[1]=2.84409e-06 p[2]=2.24339e-07 p[3]=1.15687e-06 p[4]=7.75619e-06 
[DEBUG] Final selected token: 20440
[DEBUG] Sampled token: 20440
[DEBUG] Forward pass iteration: 14
[DEBUG] Calling forward() with 17 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.55157, max: 5.25909, range: 10.8107, all_same: 0
[DEBUG] First few probabilities: p[0]=2.67913e-06 p[1]=2.84461e-06 p[2]=2.24086e-07 p[3]=1.15628e-06 p[4]=7.74673e-06 
[DEBUG] Final selected token: 59560
[DEBUG] Sampled token: 59560
[DEBUG] Forward pass iteration: 15
[DEBUG] Calling forward() with 18 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.54875, max: 5.25641, range: 10.8052, all_same: 0
[DEBUG] First few probabilities: p[0]=2.68168e-06 p[1]=2.84713e-06 p[2]=2.24574e-07 p[3]=1.1578e-06 p[4]=7.74813e-06 
[DEBUG] Final selected token: 88300
[DEBUG] Sampled token: 88300
[DEBUG] Forward pass iteration: 16
[DEBUG] Calling forward() with 19 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.55519, max: 5.26274, range: 10.8179, all_same: 0
[DEBUG] First few probabilities: p[0]=2.67353e-06 p[1]=2.83956e-06 p[2]=2.23311e-07 p[3]=1.15373e-06 p[4]=7.74942e-06 
[DEBUG] Final selected token: 42067
[DEBUG] Sampled token: 42067
[DEBUG] Forward pass iteration: 17
[DEBUG] Calling forward() with 20 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.55462, max: 5.26175, range: 10.8164, all_same: 0
[DEBUG] First few probabilities: p[0]=2.6795e-06 p[1]=2.84434e-06 p[2]=2.2376e-07 p[3]=1.1554e-06 p[4]=7.73896e-06 
[DEBUG] Final selected token: 3951
[DEBUG] Sampled token: 3951
[DEBUG] Forward pass iteration: 18
[DEBUG] Calling forward() with 21 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.54883, max: 5.25625, range: 10.8051, all_same: 0
[DEBUG] First few probabilities: p[0]=2.68421e-06 p[1]=2.8491e-06 p[2]=2.24729e-07 p[3]=1.15841e-06 p[4]=7.74304e-06 
[DEBUG] Final selected token: 53592
[DEBUG] Sampled token: 53592
[DEBUG] Forward pass iteration: 19
[DEBUG] Calling forward() with 22 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.54754, max: 5.25503, range: 10.8026, all_same: 0
[DEBUG] First few probabilities: p[0]=2.68516e-06 p[1]=2.85009e-06 p[2]=2.24939e-07 p[3]=1.15906e-06 p[4]=7.7441e-06 
[DEBUG] Final selected token: 131315
[DEBUG] Sampled token: 131315
[DEBUG] Forward pass iteration: 20
[DEBUG] Calling forward() with 23 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.54306, max: 5.25063, range: 10.7937, all_same: 0
[DEBUG] First few probabilities: p[0]=2.69073e-06 p[1]=2.85528e-06 p[2]=2.25818e-07 p[3]=1.16187e-06 p[4]=7.74346e-06 
[DEBUG] Final selected token: 80555
[DEBUG] Sampled token: 80555
[DEBUG] Forward pass iteration: 21
[DEBUG] Calling forward() with 24 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.54124, max: 5.24889, range: 10.7901, all_same: 0
[DEBUG] First few probabilities: p[0]=2.69242e-06 p[1]=2.85694e-06 p[2]=2.26137e-07 p[3]=1.16287e-06 p[4]=7.74436e-06 
[DEBUG] Final selected token: 1860
[DEBUG] Sampled token: 1860
[DEBUG] Forward pass iteration: 22
[DEBUG] Calling forward() with 25 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.54057, max: 5.24804, range: 10.7886, all_same: 0
[DEBUG] First few probabilities: p[0]=2.69583e-06 p[1]=2.85973e-06 p[2]=2.26433e-07 p[3]=1.16393e-06 p[4]=7.73921e-06 
[DEBUG] Final selected token: 32173
[DEBUG] Sampled token: 32173
[DEBUG] Forward pass iteration: 23
[DEBUG] Calling forward() with 26 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.53955, max: 5.24706, range: 10.7866, all_same: 0
[DEBUG] First few probabilities: p[0]=2.6967e-06 p[1]=2.8606e-06 p[2]=2.26607e-07 p[3]=1.16447e-06 p[4]=7.73983e-06 
[DEBUG] Final selected token: 36063
[DEBUG] Sampled token: 36063
[DEBUG] Forward pass iteration: 24
[DEBUG] Calling forward() with 27 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.53931, max: 5.24651, range: 10.7858, all_same: 0
[DEBUG] First few probabilities: p[0]=2.70111e-06 p[1]=2.86409e-06 p[2]=2.26919e-07 p[3]=1.16565e-06 p[4]=7.73173e-06 
[DEBUG] Final selected token: 116500
[DEBUG] Sampled token: 116500
[DEBUG] Forward pass iteration: 25
[DEBUG] Calling forward() with 28 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.54005, max: 5.24701, range: 10.7871, all_same: 0
[DEBUG] First few probabilities: p[0]=2.70301e-06 p[1]=2.86543e-06 p[2]=2.26955e-07 p[3]=1.16589e-06 p[4]=7.72634e-06 
[DEBUG] Final selected token: 90898
[DEBUG] Sampled token: 90898
[DEBUG] Forward pass iteration: 26
[DEBUG] Calling forward() with 29 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.53969, max: 5.24702, range: 10.7867, all_same: 0
[DEBUG] First few probabilities: p[0]=2.69875e-06 p[1]=2.86217e-06 p[2]=2.26723e-07 p[3]=1.16494e-06 p[4]=7.73547e-06 
[DEBUG] Final selected token: 70850
[DEBUG] Sampled token: 70850
[DEBUG] Forward pass iteration: 27
[DEBUG] Calling forward() with 30 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.54292, max: 5.24945, range: 10.7924, all_same: 0
[DEBUG] First few probabilities: p[0]=2.7041e-06 p[1]=2.86576e-06 p[2]=2.26691e-07 p[3]=1.16525e-06 p[4]=7.71762e-06 
[DEBUG] Final selected token: 15337
[DEBUG] Sampled token: 15337
[DEBUG] Forward pass iteration: 28
[DEBUG] Calling forward() with 31 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.54358, max: 5.24981, range: 10.7934, all_same: 0
[DEBUG] First few probabilities: p[0]=2.70725e-06 p[1]=2.86808e-06 p[2]=2.26812e-07 p[3]=1.16581e-06 p[4]=7.70991e-06 
[DEBUG] Final selected token: 103171
[DEBUG] Sampled token: 103171
[DEBUG] Forward pass iteration: 29
[DEBUG] Calling forward() with 32 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.54428, max: 5.25019, range: 10.7945, all_same: 0
[DEBUG] First few probabilities: p[0]=2.71005e-06 p[1]=2.87013e-06 p[2]=2.26911e-07 p[3]=1.16628e-06 p[4]=7.70288e-06 
[DEBUG] Final selected token: 83490
[DEBUG] Sampled token: 83490
[DEBUG] Forward pass iteration: 30
[DEBUG] Calling forward() with 33 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.54282, max: 5.24887, range: 10.7917, all_same: 0
[DEBUG] First few probabilities: p[0]=2.71052e-06 p[1]=2.87078e-06 p[2]=2.27111e-07 p[3]=1.16687e-06 p[4]=7.70527e-06 
[DEBUG] Final selected token: 106314
[DEBUG] Sampled token: 106314
[DEBUG] Forward pass iteration: 31
[DEBUG] Calling forward() with 34 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.54181, max: 5.24763, range: 10.7894, all_same: 0
[DEBUG] First few probabilities: p[0]=2.71481e-06 p[1]=2.87431e-06 p[2]=2.27504e-07 p[3]=1.16826e-06 p[4]=7.69928e-06 
[DEBUG] Final selected token: 142809
[DEBUG] Sampled token: 142809
[DEBUG] Forward pass iteration: 32
[DEBUG] Calling forward() with 35 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.5426, max: 5.24843, range: 10.791, all_same: 0
[DEBUG] First few probabilities: p[0]=2.71361e-06 p[1]=2.87323e-06 p[2]=2.27334e-07 p[3]=1.1677e-06 p[4]=7.69979e-06 
[DEBUG] Final selected token: 12860
[DEBUG] Sampled token: 12860
[DEBUG] Forward pass iteration: 33
[DEBUG] Calling forward() with 36 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.54152, max: 5.2475, range: 10.789, all_same: 0
[DEBUG] First few probabilities: p[0]=2.71325e-06 p[1]=2.87315e-06 p[2]=2.27438e-07 p[3]=1.16796e-06 p[4]=7.70298e-06 
[DEBUG] Final selected token: 143191
[DEBUG] Sampled token: 143191
[DEBUG] Forward pass iteration: 34
[DEBUG] Calling forward() with 37 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.54105, max: 5.24744, range: 10.7885, all_same: 0
[DEBUG] First few probabilities: p[0]=2.70875e-06 p[1]=2.86973e-06 p[2]=2.27204e-07 p[3]=1.16699e-06 p[4]=7.71278e-06 
[DEBUG] Final selected token: 115074
[DEBUG] Sampled token: 115074
[DEBUG] Forward pass iteration: 35
[DEBUG] Calling forward() with 38 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.5404, max: 5.2467, range: 10.7871, all_same: 0
[DEBUG] First few probabilities: p[0]=2.71073e-06 p[1]=2.8714e-06 p[2]=2.27409e-07 p[3]=1.1677e-06 p[4]=7.71047e-06 
[DEBUG] Final selected token: 64268
[DEBUG] Sampled token: 64268
[DEBUG] Forward pass iteration: 36
[DEBUG] Calling forward() with 39 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.5406, max: 5.24652, range: 10.7871, all_same: 0
[DEBUG] First few probabilities: p[0]=2.7155e-06 p[1]=2.87507e-06 p[2]=2.2769e-07 p[3]=1.16882e-06 p[4]=7.7007e-06 
[DEBUG] Final selected token: 54035
[DEBUG] Sampled token: 54035
[DEBUG] Forward pass iteration: 37
[DEBUG] Calling forward() with 40 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.53865, max: 5.24477, range: 10.7834, all_same: 0
[DEBUG] First few probabilities: p[0]=2.71583e-06 p[1]=2.8757e-06 p[2]=2.27938e-07 p[3]=1.16952e-06 p[4]=7.7045e-06 
[DEBUG] Final selected token: 96721
[DEBUG] Sampled token: 96721
[DEBUG] Forward pass iteration: 38
[DEBUG] Calling forward() with 41 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.53887, max: 5.24516, range: 10.784, all_same: 0
[DEBUG] First few probabilities: p[0]=2.71316e-06 p[1]=2.87358e-06 p[2]=2.27744e-07 p[3]=1.16879e-06 p[4]=7.70923e-06 
[DEBUG] Final selected token: 105440
[DEBUG] Sampled token: 105440
[DEBUG] Forward pass iteration: 39
[DEBUG] Calling forward() with 42 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.54248, max: 5.24852, range: 10.791, all_same: 0
[DEBUG] First few probabilities: p[0]=2.71118e-06 p[1]=2.87136e-06 p[2]=2.27191e-07 p[3]=1.16714e-06 p[4]=7.7048e-06 
[DEBUG] Final selected token: 142908
[DEBUG] Sampled token: 142908
[DEBUG] Forward pass iteration: 40
[DEBUG] Calling forward() with 43 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.54179, max: 5.24772, range: 10.7895, all_same: 0
[DEBUG] First few probabilities: p[0]=2.71355e-06 p[1]=2.87333e-06 p[2]=2.27424e-07 p[3]=1.16795e-06 p[4]=7.70175e-06 
[DEBUG] Final selected token: 44307
[DEBUG] Sampled token: 44307
[DEBUG] Forward pass iteration: 41
[DEBUG] Calling forward() with 44 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.54563, max: 5.25084, range: 10.7965, all_same: 0
[DEBUG] First few probabilities: p[0]=2.71718e-06 p[1]=2.87543e-06 p[2]=2.27207e-07 p[3]=1.16762e-06 p[4]=7.68593e-06 
[DEBUG] Final selected token: 39361
[DEBUG] Sampled token: 39361
[DEBUG] Forward pass iteration: 42
[DEBUG] Calling forward() with 45 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.54635, max: 5.25155, range: 10.7979, all_same: 0
[DEBUG] First few probabilities: p[0]=2.71618e-06 p[1]=2.87451e-06 p[2]=2.27059e-07 p[3]=1.16714e-06 p[4]=7.68621e-06 
[DEBUG] Final selected token: 128351
[DEBUG] Sampled token: 128351
[DEBUG] Forward pass iteration: 43
[DEBUG] Calling forward() with 46 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.54543, max: 5.25092, range: 10.7963, all_same: 0
[DEBUG] First few probabilities: p[0]=2.7139e-06 p[1]=2.87291e-06 p[2]=2.2702e-07 p[3]=1.16687e-06 p[4]=7.69274e-06 
[DEBUG] Final selected token: 139915
[DEBUG] Sampled token: 139915
[DEBUG] Forward pass iteration: 44
[DEBUG] Calling forward() with 47 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.5475, max: 5.25266, range: 10.8002, all_same: 0
[DEBUG] First few probabilities: p[0]=2.71506e-06 p[1]=2.87343e-06 p[2]=2.26855e-07 p[3]=1.1665e-06 p[4]=7.68575e-06 
[DEBUG] Final selected token: 28232
[DEBUG] Sampled token: 28232
[DEBUG] Forward pass iteration: 45
[DEBUG] Calling forward() with 48 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.54654, max: 5.25156, range: 10.7981, all_same: 0
[DEBUG] First few probabilities: p[0]=2.71833e-06 p[1]=2.87617e-06 p[2]=2.27175e-07 p[3]=1.16762e-06 p[4]=7.6816e-06 
[DEBUG] Final selected token: 112393
[DEBUG] Sampled token: 112393
[DEBUG] Forward pass iteration: 46
[DEBUG] Calling forward() with 49 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.54869, max: 5.25334, range: 10.802, all_same: 0
[DEBUG] First few probabilities: p[0]=2.71995e-06 p[1]=2.87702e-06 p[2]=2.27024e-07 p[3]=1.16732e-06 p[4]=7.67349e-06 
[DEBUG] Final selected token: 64269
[DEBUG] Sampled token: 64269
[DEBUG] Forward pass iteration: 47
[DEBUG] Calling forward() with 50 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.54665, max: 5.25135, range: 10.798, all_same: 0
[DEBUG] First few probabilities: p[0]=2.72217e-06 p[1]=2.87912e-06 p[2]=2.27405e-07 p[3]=1.16853e-06 p[4]=7.6739e-06 
[DEBUG] Final selected token: 136977
[DEBUG] Sampled token: 136977
[DEBUG] Forward pass iteration: 48
[DEBUG] Calling forward() with 51 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.5481, max: 5.2524, range: 10.8005, all_same: 0
[DEBUG] First few probabilities: p[0]=2.72511e-06 p[1]=2.88115e-06 p[2]=2.27424e-07 p[3]=1.16879e-06 p[4]=7.66487e-06 
[DEBUG] Final selected token: 63330
[DEBUG] Sampled token: 63330
[DEBUG] Forward pass iteration: 49
[DEBUG] Calling forward() with 52 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.55111, max: 5.25551, range: 10.8066, all_same: 0
[DEBUG] First few probabilities: p[0]=2.71948e-06 p[1]=2.87619e-06 p[2]=2.26714e-07 p[3]=1.16644e-06 p[4]=7.6689e-06 
[DEBUG] Final selected token: 141129
[DEBUG] Sampled token: 141129
[DEBUG] Forward pass iteration: 50
[DEBUG] Calling forward() with 53 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.55146, max: 5.25561, range: 10.8071, all_same: 0
[DEBUG] First few probabilities: p[0]=2.72199e-06 p[1]=2.87808e-06 p[2]=2.26833e-07 p[3]=1.16694e-06 p[4]=7.66326e-06 
[DEBUG] Final selected token: 58615
[DEBUG] Sampled token: 58615
[DEBUG] Forward pass iteration: 51
[DEBUG] Calling forward() with 54 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.5519, max: 5.25601, range: 10.8079, all_same: 0
[DEBUG] First few probabilities: p[0]=2.72201e-06 p[1]=2.87801e-06 p[2]=2.26781e-07 p[3]=1.1668e-06 p[4]=7.66217e-06 
[DEBUG] Final selected token: 45382
[DEBUG] Sampled token: 45382
[DEBUG] Forward pass iteration: 52
[DEBUG] Calling forward() with 55 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.55123, max: 5.25546, range: 10.8067, all_same: 0
[DEBUG] First few probabilities: p[0]=2.72147e-06 p[1]=2.87772e-06 p[2]=2.26825e-07 p[3]=1.16689e-06 p[4]=7.66472e-06 
[DEBUG] Final selected token: 90869
[DEBUG] Sampled token: 90869
[DEBUG] Forward pass iteration: 53
[DEBUG] Calling forward() with 56 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.55184, max: 5.25602, range: 10.8079, all_same: 0
[DEBUG] First few probabilities: p[0]=2.72136e-06 p[1]=2.87752e-06 p[2]=2.26745e-07 p[3]=1.16666e-06 p[4]=7.66355e-06 
[DEBUG] Final selected token: 30461
[DEBUG] Sampled token: 30461
[DEBUG] Forward pass iteration: 54
[DEBUG] Calling forward() with 57 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.55126, max: 5.25537, range: 10.8066, all_same: 0
[DEBUG] First few probabilities: p[0]=2.72297e-06 p[1]=2.87888e-06 p[2]=2.26916e-07 p[3]=1.16725e-06 p[4]=7.66178e-06 
[DEBUG] Final selected token: 84639
[DEBUG] Sampled token: 84639
[DEBUG] Forward pass iteration: 55
[DEBUG] Calling forward() with 58 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.55108, max: 5.25539, range: 10.8065, all_same: 0
[DEBUG] First few probabilities: p[0]=2.72079e-06 p[1]=2.87721e-06 p[2]=2.26798e-07 p[3]=1.16676e-06 p[4]=7.66641e-06 
[DEBUG] Final selected token: 129353
[DEBUG] Sampled token: 129353
[DEBUG] Forward pass iteration: 56
[DEBUG] Calling forward() with 59 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.55349, max: 5.25785, range: 10.8113, all_same: 0
[DEBUG] First few probabilities: p[0]=2.71639e-06 p[1]=2.87334e-06 p[2]=2.26239e-07 p[3]=1.16491e-06 p[4]=7.66945e-06 
[DEBUG] Final selected token: 2410
[DEBUG] Sampled token: 2410
[DEBUG] Forward pass iteration: 57
[DEBUG] Calling forward() with 60 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.55409, max: 5.25812, range: 10.8122, all_same: 0
[DEBUG] First few probabilities: p[0]=2.71978e-06 p[1]=2.87586e-06 p[2]=2.26383e-07 p[3]=1.16555e-06 p[4]=7.66147e-06 
[DEBUG] Final selected token: 2464
[DEBUG] Sampled token: 2464
[DEBUG] Forward pass iteration: 58
[DEBUG] Calling forward() with 61 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.55462, max: 5.25846, range: 10.8131, all_same: 0
[DEBUG] First few probabilities: p[0]=2.72151e-06 p[1]=2.8771e-06 p[2]=2.26432e-07 p[3]=1.16581e-06 p[4]=7.65688e-06 
[DEBUG] Final selected token: 65360
[DEBUG] Sampled token: 65360
[DEBUG] Forward pass iteration: 59
[DEBUG] Calling forward() with 62 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.55509, max: 5.25881, range: 10.8139, all_same: 0
[DEBUG] First few probabilities: p[0]=2.72236e-06 p[1]=2.87767e-06 p[2]=2.2643e-07 p[3]=1.16587e-06 p[4]=7.65421e-06 
[DEBUG] Final selected token: 21029
[DEBUG] Sampled token: 21029
[DEBUG] Forward pass iteration: 60
[DEBUG] Calling forward() with 63 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.55595, max: 5.25942, range: 10.8154, all_same: 0
[DEBUG] First few probabilities: p[0]=2.72432e-06 p[1]=2.87904e-06 p[2]=2.26454e-07 p[3]=1.16608e-06 p[4]=7.64843e-06 
[DEBUG] Final selected token: 83273
[DEBUG] Sampled token: 83273
[DEBUG] Forward pass iteration: 61
[DEBUG] Calling forward() with 64 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.55524, max: 5.25875, range: 10.814, all_same: 0
[DEBUG] First few probabilities: p[0]=2.72481e-06 p[1]=2.87955e-06 p[2]=2.26568e-07 p[3]=1.16642e-06 p[4]=7.64906e-06 
[DEBUG] Final selected token: 144430
[DEBUG] Sampled token: 144430
[DEBUG] Forward pass iteration: 62
[DEBUG] Calling forward() with 65 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.55452, max: 5.25813, range: 10.8127, all_same: 0
[DEBUG] First few probabilities: p[0]=2.72475e-06 p[1]=2.87965e-06 p[2]=2.26648e-07 p[3]=1.16664e-06 p[4]=7.65084e-06 
[DEBUG] Final selected token: 82773
[DEBUG] Sampled token: 82773
[DEBUG] Forward pass iteration: 63
[DEBUG] Calling forward() with 66 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.55408, max: 5.25777, range: 10.8119, all_same: 0
[DEBUG] First few probabilities: p[0]=2.7243e-06 p[1]=2.87937e-06 p[2]=2.26672e-07 p[3]=1.16667e-06 p[4]=7.65271e-06 
[DEBUG] Final selected token: 129849
[DEBUG] Sampled token: 129849
[DEBUG] Forward pass iteration: 64
[DEBUG] Calling forward() with 67 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.55452, max: 5.25807, range: 10.8126, all_same: 0
[DEBUG] First few probabilities: p[0]=2.72567e-06 p[1]=2.88035e-06 p[2]=2.26704e-07 p[3]=1.16686e-06 p[4]=7.64906e-06 
[DEBUG] Final selected token: 54571
[DEBUG] Sampled token: 54571
[DEBUG] Forward pass iteration: 65
[DEBUG] Calling forward() with 68 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.5552, max: 5.25867, range: 10.8139, all_same: 0
[DEBUG] First few probabilities: p[0]=2.72552e-06 p[1]=2.88012e-06 p[2]=2.26617e-07 p[3]=1.16662e-06 p[4]=7.64778e-06 
[DEBUG] Final selected token: 1847
[DEBUG] Sampled token: 1847
[DEBUG] Forward pass iteration: 66
[DEBUG] Calling forward() with 69 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.55502, max: 5.25863, range: 10.8136, all_same: 0
[DEBUG] First few probabilities: p[0]=2.72392e-06 p[1]=2.8789e-06 p[2]=2.26537e-07 p[3]=1.16628e-06 p[4]=7.65133e-06 
[DEBUG] Final selected token: 772
[DEBUG] Sampled token: 772
[DEBUG] Forward pass iteration: 67
[DEBUG] Calling forward() with 70 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.55324, max: 5.25694, range: 10.8102, all_same: 0
[DEBUG] First few probabilities: p[0]=2.72554e-06 p[1]=2.8805e-06 p[2]=2.26847e-07 p[3]=1.16724e-06 p[4]=7.65225e-06 
[DEBUG] Final selected token: 63843
[DEBUG] Sampled token: 63843
[DEBUG] Forward pass iteration: 68
[DEBUG] Calling forward() with 71 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.5533, max: 5.25692, range: 10.8102, all_same: 0
[DEBUG] First few probabilities: p[0]=2.72617e-06 p[1]=2.88098e-06 p[2]=2.26881e-07 p[3]=1.16739e-06 p[4]=7.65094e-06 
[DEBUG] Final selected token: 3972
[DEBUG] Sampled token: 3972
[DEBUG] Forward pass iteration: 69
[DEBUG] Calling forward() with 72 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.55384, max: 5.25759, range: 10.8114, all_same: 0
[DEBUG] First few probabilities: p[0]=2.72375e-06 p[1]=2.879e-06 p[2]=2.26666e-07 p[3]=1.16662e-06 p[4]=7.65436e-06 
[DEBUG] Final selected token: 100287
[DEBUG] Sampled token: 100287
[DEBUG] Forward pass iteration: 70
[DEBUG] Calling forward() with 73 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.55487, max: 5.25857, range: 10.8134, all_same: 0
[DEBUG] First few probabilities: p[0]=2.72294e-06 p[1]=2.87817e-06 p[2]=2.26493e-07 p[3]=1.16608e-06 p[4]=7.65354e-06 
[DEBUG] Final selected token: 72548
[DEBUG] Sampled token: 72548
[DEBUG] Forward pass iteration: 71
[DEBUG] Calling forward() with 74 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.55543, max: 5.25927, range: 10.8147, all_same: 0
[DEBUG] First few probabilities: p[0]=2.72037e-06 p[1]=2.87606e-06 p[2]=2.26263e-07 p[3]=1.16527e-06 p[4]=7.65723e-06 
[DEBUG] Final selected token: 57972
[DEBUG] Sampled token: 57972
[DEBUG] Forward pass iteration: 72
[DEBUG] Calling forward() with 75 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: -5.55624, max: 5.26003, range: 10.8163, all_same: 0
[DEBUG] First few probabilities: p[0]=2.71983e-06 p[1]=2.87549e-06 p[2]=2.26135e-07 p[3]=1.16488e-06 p[4]=7.65644e-06 
[DEBUG] Final selected token: 1330
[DEBUG] Sampled token: 1330
[DEBUG] Forward pass iteration: 73
[DEBUG] Calling forward() with 76 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: nan, max: nan, range: nan, all_same: 0
[DEBUG] First few probabilities: p[0]=nan p[1]=nan p[2]=nan p[3]=nan p[4]=nan 
[DEBUG] Final selected token: 0
[DEBUG] Sampled token: 0
[DEBUG] Consecutive zeros: 1
[DEBUG] Forward pass iteration: 74
[DEBUG] Calling forward() with 77 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: nan, max: nan, range: nan, all_same: 0
[DEBUG] First few probabilities: p[0]=nan p[1]=nan p[2]=nan p[3]=nan p[4]=nan 
[DEBUG] Final selected token: 0
[DEBUG] Sampled token: 0
[DEBUG] Consecutive zeros: 2
[DEBUG] Forward pass iteration: 75
[DEBUG] Calling forward() with 78 tokens
[DEBUG] Forward pass completed, sampling token
[DEBUG] Entering sampleToken with 151936 logits
[DEBUG] Logits stats - min: nan, max: nan, range: nan, all_same: 0
[DEBUG] First few probabilities: p[0]=nan p[1]=nan p[2]=nan p[3]=nan p[4]=nan 
[DEBUG] Final selected token: 0
[DEBUG] Sampled token: 0
[DEBUG] Consecutive zeros: 3
[DEBUG] Too many consecutive zeros, treating as EOS and stopping
[DEBUG] Detokenizing 75 tokens
[DEBUG] Processing token: 137079
[DEBUG] Token 137079 -> "×¦×ĵ×§"
[DEBUG] Processing token: 120328
[DEBUG] Token 120328 -> "æ¢ĥ"
[DEBUG] Processing token: 6231
[DEBUG] Token 6231 -> "Login"
[DEBUG] Processing token: 12551
[DEBUG] Token 12551 -> "_BO"
[DEBUG] Processing token: 137306
[DEBUG] Token 137306 -> "×Ķ×ª×Ļ×Ļ×Ĺ×¡"
[DEBUG] Processing token: 83812
[DEBUG] Token 83812 -> "_mpi"
[DEBUG] Processing token: 27742
[DEBUG] Token 27742 -> "marker"
[DEBUG] Processing token: 23979
[DEBUG] Token 23979 -> ".Null"
[DEBUG] Processing token: 37962
[DEBUG] Token 37962 -> "ĠNSInteger"
[DEBUG] Processing token: 68228
[DEBUG] Token 68228 -> "lÃ©"
[DEBUG] Processing token: 114923
[DEBUG] Token 114923 -> "åĨ·ç¬ĳ"
[DEBUG] Processing token: 55370
[DEBUG] Token 55370 -> "?s"
[DEBUG] Processing token: 13774
[DEBUG] Token 13774 -> "esis"
[DEBUG] Processing token: 20440
[DEBUG] Token 20440 -> ".setString"
[DEBUG] Processing token: 59560
[DEBUG] Token 59560 -> "ETO"
[DEBUG] Processing token: 88300
[DEBUG] Token 88300 -> "GetY"
[DEBUG] Processing token: 42067
[DEBUG] Token 42067 -> "éĽĨ"
[DEBUG] Processing token: 3951
[DEBUG] Token 3951 -> "Ġmonths"
[DEBUG] Processing token: 53592
[DEBUG] Token 53592 -> "ĠSceneManager"
[DEBUG] Processing token: 131315
[DEBUG] Token 131315 -> "rÃ¶"
[DEBUG] Processing token: 80555
[DEBUG] Token 80555 -> "Lots"
[DEBUG] Processing token: 1860
[DEBUG] Token 1860 -> "Service"
[DEBUG] Processing token: 32173
[DEBUG] Token 32173 -> "Ġdetention"
[DEBUG] Processing token: 36063
[DEBUG] Token 36063 -> "Ġsmiling"
[DEBUG] Processing token: 116500
[DEBUG] Token 116500 -> "æĢ»éĺŁ"
[DEBUG] Processing token: 90898
[DEBUG] Token 90898 -> "tep"
[DEBUG] Processing token: 70850
[DEBUG] Token 70850 -> "Ġgladly"
[DEBUG] Processing token: 15337
[DEBUG] Token 15337 -> "aland"
[DEBUG] Processing token: 103171
[DEBUG] Token 103171 -> "èįĨ"
[DEBUG] Processing token: 83490
[DEBUG] Token 83490 -> "ĠSAME"
[DEBUG] Processing token: 106314
[DEBUG] Token 106314 -> "åľ¨ä»ĸ"
[DEBUG] Processing token: 142809
[DEBUG] Token 142809 -> "å¼ķãģ£è¶ĬãģĹ"
[DEBUG] Processing token: 12860
[DEBUG] Token 12860 -> "(stderr"
[DEBUG] Processing token: 143191
[DEBUG] Token 143191 -> "×Ĳ×ķ×Ĵ×ķ×¡"
[DEBUG] Processing token: 115074
[DEBUG] Token 115074 -> "åıªèĥ½è¯´"
[DEBUG] Processing token: 64268
[DEBUG] Token 64268 -> "AMENT"
[DEBUG] Processing token: 54035
[DEBUG] Token 54035 -> "ĠMajority"
[DEBUG] Processing token: 96721
[DEBUG] Token 96721 -> "]]];Ċ"
[DEBUG] Processing token: 105440
[DEBUG] Token 105440 -> "å¾Īé«ĺ"
[DEBUG] Processing token: 142908
[DEBUG] Token 142908 -> "à¹Ģà¸Ļà¸Ńà¸£à¹Į"
[DEBUG] Processing token: 44307
[DEBUG] Token 44307 -> "')""
[DEBUG] Processing token: 39361
[DEBUG] Token 39361 -> "ĠManifest"
[DEBUG] Processing token: 128351
[DEBUG] Token 128351 -> "ĠkhÃ¡ch"
[DEBUG] Processing token: 139915
[DEBUG] Token 139915 -> "Ġ×Ļ×Ķ×ķ×ĵ×Ķ"
[DEBUG] Processing token: 28232
[DEBUG] Token 28232 -> "ĠSnap"
[DEBUG] Processing token: 112393
[DEBUG] Token 112393 -> "çī©ä¸ļæľįåĬ¡"
[DEBUG] Processing token: 64269
[DEBUG] Token 64269 -> ".sharedInstance"
[DEBUG] Processing token: 136977
[DEBUG] Token 136977 -> "ĠtÃ²a"
[DEBUG] Processing token: 63330
[DEBUG] Token 63330 -> "Ġroc"
[DEBUG] Processing token: 141129
[DEBUG] Token 141129 -> "à¸ļà¸²à¸Ħà¸²à¸£"
[DEBUG] Processing token: 58615
[DEBUG] Token 58615 -> ".TableLayoutPanel"
[DEBUG] Processing token: 45382
[DEBUG] Token 45382 -> "Ġvegetarian"
[DEBUG] Processing token: 90869
[DEBUG] Token 90869 -> "rms"
[DEBUG] Processing token: 30461
[DEBUG] Token 30461 -> ".colors"
[DEBUG] Processing token: 84639
[DEBUG] Token 84639 -> "Ġ'|'"
[DEBUG] Processing token: 129353
[DEBUG] Token 129353 -> "ĠØ¹Ø¨Ø±"
[DEBUG] Processing token: 2410
[DEBUG] Token 2410 -> "ĠAd"
[DEBUG] Processing token: 2464
[DEBUG] Token 2464 -> "Ġfun"
[DEBUG] Processing token: 65360
[DEBUG] Token 65360 -> "(Account"
[DEBUG] Processing token: 21029
[DEBUG] Token 21029 -> "ĠExcel"
[DEBUG] Processing token: 83273
[DEBUG] Token 83273 -> "Ġpropositions"
[DEBUG] Processing token: 144430
[DEBUG] Token 144430 -> "ïºĳ"
[DEBUG] Processing token: 82773
[DEBUG] Token 82773 -> "-pic"
[DEBUG] Processing token: 129849
[DEBUG] Token 129849 -> "Ø¹ÙĤØ¯"
[DEBUG] Processing token: 54571
[DEBUG] Token 54571 -> "Ġpolicing"
[DEBUG] Processing token: 1847
[DEBUG] Token 1847 -> "off"
[DEBUG] Processing token: 772
[DEBUG] Token 772 -> "cont"
[DEBUG] Processing token: 63843
[DEBUG] Token 63843 -> ":D"
[DEBUG] Processing token: 3972
[DEBUG] Token 3972 -> "ĠEnd"
[DEBUG] Processing token: 100287
[DEBUG] Token 100287 -> "è¿ģ"
[DEBUG] Processing token: 72548
[DEBUG] Token 72548 -> ".appspot"
[DEBUG] Processing token: 57972
[DEBUG] Token 57972 -> "ItemSelectedListener"
[DEBUG] Processing token: 1330
[DEBUG] Token 1330 -> "ins"
[DEBUG] Processing token: 0
[DEBUG] Token 0 -> "!"
[DEBUG] Processing token: 0
[DEBUG] Token 0 -> "!"
[DEBUG] Final detokenized result: "×¦×ĵ×§æ¢ĥLogin_BO×Ķ×ª×Ļ×Ļ×Ĺ×¡_mpimarker.NullĠNSIntegerlÃ©åĨ·ç¬ĳ?sesis.setStringETOGetYéĽĨĠmonthsĠSceneManagerrÃ¶LotsServiceĠdetentionĠsmilingæĢ»éĺŁtepĠgladlyalandèįĨĠSAMEåľ¨ä»ĸå¼ķãģ£è¶ĬãģĹ(stderr×Ĳ×ķ×Ĵ×ķ×¡åıªèĥ½è¯´AMENTĠMajority]]];Ċå¾Īé«ĺà¹Ģà¸Ļà¸Ńà¸£à¹Į')"ĠManifestĠkhÃ¡chĠ×Ļ×Ķ×ķ×ĵ×ĶĠSnapçī©ä¸ļæľįåĬ¡.sharedInstanceĠtÃ²aĠrocà¸ļà¸²à¸Ħà¸²à¸£.TableLayoutPanelĠvegetarianrms.colorsĠ'|'ĠØ¹Ø¨Ø±ĠAdĠfun(AccountĠExcelĠpropositionsïºĳ-picØ¹ÙĤØ¯Ġpolicingoffcont:DĠEndè¿ģ.appspotItemSelectedListenerins!!"
[DEBUG] engine->generateText completed successfully
[DEBUG] Tokenizing generated text for token count
[DEBUG] Tokenizing text: "×¦×ĵ×§æ¢ĥLogin_BO×Ķ×ª×Ļ×Ļ×Ĺ×¡_mpimarker.NullĠNSIntegerlÃ©åĨ·ç¬ĳ?sesis.setStringETOGetYéĽĨĠmonthsĠSceneManagerrÃ¶LotsServiceĠdetentionĠsmilingæĢ»éĺŁtepĠgladlyalandèįĨĠSAMEåľ¨ä»ĸå¼ķãģ£è¶ĬãģĹ(stderr×Ĳ×ķ×Ĵ×ķ×¡åıªèĥ½è¯´AMENTĠMajority]]];Ċå¾Īé«ĺà¹Ģà¸Ļà¸Ńà¸£à¹Į')"ĠManifestĠkhÃ¡chĠ×Ļ×Ķ×ķ×ĵ×ĶĠSnapçī©ä¸ļæľįåĬ¡.sharedInstanceĠtÃ²aĠrocà¸ļà¸²à¸Ħà¸²à¸£.TableLayoutPanelĠvegetarianrms.colorsĠ'|'ĠØ¹Ø¨Ø±ĠAdĠfun(AccountĠExcelĠpropositionsïºĳ-picØ¹ÙĤØ¯Ġpolicingoffcont:DĠEndè¿ģ.appspotItemSelectedListenerins!!"
[DEBUG] Used legacy tokenizer, got 679 tokens: 151643 10195 10151 10194 10166 10195 10151 10196 10181 10195 ...
[DEBUG] Generated 679 tokens
[DEBUG] OllamaModelManager::generateText completed in 1.42193e+06ms
[DEBUG] Ollama inference successful: ×¦×ĵ×§æ¢ĥLogin_BO×Ķ×ª×Ļ×Ļ×Ĺ×¡...
[DEBUG] TextGenerator returning result: ×¦×ĵ×§æ¢ĥLogin_BO×Ķ...
[DEBUG] ChatView: Text generation completed successfully
Found key session_list with value ["chat_1758019709191_6342"]

### 原因分析
- 推理很慢的主因：没有使用任何实质性的 KV Cache，注意力每步都在对全序列做 O(L^2) 计算，而且每层、每步都新建 ggml 上下文，默认线程还是 1，再叠加大量 DEBUG 日志，整体会非常慢（你的 1.42e6ms≈23.7 分钟是符合这类实现特征的）。
- 输出“编码不对/乱码”的主因：Token Embedding 根本没有从 GGUF 加载，现在是随机初始化的占位实现；注意力很多时候走了“fallback”路径；RoPE 暂时被禁用；而分词/解码也可能与模型不匹配（把 GPT2-BPE 用到了非 GPT2 的模型）。这些综合起来，哪怕日志里每个 token id 都能打印出来，对应的向量/解码也必然是错的或无意义的。
下面把具体“证据”对应到你代码里的位置，并给出逐项修复建议。

一、导致“编码错/输出乱”的关键问题与证据

1. 1.
   Token Embedding 未从 GGUF 加载，当前是随机初始化
- 代码位置： `loadTokenEmbedding`
- 现状：实现里直接随机初始化（占位）而不是从 GGUF 拿真正的嵌入矩阵。这样一来，任何 logits/解码都会变成“随机信号”，自然“编码看起来不对”。
2. 1.
   注意力经常走 fallback，RoPE 还被禁用
- 注意力实现： `multiHeadAttention`
- 在该函数的中段你能看到“暂时跳过RoPE应用”的注释（“暂时跳过RoPE应用以避免编译错误”）。这意味着没有位置编码，模型无法正常工作。
- 同时，代码中对形状不匹配或权重未就绪的情况会频繁打印 WARNING/ERROR 并“回退”（fallback）到把输出设为 input 的缩放（乘 0.5）；这种回退输出当然与真实模型无关。

3. 1.
   输出投影权重是否加载到位？
- 输出投影在注意力尾部会使用 head.output_weights（如果尺寸足够）；但如果 `loadOutputWeights` 或其它层权重未正确从 GGUF 装载，注意力就会继续 fallback。结合 1) 的随机 embedding，即使这里工作正常，也会是垃圾输入。

二、导致“推理很慢”的关键问题与证据
1. 1.
   没用 KV Cache：每步都对全序列做注意力
- 注意力实现： `multiHeadAttention`
- KVCache 的开关虽有： `enableKVCache` ，但在注意力里并没有把历史 K/V 追加起来复用，导致复杂度是每步 O(L^2)，随生成长度 L 呈平方级增长，速度必然很慢。
2. 1.
   每层/每次都新建 ggml 上下文，且默认线程=1
- 在注意力里你可以看到反复建立 ggml context、reshape/permute/构图/执行；默认线程数在构造时为 1，未主动提升： `qwen25vl_inference_engine.cpp` （构造函数 19-40 行附近），以及 `setNumThreads` 存在但未被调用。
- 这种使用方式对 CPU 性能非常不友好。
3. 1.
   大量 DEBUG 日志处于热路径
- multiHeadAttention() 中到处是 DEBUG/WARNING/ERROR 的日志打印，字符串拼接与 IO 本身就很耗时，放在每步、每层的热路径会严重拖慢速度。
4. 1.
   forward 每次都在“整段跑”，没有真正的 incremental 模式
- 主推理入口： `forward`
- 结合未使用 KV cache 的事实，意味着每生成一个 token，都把全序列重新过一遍所有层，复杂度和内存分配都非常高。
三、修复建议（按优先级从高到低）
P0-阻断错误输出/乱码

- 正确加载嵌入与输出层权重
  - 在 `loadTokenEmbedding` 中，从 GGUF 读取 embedding 权重（通常是 vocab_size x hidden_size 或 hidden_size x vocab_size，确保维度与你的 Tensor 排布一致），不要随机初始化。
  - 根据模型是否使用 weight tying，把输出投影设为 embedding 的转置或单独从 GGUF 读取，在 `loadOutputWeights` 中完成。
  - 加强 `loadTensorFromGGUF` ：列举 GGUF 中的所有张量名，建立从“模型结构字段名”到“GGUF张量名”的映射，严格校验张量尺寸，加载后对少量元素做非零/均值方差检查，防止 silent failure。
- 重新启用并正确应用 RoPE
  - 在 `multiHeadAttention` 内对 Q、K 使用 ggml_rope（按 head_dim/旋转维度、theta、层偏移等参数），不要使用当前那个“简化版 applyRoPE 到整个向量”的做法；现有注释“暂时跳过RoPE”应去掉，改为在 Q、K reshape/permute 之前或之后，按正确维度处理。
- tokenizer/解码与模型对齐
  - 在 `loadVocabulary` 和 `loadTokenizerFromGGUF` 中读取并遵循 GGUF 的 tokenizer 元信息（类型、词表、merges、特殊符号等），不要强行用 GPT2-BPE。如果 GGUF 写的是 SentencePiece，就用 SPM；如果是 GPT2-BPE，就加载 merges.json 和 vocab.json 栈。
  - detokenize 时遵循该 tokenizer 的规则（例如 SPM 有 ▁ 前缀；GPT2 会在空格处理上有特殊规则）。
P0-加速的根本性措施

- 实现并使用 KV Cache
  - 在 `multiHeadAttention` 中，把当前 step 计算出来的 K/V 追加到每层的缓存；下一步只对新 token 做 Q 与“缓存过的所有 K/V”做注意力。这样每步复杂度从 O(L^2) 降为 O(L·d) 级。
  - 你已有 KVCache 结构体定义（在头文件里），但当前注意力没用它来读写缓存。把它真正串起来，并在 `enableKVCache` 控制开关。
- 复用 ggml 上下文，避免每步/每层都重建
  - 建议在一次 forward/generate 的生命周期内，复用一个较大的 ggml_context（或使用内存池/arena），不要在注意力函数里频繁 ggml_init/ggml_free。把构图-执行的粒度拉大，减少碎片化。
P1-重要的性能优化

- 把线程数设置为硬件并行度，并提供对外配置
  - 在模型加载后默认执行 `setNumThreads` ，值取 std::thread::hardware_concurrency()（允许用户覆盖）。
- 精简热路径日志
  - 将 multiHeadAttention()、feedForward() 中的大量 DEBUG/WARN 输出改为条件编译或者按 verbose 标志位聚合/抽样输出，避免每步大量 IO。
- 减少不必要的 reshape/permute
  - 一些维度转换可以通过权重排布预先整理来减少在热路径中的张量操作次数。
P2-正确性与健壮性增强

- 在 `forward` 中，引入“增量模式”的输入/输出接口（传入 past_seq_len、只 embed 新 token），结合 KV cache。
- 加强所有权重加载后的维度验证与数值检查（均值、方差、非零率），对不上就 fail fast，而不是静默 fallback。
- 在 `getVocabSize` 与 Embedding 矩阵尺寸做一致性校验（vocab_size 一致），避免 id 溢出或 padding 错位。
四、你当前日志数字的合理性

- 1.42193e+06 ms 的量级非常符合“无 KV cache + 每步全序列注意力 + 单线程 + 重度日志”的实现特征。
- “每个 token id 对应的编码向量都在文档里”这说明流水线能跑，但是鉴于嵌入是随机的、RoPE 关闭、注意力频繁 fallback，你看到的编码/解码自然是“乱”的。
五、接下来我可以帮你做什么

- 优化
  1. 1.
     把 embedding、输出投影、Q/K/V 权重等全部从 GGUF 正确加载并校验；
  2. 2.
     正确接回 RoPE（基于 ggml_rope）；
  3. 3.
     接通 per-layer KV cache，改造成真正增量生成；
  4. 4.
     统一管理 ggml 上下文、提升线程数、下掉热路径日志；

