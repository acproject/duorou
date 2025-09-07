### CLI端的日志信息
```sh
[2025-09-07 09:21:29] [WARNING] Failed to load FFN up weights for layer 6
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.6.mlp.down_proj.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.6.mlp.down_proj.weight
[2025-09-07 09:21:29] [WARNING] Failed to load FFN down weights for layer 6
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.6.input_layernorm.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.6.input_layernorm.weight
[2025-09-07 09:21:29] [WARNING] Failed to load attention norm weights for layer 6
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.6.post_attention_layernorm.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.6.post_attention_layernorm.weight
[2025-09-07 09:21:29] [WARNING] Failed to load FFN norm weights for layer 6
[2025-09-07 09:21:29] [DEBUG] Layer 6 loaded - FFN gate: 8388608, up: 8388608, down: 8388608, attn heads: 1
[2025-09-07 09:21:29] [DEBUG] QKV weights memory requirement: 147MB for layer 7
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.7.self_attn.qkv_proj.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.7.self_attn.qkv_proj.weight
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.7.self_attn.q_proj.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.7.self_attn.q_proj.weight
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.7.self_attn.k_proj.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.7.self_attn.k_proj.weight
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.7.self_attn.v_proj.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.7.self_attn.v_proj.weight
[2025-09-07 09:21:29] [WARNING] Failed to load attention weights for layer 7
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.7.self_attn.o_proj.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.7.self_attn.o_proj.weight
[2025-09-07 09:21:29] [WARNING] Failed to load attention output weights for layer 7
[2025-09-07 09:21:29] [DEBUG] FFN weights memory requirement: gate/up=259MB, down=259MB for layer 7
[2025-09-07 09:21:29] [WARNING] FFN weights too large, using fallback sizes
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.7.mlp.gate_proj.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.7.mlp.gate_proj.weight
[2025-09-07 09:21:29] [WARNING] Failed to load FFN gate weights for layer 7
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.7.mlp.up_proj.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.7.mlp.up_proj.weight
[2025-09-07 09:21:29] [WARNING] Failed to load FFN up weights for layer 7
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.7.mlp.down_proj.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.7.mlp.down_proj.weight
[2025-09-07 09:21:29] [WARNING] Failed to load FFN down weights for layer 7
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.7.input_layernorm.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.7.input_layernorm.weight
[2025-09-07 09:21:29] [WARNING] Failed to load attention norm weights for layer 7
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.7.post_attention_layernorm.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.7.post_attention_layernorm.weight
[2025-09-07 09:21:29] [WARNING] Failed to load FFN norm weights for layer 7
[2025-09-07 09:21:29] [DEBUG] Layer 7 loaded - FFN gate: 8388608, up: 8388608, down: 8388608, attn heads: 1
[2025-09-07 09:21:29] [DEBUG] QKV weights memory requirement: 147MB for layer 8
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.8.self_attn.qkv_proj.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.8.self_attn.qkv_proj.weight
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.8.self_attn.q_proj.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.8.self_attn.q_proj.weight
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.8.self_attn.k_proj.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.8.self_attn.k_proj.weight
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.8.self_attn.v_proj.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.8.self_attn.v_proj.weight
[2025-09-07 09:21:29] [WARNING] Failed to load attention weights for layer 8
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.8.self_attn.o_proj.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.8.self_attn.o_proj.weight
[2025-09-07 09:21:29] [WARNING] Failed to load attention output weights for layer 8
[2025-09-07 09:21:29] [DEBUG] FFN weights memory requirement: gate/up=259MB, down=259MB for layer 8
[2025-09-07 09:21:29] [WARNING] FFN weights too large, using fallback sizes
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.8.mlp.gate_proj.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.8.mlp.gate_proj.weight
[2025-09-07 09:21:29] [WARNING] Failed to load FFN gate weights for layer 8
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.8.mlp.up_proj.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.8.mlp.up_proj.weight
[2025-09-07 09:21:29] [WARNING] Failed to load FFN up weights for layer 8
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.8.mlp.down_proj.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.8.mlp.down_proj.weight
[2025-09-07 09:21:29] [WARNING] Failed to load FFN down weights for layer 8
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.8.input_layernorm.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.8.input_layernorm.weight
[2025-09-07 09:21:29] [WARNING] Failed to load attention norm weights for layer 8
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.8.post_attention_layernorm.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.8.post_attention_layernorm.weight
[2025-09-07 09:21:29] [WARNING] Failed to load FFN norm weights for layer 8
[2025-09-07 09:21:29] [DEBUG] Layer 8 loaded - FFN gate: 8388608, up: 8388608, down: 8388608, attn heads: 1
[2025-09-07 09:21:29] [DEBUG] QKV weights memory requirement: 147MB for layer 9
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.9.self_attn.qkv_proj.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.9.self_attn.qkv_proj.weight
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.9.self_attn.q_proj.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.9.self_attn.q_proj.weight
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.9.self_attn.k_proj.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.9.self_attn.k_proj.weight
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.9.self_attn.v_proj.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.9.self_attn.v_proj.weight
[2025-09-07 09:21:29] [WARNING] Failed to load attention weights for layer 9
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.9.self_attn.o_proj.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.9.self_attn.o_proj.weight
[2025-09-07 09:21:29] [WARNING] Failed to load attention output weights for layer 9
[2025-09-07 09:21:29] [DEBUG] FFN weights memory requirement: gate/up=259MB, down=259MB for layer 9
[2025-09-07 09:21:29] [WARNING] FFN weights too large, using fallback sizes
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.9.mlp.gate_proj.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.9.mlp.gate_proj.weight
[2025-09-07 09:21:29] [WARNING] Failed to load FFN gate weights for layer 9
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.9.mlp.up_proj.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.9.mlp.up_proj.weight
[2025-09-07 09:21:29] [WARNING] Failed to load FFN up weights for layer 9
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.9.mlp.down_proj.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.9.mlp.down_proj.weight
[2025-09-07 09:21:29] [WARNING] Failed to load FFN down weights for layer 9
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.9.input_layernorm.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.9.input_layernorm.weight
[2025-09-07 09:21:29] [WARNING] Failed to load attention norm weights for layer 9
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.9.post_attention_layernorm.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.9.post_attention_layernorm.weight
[2025-09-07 09:21:29] [WARNING] Failed to load FFN norm weights for layer 9
[2025-09-07 09:21:29] [DEBUG] Layer 9 loaded - FFN gate: 8388608, up: 8388608, down: 8388608, attn heads: 1
[2025-09-07 09:21:29] [DEBUG] QKV weights memory requirement: 147MB for layer 10
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.10.self_attn.qkv_proj.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.10.self_attn.qkv_proj.weight
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.10.self_attn.q_proj.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.10.self_attn.q_proj.weight
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.10.self_attn.k_proj.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.10.self_attn.k_proj.weight
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.10.self_attn.v_proj.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.10.self_attn.v_proj.weight
[2025-09-07 09:21:29] [WARNING] Failed to load attention weights for layer 10
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.10.self_attn.o_proj.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.10.self_attn.o_proj.weight
[2025-09-07 09:21:29] [WARNING] Failed to load attention output weights for layer 10
[2025-09-07 09:21:29] [DEBUG] FFN weights memory requirement: gate/up=259MB, down=259MB for layer 10
[2025-09-07 09:21:29] [WARNING] FFN weights too large, using fallback sizes
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.10.mlp.gate_proj.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.10.mlp.gate_proj.weight
[2025-09-07 09:21:29] [WARNING] Failed to load FFN gate weights for layer 10
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.10.mlp.up_proj.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.10.mlp.up_proj.weight
[2025-09-07 09:21:29] [WARNING] Failed to load FFN up weights for layer 10
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.10.mlp.down_proj.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.10.mlp.down_proj.weight
[2025-09-07 09:21:29] [WARNING] Failed to load FFN down weights for layer 10
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.10.input_layernorm.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.10.input_layernorm.weight
[2025-09-07 09:21:29] [WARNING] Failed to load attention norm weights for layer 10
[2025-09-07 09:21:29] [INFO] Loading tensor: model.layers.10.post_attention_layernorm.weight
[2025-09-07 09:21:29] [ERROR] Tensor not found in GGUF file: model.layers.10.post_attention_layernorm.weight
[2025-09-07 09:21:29] [WARNING] Failed to load FFN norm weights for layer 10
[2025-09-07 09:21:29] [DEBUG] Layer 10 loaded - FFN gate: 8388608, up: 8388608, down: 8388608, attn heads: 1
[2025-09-07 09:21:30] [DEBUG] QKV weights memory requirement: 147MB for layer 11
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.11.self_attn.qkv_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.11.self_attn.qkv_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.11.self_attn.q_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.11.self_attn.q_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.11.self_attn.k_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.11.self_attn.k_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.11.self_attn.v_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.11.self_attn.v_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention weights for layer 11
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.11.self_attn.o_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.11.self_attn.o_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention output weights for layer 11
[2025-09-07 09:21:30] [DEBUG] FFN weights memory requirement: gate/up=259MB, down=259MB for layer 11
[2025-09-07 09:21:30] [WARNING] FFN weights too large, using fallback sizes
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.11.mlp.gate_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.11.mlp.gate_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN gate weights for layer 11
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.11.mlp.up_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.11.mlp.up_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN up weights for layer 11
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.11.mlp.down_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.11.mlp.down_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN down weights for layer 11
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.11.input_layernorm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.11.input_layernorm.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention norm weights for layer 11
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.11.post_attention_layernorm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.11.post_attention_layernorm.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN norm weights for layer 11
[2025-09-07 09:21:30] [DEBUG] Layer 11 loaded - FFN gate: 8388608, up: 8388608, down: 8388608, attn heads: 1
[2025-09-07 09:21:30] [DEBUG] QKV weights memory requirement: 147MB for layer 12
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.12.self_attn.qkv_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.12.self_attn.qkv_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.12.self_attn.q_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.12.self_attn.q_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.12.self_attn.k_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.12.self_attn.k_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.12.self_attn.v_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.12.self_attn.v_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention weights for layer 12
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.12.self_attn.o_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.12.self_attn.o_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention output weights for layer 12
[2025-09-07 09:21:30] [DEBUG] FFN weights memory requirement: gate/up=259MB, down=259MB for layer 12
[2025-09-07 09:21:30] [WARNING] FFN weights too large, using fallback sizes
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.12.mlp.gate_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.12.mlp.gate_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN gate weights for layer 12
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.12.mlp.up_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.12.mlp.up_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN up weights for layer 12
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.12.mlp.down_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.12.mlp.down_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN down weights for layer 12
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.12.input_layernorm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.12.input_layernorm.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention norm weights for layer 12
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.12.post_attention_layernorm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.12.post_attention_layernorm.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN norm weights for layer 12
[2025-09-07 09:21:30] [DEBUG] Layer 12 loaded - FFN gate: 8388608, up: 8388608, down: 8388608, attn heads: 1
[2025-09-07 09:21:30] [DEBUG] QKV weights memory requirement: 147MB for layer 13
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.13.self_attn.qkv_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.13.self_attn.qkv_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.13.self_attn.q_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.13.self_attn.q_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.13.self_attn.k_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.13.self_attn.k_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.13.self_attn.v_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.13.self_attn.v_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention weights for layer 13
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.13.self_attn.o_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.13.self_attn.o_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention output weights for layer 13
[2025-09-07 09:21:30] [DEBUG] FFN weights memory requirement: gate/up=259MB, down=259MB for layer 13
[2025-09-07 09:21:30] [WARNING] FFN weights too large, using fallback sizes
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.13.mlp.gate_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.13.mlp.gate_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN gate weights for layer 13
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.13.mlp.up_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.13.mlp.up_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN up weights for layer 13
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.13.mlp.down_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.13.mlp.down_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN down weights for layer 13
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.13.input_layernorm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.13.input_layernorm.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention norm weights for layer 13
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.13.post_attention_layernorm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.13.post_attention_layernorm.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN norm weights for layer 13
[2025-09-07 09:21:30] [DEBUG] Layer 13 loaded - FFN gate: 8388608, up: 8388608, down: 8388608, attn heads: 1
[2025-09-07 09:21:30] [DEBUG] QKV weights memory requirement: 147MB for layer 14
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.14.self_attn.qkv_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.14.self_attn.qkv_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.14.self_attn.q_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.14.self_attn.q_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.14.self_attn.k_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.14.self_attn.k_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.14.self_attn.v_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.14.self_attn.v_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention weights for layer 14
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.14.self_attn.o_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.14.self_attn.o_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention output weights for layer 14
[2025-09-07 09:21:30] [DEBUG] FFN weights memory requirement: gate/up=259MB, down=259MB for layer 14
[2025-09-07 09:21:30] [WARNING] FFN weights too large, using fallback sizes
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.14.mlp.gate_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.14.mlp.gate_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN gate weights for layer 14
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.14.mlp.up_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.14.mlp.up_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN up weights for layer 14
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.14.mlp.down_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.14.mlp.down_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN down weights for layer 14
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.14.input_layernorm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.14.input_layernorm.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention norm weights for layer 14
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.14.post_attention_layernorm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.14.post_attention_layernorm.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN norm weights for layer 14
[2025-09-07 09:21:30] [DEBUG] Layer 14 loaded - FFN gate: 8388608, up: 8388608, down: 8388608, attn heads: 1
[2025-09-07 09:21:30] [DEBUG] QKV weights memory requirement: 147MB for layer 15
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.15.self_attn.qkv_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.15.self_attn.qkv_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.15.self_attn.q_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.15.self_attn.q_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.15.self_attn.k_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.15.self_attn.k_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.15.self_attn.v_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.15.self_attn.v_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention weights for layer 15
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.15.self_attn.o_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.15.self_attn.o_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention output weights for layer 15
[2025-09-07 09:21:30] [DEBUG] FFN weights memory requirement: gate/up=259MB, down=259MB for layer 15
[2025-09-07 09:21:30] [WARNING] FFN weights too large, using fallback sizes
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.15.mlp.gate_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.15.mlp.gate_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN gate weights for layer 15
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.15.mlp.up_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.15.mlp.up_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN up weights for layer 15
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.15.mlp.down_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.15.mlp.down_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN down weights for layer 15
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.15.input_layernorm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.15.input_layernorm.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention norm weights for layer 15
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.15.post_attention_layernorm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.15.post_attention_layernorm.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN norm weights for layer 15
[2025-09-07 09:21:30] [DEBUG] Layer 15 loaded - FFN gate: 8388608, up: 8388608, down: 8388608, attn heads: 1
[2025-09-07 09:21:30] [DEBUG] QKV weights memory requirement: 147MB for layer 16
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.16.self_attn.qkv_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.16.self_attn.qkv_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.16.self_attn.q_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.16.self_attn.q_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.16.self_attn.k_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.16.self_attn.k_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.16.self_attn.v_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.16.self_attn.v_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention weights for layer 16
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.16.self_attn.o_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.16.self_attn.o_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention output weights for layer 16
[2025-09-07 09:21:30] [DEBUG] FFN weights memory requirement: gate/up=259MB, down=259MB for layer 16
[2025-09-07 09:21:30] [WARNING] FFN weights too large, using fallback sizes
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.16.mlp.gate_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.16.mlp.gate_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN gate weights for layer 16
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.16.mlp.up_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.16.mlp.up_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN up weights for layer 16
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.16.mlp.down_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.16.mlp.down_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN down weights for layer 16
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.16.input_layernorm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.16.input_layernorm.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention norm weights for layer 16
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.16.post_attention_layernorm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.16.post_attention_layernorm.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN norm weights for layer 16
[2025-09-07 09:21:30] [DEBUG] Layer 16 loaded - FFN gate: 8388608, up: 8388608, down: 8388608, attn heads: 1
[2025-09-07 09:21:30] [DEBUG] QKV weights memory requirement: 147MB for layer 17
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.17.self_attn.qkv_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.17.self_attn.qkv_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.17.self_attn.q_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.17.self_attn.q_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.17.self_attn.k_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.17.self_attn.k_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.17.self_attn.v_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.17.self_attn.v_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention weights for layer 17
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.17.self_attn.o_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.17.self_attn.o_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention output weights for layer 17
[2025-09-07 09:21:30] [DEBUG] FFN weights memory requirement: gate/up=259MB, down=259MB for layer 17
[2025-09-07 09:21:30] [WARNING] FFN weights too large, using fallback sizes
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.17.mlp.gate_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.17.mlp.gate_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN gate weights for layer 17
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.17.mlp.up_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.17.mlp.up_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN up weights for layer 17
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.17.mlp.down_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.17.mlp.down_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN down weights for layer 17
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.17.input_layernorm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.17.input_layernorm.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention norm weights for layer 17
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.17.post_attention_layernorm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.17.post_attention_layernorm.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN norm weights for layer 17
[2025-09-07 09:21:30] [DEBUG] Layer 17 loaded - FFN gate: 8388608, up: 8388608, down: 8388608, attn heads: 1
[2025-09-07 09:21:30] [DEBUG] QKV weights memory requirement: 147MB for layer 18
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.18.self_attn.qkv_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.18.self_attn.qkv_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.18.self_attn.q_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.18.self_attn.q_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.18.self_attn.k_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.18.self_attn.k_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.18.self_attn.v_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.18.self_attn.v_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention weights for layer 18
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.18.self_attn.o_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.18.self_attn.o_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention output weights for layer 18
[2025-09-07 09:21:30] [DEBUG] FFN weights memory requirement: gate/up=259MB, down=259MB for layer 18
[2025-09-07 09:21:30] [WARNING] FFN weights too large, using fallback sizes
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.18.mlp.gate_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.18.mlp.gate_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN gate weights for layer 18
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.18.mlp.up_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.18.mlp.up_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN up weights for layer 18
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.18.mlp.down_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.18.mlp.down_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN down weights for layer 18
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.18.input_layernorm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.18.input_layernorm.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention norm weights for layer 18
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.18.post_attention_layernorm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.18.post_attention_layernorm.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN norm weights for layer 18
[2025-09-07 09:21:30] [DEBUG] Layer 18 loaded - FFN gate: 8388608, up: 8388608, down: 8388608, attn heads: 1
[2025-09-07 09:21:30] [DEBUG] QKV weights memory requirement: 147MB for layer 19
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.19.self_attn.qkv_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.19.self_attn.qkv_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.19.self_attn.q_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.19.self_attn.q_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.19.self_attn.k_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.19.self_attn.k_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.19.self_attn.v_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.19.self_attn.v_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention weights for layer 19
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.19.self_attn.o_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.19.self_attn.o_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention output weights for layer 19
[2025-09-07 09:21:30] [DEBUG] FFN weights memory requirement: gate/up=259MB, down=259MB for layer 19
[2025-09-07 09:21:30] [WARNING] FFN weights too large, using fallback sizes
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.19.mlp.gate_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.19.mlp.gate_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN gate weights for layer 19
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.19.mlp.up_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.19.mlp.up_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN up weights for layer 19
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.19.mlp.down_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.19.mlp.down_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN down weights for layer 19
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.19.input_layernorm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.19.input_layernorm.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention norm weights for layer 19
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.19.post_attention_layernorm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.19.post_attention_layernorm.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN norm weights for layer 19
[2025-09-07 09:21:30] [DEBUG] Layer 19 loaded - FFN gate: 8388608, up: 8388608, down: 8388608, attn heads: 1
[2025-09-07 09:21:30] [DEBUG] QKV weights memory requirement: 147MB for layer 20
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.20.self_attn.qkv_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.20.self_attn.qkv_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.20.self_attn.q_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.20.self_attn.q_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.20.self_attn.k_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.20.self_attn.k_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.20.self_attn.v_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.20.self_attn.v_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention weights for layer 20
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.20.self_attn.o_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.20.self_attn.o_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention output weights for layer 20
[2025-09-07 09:21:30] [DEBUG] FFN weights memory requirement: gate/up=259MB, down=259MB for layer 20
[2025-09-07 09:21:30] [WARNING] FFN weights too large, using fallback sizes
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.20.mlp.gate_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.20.mlp.gate_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN gate weights for layer 20
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.20.mlp.up_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.20.mlp.up_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN up weights for layer 20
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.20.mlp.down_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.20.mlp.down_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN down weights for layer 20
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.20.input_layernorm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.20.input_layernorm.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention norm weights for layer 20
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.20.post_attention_layernorm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.20.post_attention_layernorm.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN norm weights for layer 20
[2025-09-07 09:21:30] [DEBUG] Layer 20 loaded - FFN gate: 8388608, up: 8388608, down: 8388608, attn heads: 1
[2025-09-07 09:21:30] [DEBUG] QKV weights memory requirement: 147MB for layer 21
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.21.self_attn.qkv_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.21.self_attn.qkv_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.21.self_attn.q_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.21.self_attn.q_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.21.self_attn.k_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.21.self_attn.k_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.21.self_attn.v_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.21.self_attn.v_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention weights for layer 21
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.21.self_attn.o_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.21.self_attn.o_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention output weights for layer 21
[2025-09-07 09:21:30] [DEBUG] FFN weights memory requirement: gate/up=259MB, down=259MB for layer 21
[2025-09-07 09:21:30] [WARNING] FFN weights too large, using fallback sizes
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.21.mlp.gate_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.21.mlp.gate_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN gate weights for layer 21
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.21.mlp.up_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.21.mlp.up_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN up weights for layer 21
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.21.mlp.down_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.21.mlp.down_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN down weights for layer 21
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.21.input_layernorm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.21.input_layernorm.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention norm weights for layer 21
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.21.post_attention_layernorm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.21.post_attention_layernorm.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN norm weights for layer 21
[2025-09-07 09:21:30] [DEBUG] Layer 21 loaded - FFN gate: 8388608, up: 8388608, down: 8388608, attn heads: 1
[2025-09-07 09:21:30] [DEBUG] QKV weights memory requirement: 147MB for layer 22
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.22.self_attn.qkv_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.22.self_attn.qkv_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.22.self_attn.q_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.22.self_attn.q_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.22.self_attn.k_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.22.self_attn.k_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.22.self_attn.v_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.22.self_attn.v_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention weights for layer 22
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.22.self_attn.o_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.22.self_attn.o_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention output weights for layer 22
[2025-09-07 09:21:30] [DEBUG] FFN weights memory requirement: gate/up=259MB, down=259MB for layer 22
[2025-09-07 09:21:30] [WARNING] FFN weights too large, using fallback sizes
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.22.mlp.gate_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.22.mlp.gate_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN gate weights for layer 22
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.22.mlp.up_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.22.mlp.up_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN up weights for layer 22
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.22.mlp.down_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.22.mlp.down_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN down weights for layer 22
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.22.input_layernorm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.22.input_layernorm.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention norm weights for layer 22
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.22.post_attention_layernorm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.22.post_attention_layernorm.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN norm weights for layer 22
[2025-09-07 09:21:30] [DEBUG] Layer 22 loaded - FFN gate: 8388608, up: 8388608, down: 8388608, attn heads: 1
[2025-09-07 09:21:30] [DEBUG] QKV weights memory requirement: 147MB for layer 23
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.23.self_attn.qkv_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.23.self_attn.qkv_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.23.self_attn.q_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.23.self_attn.q_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.23.self_attn.k_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.23.self_attn.k_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.23.self_attn.v_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.23.self_attn.v_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention weights for layer 23
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.23.self_attn.o_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.23.self_attn.o_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention output weights for layer 23
[2025-09-07 09:21:30] [DEBUG] FFN weights memory requirement: gate/up=259MB, down=259MB for layer 23
[2025-09-07 09:21:30] [WARNING] FFN weights too large, using fallback sizes
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.23.mlp.gate_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.23.mlp.gate_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN gate weights for layer 23
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.23.mlp.up_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.23.mlp.up_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN up weights for layer 23
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.23.mlp.down_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.23.mlp.down_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN down weights for layer 23
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.23.input_layernorm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.23.input_layernorm.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention norm weights for layer 23
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.23.post_attention_layernorm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.23.post_attention_layernorm.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN norm weights for layer 23
[2025-09-07 09:21:30] [DEBUG] Layer 23 loaded - FFN gate: 8388608, up: 8388608, down: 8388608, attn heads: 1
[2025-09-07 09:21:30] [DEBUG] QKV weights memory requirement: 147MB for layer 24
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.24.self_attn.qkv_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.24.self_attn.qkv_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.24.self_attn.q_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.24.self_attn.q_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.24.self_attn.k_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.24.self_attn.k_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.24.self_attn.v_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.24.self_attn.v_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention weights for layer 24
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.24.self_attn.o_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.24.self_attn.o_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention output weights for layer 24
[2025-09-07 09:21:30] [DEBUG] FFN weights memory requirement: gate/up=259MB, down=259MB for layer 24
[2025-09-07 09:21:30] [WARNING] FFN weights too large, using fallback sizes
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.24.mlp.gate_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.24.mlp.gate_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN gate weights for layer 24
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.24.mlp.up_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.24.mlp.up_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN up weights for layer 24
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.24.mlp.down_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.24.mlp.down_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN down weights for layer 24
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.24.input_layernorm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.24.input_layernorm.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention norm weights for layer 24
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.24.post_attention_layernorm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.24.post_attention_layernorm.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN norm weights for layer 24
[2025-09-07 09:21:30] [DEBUG] Layer 24 loaded - FFN gate: 8388608, up: 8388608, down: 8388608, attn heads: 1
[2025-09-07 09:21:30] [DEBUG] QKV weights memory requirement: 147MB for layer 25
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.25.self_attn.qkv_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.25.self_attn.qkv_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.25.self_attn.q_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.25.self_attn.q_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.25.self_attn.k_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.25.self_attn.k_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.25.self_attn.v_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.25.self_attn.v_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention weights for layer 25
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.25.self_attn.o_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.25.self_attn.o_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention output weights for layer 25
[2025-09-07 09:21:30] [DEBUG] FFN weights memory requirement: gate/up=259MB, down=259MB for layer 25
[2025-09-07 09:21:30] [WARNING] FFN weights too large, using fallback sizes
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.25.mlp.gate_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.25.mlp.gate_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN gate weights for layer 25
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.25.mlp.up_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.25.mlp.up_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN up weights for layer 25
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.25.mlp.down_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.25.mlp.down_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN down weights for layer 25
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.25.input_layernorm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.25.input_layernorm.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention norm weights for layer 25
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.25.post_attention_layernorm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.25.post_attention_layernorm.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN norm weights for layer 25
[2025-09-07 09:21:30] [DEBUG] Layer 25 loaded - FFN gate: 8388608, up: 8388608, down: 8388608, attn heads: 1
[2025-09-07 09:21:30] [DEBUG] QKV weights memory requirement: 147MB for layer 26
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.26.self_attn.qkv_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.26.self_attn.qkv_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.26.self_attn.q_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.26.self_attn.q_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.26.self_attn.k_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.26.self_attn.k_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.26.self_attn.v_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.26.self_attn.v_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention weights for layer 26
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.26.self_attn.o_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.26.self_attn.o_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention output weights for layer 26
[2025-09-07 09:21:30] [DEBUG] FFN weights memory requirement: gate/up=259MB, down=259MB for layer 26
[2025-09-07 09:21:30] [WARNING] FFN weights too large, using fallback sizes
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.26.mlp.gate_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.26.mlp.gate_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN gate weights for layer 26
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.26.mlp.up_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.26.mlp.up_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN up weights for layer 26
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.26.mlp.down_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.26.mlp.down_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN down weights for layer 26
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.26.input_layernorm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.26.input_layernorm.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention norm weights for layer 26
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.26.post_attention_layernorm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.26.post_attention_layernorm.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN norm weights for layer 26
[2025-09-07 09:21:30] [DEBUG] Layer 26 loaded - FFN gate: 8388608, up: 8388608, down: 8388608, attn heads: 1
[2025-09-07 09:21:30] [DEBUG] QKV weights memory requirement: 147MB for layer 27
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.27.self_attn.qkv_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.27.self_attn.qkv_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.27.self_attn.q_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.27.self_attn.q_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.27.self_attn.k_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.27.self_attn.k_proj.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.27.self_attn.v_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.27.self_attn.v_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention weights for layer 27
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.27.self_attn.o_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.27.self_attn.o_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention output weights for layer 27
[2025-09-07 09:21:30] [DEBUG] FFN weights memory requirement: gate/up=259MB, down=259MB for layer 27
[2025-09-07 09:21:30] [WARNING] FFN weights too large, using fallback sizes
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.27.mlp.gate_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.27.mlp.gate_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN gate weights for layer 27
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.27.mlp.up_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.27.mlp.up_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN up weights for layer 27
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.27.mlp.down_proj.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.27.mlp.down_proj.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN down weights for layer 27
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.27.input_layernorm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.27.input_layernorm.weight
[2025-09-07 09:21:30] [WARNING] Failed to load attention norm weights for layer 27
[2025-09-07 09:21:30] [INFO] Loading tensor: model.layers.27.post_attention_layernorm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.layers.27.post_attention_layernorm.weight
[2025-09-07 09:21:30] [WARNING] Failed to load FFN norm weights for layer 27
[2025-09-07 09:21:30] [DEBUG] Layer 27 loaded - FFN gate: 8388608, up: 8388608, down: 8388608, attn heads: 1
[2025-09-07 09:21:30] [INFO] Transformer layers loaded successfully
[2025-09-07 09:21:30] [INFO] Loading output weights
[2025-09-07 09:21:30] [DEBUG] Output norm weights memory requirement: 0MB
[2025-09-07 09:21:30] [INFO] Loading tensor: model.norm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.norm.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: norm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: norm.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: ln_f.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: ln_f.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: transformer.ln_f.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: transformer.ln_f.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: transformer.norm.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: transformer.norm.weight
[2025-09-07 09:21:30] [WARNING] Failed to load output norm weights, using ones
[2025-09-07 09:21:30] [DEBUG] Output projection memory requirement: 2077MB
[2025-09-07 09:21:30] [DEBUG] vocab_size: 151936, hidden_size: 3584
[2025-09-07 09:21:30] [ERROR] Output projection too large (2077MB), exceeds limit (1024MB)
[2025-09-07 09:21:30] [INFO] Using fallback: smaller projection or memory mapping
[2025-09-07 09:21:30] [INFO] Using fallback vocab_size: 32000
[2025-09-07 09:21:30] [INFO] Loading tensor: lm_head.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: lm_head.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: model.lm_head.weight
[2025-09-07 09:21:30] [ERROR] Tensor not found in GGUF file: model.lm_head.weight
[2025-09-07 09:21:30] [INFO] Loading tensor: output.weight
[2025-09-07 09:21:30] [DEBUG] Tensor output.weight found, dimensions: 2
[2025-09-07 09:21:30] [DEBUG]   Dimension 0: 3584
[2025-09-07 09:21:30] [DEBUG]   Dimension 1: 152064
[2025-09-07 09:21:31] [DEBUG] Successfully read 104857600 bytes of raw data for tensor: output.weight
[2025-09-07 09:21:31] [ERROR] Unsupported tensor type for tensor: output.weight
[2025-09-07 09:21:31] [INFO] Loading tensor: embed_out.weight
[2025-09-07 09:21:31] [ERROR] Tensor not found in GGUF file: embed_out.weight
[2025-09-07 09:21:31] [INFO] Loading tensor: transformer.wte.weight
[2025-09-07 09:21:31] [ERROR] Tensor not found in GGUF file: transformer.wte.weight
[2025-09-07 09:21:31] [WARNING] Failed to load output projection weights from GGUF file
[2025-09-07 09:21:31] [INFO] Using shared token embeddings as output projection
[2025-09-07 09:21:31] [ERROR] Memory allocation failed in loadOutputWeights: std::bad_alloc
[2025-09-07 09:21:31] [ERROR] Failed to load model weights
[ERROR] OllamaModelManager: engine->loadModel() returned false for model: registry.ollama.ai_library_qwen2.5vl_7b
[2025-09-07 09:21:31] [INFO] Qwen25VLInferenceEngine destroyed
[ERROR] OllamaModelManager: Failed to load model: registry.ollama.ai_library_qwen2.5vl_7b
[ERROR] Failed to load Ollama model: registry.ollama.ai_library_qwen2.5vl_7b
[ERROR] Failed to load model: registry.ollama.ai/library/qwen2.5vl:7b (took 8101ms)
[DEBUG] Model details - Path: registry.ollama.ai/library/qwen2.5vl:7b, Type: 0, Memory limit: 4096MB
[INFO] OllamaModelManager: OllamaModelManager destroyed
```