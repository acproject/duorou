[DEBUG] splitToHeads input data size: 100352, tensor size: 100352
[DEBUG] splitToHeads input shape size: 3
[DEBUG] splitToHeads input shape: [1, 28, 512], num_heads=4, head_dim=128
[DEBUG] splitToHeads input data size: 14336, tensor size: 14336
[DEBUG] splitToHeads input shape size: 3
[DEBUG] splitToHeads input shape: [1, 28, 512], num_heads=4, head_dim=128
[DEBUG] splitToHeads input data size: 14336, tensor size: 14336
[DEBUG] splitKVCacheToHeads: cache shape [1, 4, 131072, 128]
[DEBUG] splitKVCacheToHeads: cache data size = 67108864
[DEBUG] Creating head tensor 0 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 1 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 2 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 3 with shape [1, 131072, 128]
[DEBUG] splitKVCacheToHeads: cache shape [1, 4, 131072, 128]
[DEBUG] splitKVCacheToHeads: cache data size = 67108864
[DEBUG] Creating head tensor 0 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 1 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 2 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 3 with shape [1, 131072, 128]
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
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
[DEBUG] Transformer layer 4 completed
[DEBUG] Processing transformer layer 5/28 (with KV cache)
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
[DEBUG] MultiHeadAttention::computeWithCache called
[DEBUG] Key cache shape: [1, 4, 131072, 128]
[DEBUG] Value cache shape: [1, 4, 131072, 128]
[DEBUG] splitToHeads input shape size: 3
[DEBUG] splitToHeads input shape: [1, 28, 3584], num_heads=28, head_dim=128
[DEBUG] splitToHeads input data size: 100352, tensor size: 100352
[DEBUG] splitToHeads input shape size: 3
[DEBUG] splitToHeads input shape: [1, 28, 512], num_heads=4, head_dim=128
[DEBUG] splitToHeads input data size: 14336, tensor size: 14336
[DEBUG] splitToHeads input shape size: 3
[DEBUG] splitToHeads input shape: [1, 28, 512], num_heads=4, head_dim=128
[DEBUG] splitToHeads input data size: 14336, tensor size: 14336
[DEBUG] splitKVCacheToHeads: cache shape [1, 4, 131072, 128]
[DEBUG] splitKVCacheToHeads: cache data size = 67108864
[DEBUG] Creating head tensor 0 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 1 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 2 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 3 with shape [1, 131072, 128]
[DEBUG] splitKVCacheToHeads: cache shape [1, 4, 131072, 128]
[DEBUG] splitKVCacheToHeads: cache data size = 67108864
[DEBUG] Creating head tensor 0 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 1 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 2 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 3 with shape [1, 131072, 128]
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
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
[DEBUG] Transformer layer 5 completed
[DEBUG] Processing transformer layer 6/28 (with KV cache)
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
[DEBUG] MultiHeadAttention::computeWithCache called
[DEBUG] Key cache shape: [1, 4, 131072, 128]
[DEBUG] Value cache shape: [1, 4, 131072, 128]
[DEBUG] splitToHeads input shape size: 3
[DEBUG] splitToHeads input shape: [1, 28, 3584], num_heads=28, head_dim=128
[DEBUG] splitToHeads input data size: 100352, tensor size: 100352
[DEBUG] splitToHeads input shape size: 3
[DEBUG] splitToHeads input shape: [1, 28, 512], num_heads=4, head_dim=128
[DEBUG] splitToHeads input data size: 14336, tensor size: 14336
[DEBUG] splitToHeads input shape size: 3
[DEBUG] splitToHeads input shape: [1, 28, 512], num_heads=4, head_dim=128
[DEBUG] splitToHeads input data size: 14336, tensor size: 14336
[DEBUG] splitKVCacheToHeads: cache shape [1, 4, 131072, 128]
[DEBUG] splitKVCacheToHeads: cache data size = 67108864
[DEBUG] Creating head tensor 0 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 1 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 2 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 3 with shape [1, 131072, 128]
[DEBUG] splitKVCacheToHeads: cache shape [1, 4, 131072, 128]
[DEBUG] splitKVCacheToHeads: cache data size = 67108864
[DEBUG] Creating head tensor 0 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 1 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 2 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 3 with shape [1, 131072, 128]
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
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
[DEBUG] Transformer layer 6 completed
[DEBUG] Processing transformer layer 7/28 (with KV cache)
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
[DEBUG] MultiHeadAttention::computeWithCache called
[DEBUG] Key cache shape: [1, 4, 131072, 128]
[DEBUG] Value cache shape: [1, 4, 131072, 128]
[DEBUG] splitToHeads input shape size: 3
[DEBUG] splitToHeads input shape: [1, 28, 3584], num_heads=28, head_dim=128
[DEBUG] splitToHeads input data size: 100352, tensor size: 100352
[DEBUG] splitToHeads input shape size: 3
[DEBUG] splitToHeads input shape: [1, 28, 512], num_heads=4, head_dim=128
[DEBUG] splitToHeads input data size: 14336, tensor size: 14336
[DEBUG] splitToHeads input shape size: 3
[DEBUG] splitToHeads input shape: [1, 28, 512], num_heads=4, head_dim=128
[DEBUG] splitToHeads input data size: 14336, tensor size: 14336
[DEBUG] splitKVCacheToHeads: cache shape [1, 4, 131072, 128]
[DEBUG] splitKVCacheToHeads: cache data size = 67108864
[DEBUG] Creating head tensor 0 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 1 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 2 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 3 with shape [1, 131072, 128]
[DEBUG] splitKVCacheToHeads: cache shape [1, 4, 131072, 128]
[DEBUG] splitKVCacheToHeads: cache data size = 67108864
[DEBUG] Creating head tensor 0 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 1 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 2 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 3 with shape [1, 131072, 128]
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
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
[DEBUG] Transformer layer 7 completed
[DEBUG] Processing transformer layer 8/28 (with KV cache)
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
[DEBUG] MultiHeadAttention::computeWithCache called
[DEBUG] Key cache shape: [1, 4, 131072, 128]
[DEBUG] Value cache shape: [1, 4, 131072, 128]
[DEBUG] splitToHeads input shape size: 3
[DEBUG] splitToHeads input shape: [1, 28, 3584], num_heads=28, head_dim=128
[DEBUG] splitToHeads input data size: 100352, tensor size: 100352
[DEBUG] splitToHeads input shape size: 3
[DEBUG] splitToHeads input shape: [1, 28, 512], num_heads=4, head_dim=128
[DEBUG] splitToHeads input data size: 14336, tensor size: 14336
[DEBUG] splitToHeads input shape size: 3
[DEBUG] splitToHeads input shape: [1, 28, 512], num_heads=4, head_dim=128
[DEBUG] splitToHeads input data size: 14336, tensor size: 14336
[DEBUG] splitKVCacheToHeads: cache shape [1, 4, 131072, 128]
[DEBUG] splitKVCacheToHeads: cache data size = 67108864
[DEBUG] Creating head tensor 0 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 1 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 2 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 3 with shape [1, 131072, 128]
[DEBUG] splitKVCacheToHeads: cache shape [1, 4, 131072, 128]
[DEBUG] splitKVCacheToHeads: cache data size = 67108864
[DEBUG] Creating head tensor 0 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 1 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 2 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 3 with shape [1, 131072, 128]
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
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
[DEBUG] Transformer layer 8 completed
[DEBUG] Processing transformer layer 9/28 (with KV cache)
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
[DEBUG] MultiHeadAttention::computeWithCache called
[DEBUG] Key cache shape: [1, 4, 131072, 128]
[DEBUG] Value cache shape: [1, 4, 131072, 128]
[DEBUG] splitToHeads input shape size: 3
[DEBUG] splitToHeads input shape: [1, 28, 3584], num_heads=28, head_dim=128
[DEBUG] splitToHeads input data size: 100352, tensor size: 100352
[DEBUG] splitToHeads input shape size: 3
[DEBUG] splitToHeads input shape: [1, 28, 512], num_heads=4, head_dim=128
[DEBUG] splitToHeads input data size: 14336, tensor size: 14336
[DEBUG] splitToHeads input shape size: 3
[DEBUG] splitToHeads input shape: [1, 28, 512], num_heads=4, head_dim=128
[DEBUG] splitToHeads input data size: 14336, tensor size: 14336
[DEBUG] splitKVCacheToHeads: cache shape [1, 4, 131072, 128]
[DEBUG] splitKVCacheToHeads: cache data size = 67108864
[DEBUG] Creating head tensor 0 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 1 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 2 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 3 with shape [1, 131072, 128]
[DEBUG] splitKVCacheToHeads: cache shape [1, 4, 131072, 128]
[DEBUG] splitKVCacheToHeads: cache data size = 67108864
[DEBUG] Creating head tensor 0 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 1 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 2 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 3 with shape [1, 131072, 128]
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
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
[DEBUG] Transformer layer 9 completed
[DEBUG] Processing transformer layer 10/28 (with KV cache)
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
[DEBUG] MultiHeadAttention::computeWithCache called
[DEBUG] Key cache shape: [1, 4, 131072, 128]
[DEBUG] Value cache shape: [1, 4, 131072, 128]
[DEBUG] splitToHeads input shape size: 3
[DEBUG] splitToHeads input shape: [1, 28, 3584], num_heads=28, head_dim=128
[DEBUG] splitToHeads input data size: 100352, tensor size: 100352
[DEBUG] splitToHeads input shape size: 3
[DEBUG] splitToHeads input shape: [1, 28, 512], num_heads=4, head_dim=128
[DEBUG] splitToHeads input data size: 14336, tensor size: 14336
[DEBUG] splitToHeads input shape size: 3
[DEBUG] splitToHeads input shape: [1, 28, 512], num_heads=4, head_dim=128
[DEBUG] splitToHeads input data size: 14336, tensor size: 14336
[DEBUG] splitKVCacheToHeads: cache shape [1, 4, 131072, 128]
[DEBUG] splitKVCacheToHeads: cache data size = 67108864
[DEBUG] Creating head tensor 0 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 1 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 2 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 3 with shape [1, 131072, 128]
[DEBUG] splitKVCacheToHeads: cache shape [1, 4, 131072, 128]
[DEBUG] splitKVCacheToHeads: cache data size = 67108864
[DEBUG] Creating head tensor 0 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 1 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 2 with shape [1, 131072, 128]
[DEBUG] Creating head tensor 3 with shape [1, 131072, 128]
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
[DEBUG] FastAttention::compute called
[DEBUG] FastAttention input validation passed
[DEBUG] KV Cache updated at position 0
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
