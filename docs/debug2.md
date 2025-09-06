目前的multiHeadAttention（多头注意力）算法不是特别好
我们可以借助llama.cpp中的思想，用高效的 GEMM (General Matrix Multiply) 来替代，
把 Q、K、V 展开成大矩阵：
输入形状: (n_tokens, d_model)
映射到 Q, K, V:
Q, K, V 形状是 (n_tokens, n_heads, head_dim)这样就能批量处理每个 head 的计算，而不是 head-by-head。

计算注意力分数 (QK^T)lama.cpp 用一个高效的 矩阵乘法例程（内部用 ggml_mul_mat / BLAS / CUDA kernel），一次性完成 Q × K^T，得到 (n_heads, n_tokens, n_tokens) 的注意力分数张量，这一步等价于内层的 token_i × token_j 点积，但用矩阵乘法批量完成了

softmax + dropout对每个 head、每个 query token，做 softmax 归一化，在 CPU 后端就是 逐行 softmax，在 GPU 上会 fuse 到 kernel

与 V 相乘，也是通过 矩阵乘法：
内部调用 ggml_mul_mat，批量完成
优化：
RoPE (旋转位置编码)
在计算 Q、K 之前对向量分块做复数旋转，替代绝对位置编码

KV Cache
在推理时，历史 token 的 K、V 会被缓存，不需要每一步重新计算，只增量更新

Flash Attention-like 优化
在某些后端（CUDA/OpenCL）里，llama.cpp 借鉴了 Flash Attention 的思想：

按 block 处理 QK^T，避免存储完整 n_tokens × n_tokens 矩阵

边算边做 softmax + V 乘法，减少内存带宽