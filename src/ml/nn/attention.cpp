#include "attention.h"
#include <cmath>
#include <stdexcept>
#include <cstring>
#include <limits>
#include "../backend/backend.h"
#include "core/logger.h"

namespace duorou {
namespace ml {
namespace nn {

// MultiHeadAttention implementation
MultiHeadAttention::MultiHeadAttention(int64_t embedDim, int64_t numHeads, 
                                     int64_t kvHeads, bool bias, float dropout)
    : embedDim_(embedDim), numHeads_(numHeads), 
      kvHeads_(kvHeads == -1 ? numHeads : kvHeads), 
      headDim_(embedDim / numHeads), hasBias_(bias), dropout_(dropout),
      // project Q to [numHeads * headDim]
      queryWeight_({embedDim, numHeads_ * headDim_}),
      // project K/V to [kvHeads * headDim]
      keyWeight_({embedDim, kvHeads_ * headDim_}),
      valueWeight_({embedDim, kvHeads_ * headDim_}),
      // output projection from concatenated heads back to embedDim
      outputWeight_({numHeads_ * headDim_, embedDim}) {
    if (embedDim % numHeads != 0) {
        throw std::invalid_argument("MultiHeadAttention: embedDim must be divisible by numHeads");
    }
    if (hasBias_) {
        queryBias_ = Tensor({numHeads_ * headDim_});
        keyBias_ = Tensor({kvHeads_ * headDim_});
        valueBias_ = Tensor({kvHeads_ * headDim_});
        outputBias_ = Tensor({embedDim});
    }
}

Tensor MultiHeadAttention::forward(Context& ctx, const Tensor& query, 
                                  const Tensor& key, const Tensor& value,
                                  kvcache::Cache* cache, const Tensor& mask) {
    static duorou::core::Logger logger;
    // Validate dtype
    if (query.dtype() != DataType::FLOAT32) {
        throw std::runtime_error("MultiHeadAttention::forward: only FLOAT32 supported");
    }
    if (kvHeads_ != numHeads_) {
        // For now, simplify: require kvHeads == numHeads
        throw std::runtime_error("MultiHeadAttention::forward: kvHeads must equal numHeads in current implementation");
    }
    // Determine input shape
    auto qShape = query.shape();
    bool is3D = qShape.size() == 3;
    int64_t B = is3D ? qShape[0] : 1;
    int64_t Sq = is3D ? qShape[1] : qShape[0];
    int64_t E = is3D ? qShape[2] : qShape[1];

    // Default key/value to query when not provided
    const Tensor& keyRef = (key.data() ? key : query);
    const Tensor& valueRef = (value.data() ? value : keyRef);
    auto kShape = keyRef.shape();
    bool kIs3D = kShape.size() == 3;
    int64_t Sk = kIs3D ? kShape[1] : kShape[0];

    // Reshape inputs to 2D for linear projections
    Tensor query2D = is3D ? query.reshape({B * Sq, E}) : query;
    Tensor key2D   = kIs3D ? keyRef.reshape({B * Sk, E}) : keyRef;
    Tensor value2D = kIs3D ? valueRef.reshape({B * Sk, E}) : valueRef;

    // Linear projections
    Tensor qProj = query2D.matmul(ctx, queryWeight_); // [B*Sq, H*D]
    Tensor kProj = key2D.matmul(ctx, keyWeight_);     // [B*Sk, H*D]
    Tensor vProj = value2D.matmul(ctx, valueWeight_); // [B*Sk, H*D]

    if (hasBias_) {
        qProj = qProj.add(ctx, queryBias_);
        kProj = kProj.add(ctx, keyBias_);
        vProj = vProj.add(ctx, valueBias_);
    }

    // Reshape to 4D: [B, S, H, D]
    Tensor q4 = qProj.reshape({B, Sq, numHeads_, headDim_});
    Tensor k4 = kProj.reshape({B, Sk, numHeads_, headDim_});
    Tensor v4 = vProj.reshape({B, Sk, numHeads_, headDim_});
    logger.debug("[MHA] Shapes after projection: q4=[" + std::to_string(B) + "," + std::to_string(Sq) + "," + std::to_string(numHeads_) + "," + std::to_string(headDim_) + "] k4=[" + std::to_string(B) + "," + std::to_string(Sk) + "," + std::to_string(numHeads_) + "," + std::to_string(headDim_) + "] v4=[" + std::to_string(B) + "," + std::to_string(Sk) + "," + std::to_string(numHeads_) + "," + std::to_string(headDim_) + "]");

    // Prepare KV Cache integration: fetch previous K/V and concatenate
    int64_t prevLen = 0;
    // static duorou::core::Logger logger; // removed duplicate
    if (cache) {
        // Local adapter bridging ml::Backend to kvcache::Backend
        struct MLKVBackendAdapter : public duorou::kvcache::Backend {
            explicit MLKVBackendAdapter(duorou::ml::Backend* b) : mlBackend(b) {}
            void* allocate(size_t bytes) override {
                if (mlBackend) return mlBackend->allocate(bytes);
                return std::malloc(bytes);
            }
            void deallocate(void* ptr) override {
                if (!ptr) return;
                if (mlBackend) mlBackend->deallocate(ptr);
                else std::free(ptr);
            }
            void copy(void* dst, const void* src, size_t bytes) override {
                if (!dst || !src || bytes == 0) return;
                if (mlBackend) mlBackend->copyDeviceToDevice(dst, src, bytes);
                else std::memcpy(dst, src, bytes);
            }
            duorou::ml::Backend* mlBackend;
        } adapter(ctx.getBackend());
        duorou::kvcache::Context kctx(&adapter);

        // Retrieve previous cached K/V for sequence 0 (single-seq scenario)
        auto kvPrev = cache->get(kctx, /*seq=*/0, /*startPos=*/0, /*endPos=*/std::numeric_limits<int32_t>::max());
        duorou::kvcache::Tensor kPrevKV = std::get<0>(kvPrev);
        duorou::kvcache::Tensor vPrevKV = std::get<1>(kvPrev);

        // If previous cache exists and is valid, concatenate
        if (kPrevKV.data() && kPrevKV.bytesSize() > 0) {
            // Map shape to int64_t
            std::vector<int64_t> prevShape64;
            for (int dim : kPrevKV.shape()) prevShape64.push_back(static_cast<int64_t>(dim));
            if (prevShape64.size() == 4) {
                prevLen = prevShape64[1];
                size_t kvPrevBytes = static_cast<size_t>(kPrevKV.bytesSize());

                // Apply RoPE to current K with offset equal to previous length
                k4 = applyRotaryPositionEmbedding(ctx, k4, Sk, /*offset=*/prevLen);

                // Concatenate along sequence dimension: [B, prevLen + Sk, H, D]
                int64_t totalSk = prevLen + Sk;
                Tensor kFull({B, totalSk, numHeads_, headDim_}, DataType::FLOAT32);
                Tensor vFull({B, totalSk, numHeads_, headDim_}, DataType::FLOAT32);
                if (auto* b = ctx.getBackend()) { kFull.setBackend(b); vFull.setBackend(b); }
                kFull.allocate();
                vFull.allocate();

                size_t prevBytes = static_cast<size_t>(B * prevLen * numHeads_ * headDim_) * sizeof(float);
                size_t newBytes  = static_cast<size_t>(B * Sk * numHeads_ * headDim_) * sizeof(float);
                size_t k4Bytes = static_cast<size_t>(k4.nbytes());
                size_t v4Bytes = static_cast<size_t>(v4.nbytes());

                // Sanity checks to avoid out-of-bounds copies
                if (prevBytes > kvPrevBytes) {
                    logger.debug("[MHA] KV concat prevBytes exceeds previous cache tensor bytes; skipping concat to prevent OOB");
                } else if (newBytes > k4Bytes || newBytes > v4Bytes) {
                    logger.debug("[MHA] KV concat newBytes exceeds current K/V tensor bytes; skipping concat to prevent OOB");
                } else {
                    // Copy previous part directly from cache tensors
                    adapter.copy(kFull.data(), kPrevKV.data(), prevBytes);
                    adapter.copy(vFull.data(), vPrevKV.data(), prevBytes);
                    // Copy new part right after previous
                    void* kDst = static_cast<void*>(static_cast<char*>(kFull.data()) + prevBytes);
                    void* vDst = static_cast<void*>(static_cast<char*>(vFull.data()) + prevBytes);
                    adapter.copy(kDst, k4.data(), newBytes);
                    adapter.copy(vDst, v4.data(), newBytes);

                    // Update k4/v4 and Sk to total length
                    k4 = kFull;
                    v4 = vFull;
                    Sk = totalSk;
                    logger.debug("[MHA] KV concat done: B=" + std::to_string(B) + 
                                  ", prevLen=" + std::to_string(prevLen) +
                                  ", newSk=" + std::to_string((kIs3D ? keyRef.shape()[1] : keyRef.shape()[0])) +
                                  ", totalSk=" + std::to_string(Sk) +
                                  ", bytes(prev,new)=" + std::to_string(prevBytes) + "," + std::to_string(newBytes));
                }
            } else {
                // No valid previous shape, apply default RoPE for K
                k4 = applyRotaryPositionEmbedding(ctx, k4, Sk, /*offset=*/0);
            }
        } else {
            // No previous cache; apply default RoPE for K
            k4 = applyRotaryPositionEmbedding(ctx, k4, Sk, /*offset=*/0);
        }
    } else {
        // No cache provided; apply default RoPE for K
        k4 = applyRotaryPositionEmbedding(ctx, k4, Sk, /*offset=*/0);
    }

    // Apply RoPE to Q with offset equal to number of previous tokens
    q4 = applyRotaryPositionEmbedding(ctx, q4, Sq, /*offset=*/prevLen);
    logger.debug("[MHA] RoPE applied: prevLen=" + std::to_string(prevLen) + 
                 ", q4(seqLen)=" + std::to_string(Sq) + ", k4(seqLen)=" + std::to_string(Sk));

    // If cache exists, store K, V using backend adapter bridging
    if (cache && k4.data() && v4.data()) {
        // Local adapter bridging ml::Backend to kvcache::Backend
        struct MLKVBackendAdapter : public duorou::kvcache::Backend {
            explicit MLKVBackendAdapter(duorou::ml::Backend* b) : mlBackend(b) {}
            void* allocate(size_t bytes) override {
                if (mlBackend) return mlBackend->allocate(bytes);
                return std::malloc(bytes);
            }
            void deallocate(void* ptr) override {
                if (!ptr) return;
                if (mlBackend) mlBackend->deallocate(ptr);
                else std::free(ptr);
            }
            void copy(void* dst, const void* src, size_t bytes) override {
                if (!dst || !src || bytes == 0) return;
                if (mlBackend) mlBackend->copyDeviceToDevice(dst, src, bytes);
                else std::memcpy(dst, src, bytes);
            }
            duorou::ml::Backend* mlBackend;
        } adapter(ctx.getBackend());

        duorou::kvcache::Context kctx(&adapter);

        // Build kvcache tensors for the NEW segment only [B, newSk, H, D]
        int64_t newSk = is3D ? keyRef.shape()[1] : keyRef.shape()[0];
        std::vector<int> kvShape = {static_cast<int>(B), static_cast<int>(newSk), static_cast<int>(numHeads_), static_cast<int>(headDim_)};
        duorou::kvcache::Tensor kKV(kvShape, duorou::kvcache::DType::FLOAT32, &adapter);
        duorou::kvcache::Tensor vKV(kvShape, duorou::kvcache::DType::FLOAT32, &adapter);

        // Copy data from ml::Tensor to kvcache::Tensor (ONLY new segment)
        size_t newBytes = static_cast<size_t>(B * newSk * numHeads_ * headDim_) * sizeof(float);
        // Offset source by previous length if k4/v4 contain concatenated [prevLen + newSk]
        size_t prevBytes = static_cast<size_t>(B * prevLen * numHeads_ * headDim_) * sizeof(float);
        const void* kNewSrc = (k4.data() ? static_cast<const void*>(static_cast<const char*>(k4.data()) + prevBytes) : nullptr);
        const void* vNewSrc = (v4.data() ? static_cast<const void*>(static_cast<const char*>(v4.data()) + prevBytes) : nullptr);
        if (kNewSrc == nullptr || vNewSrc == nullptr) {
            logger.debug("[MHA] KV put skipped: null kNewSrc/vNewSrc");
        } else if (prevBytes + newBytes > static_cast<size_t>(k4.nbytes()) || prevBytes + newBytes > static_cast<size_t>(v4.nbytes())) {
            logger.debug("[MHA] KV put skipped: (prevBytes + newBytes) exceeds k4/v4 bytes; prevBytes=" + std::to_string(prevBytes) +
                         ", newBytes=" + std::to_string(newBytes) +
                         ", k4Bytes=" + std::to_string(static_cast<size_t>(k4.nbytes())) +
                         ", v4Bytes=" + std::to_string(static_cast<size_t>(v4.nbytes())));
        } else {
            adapter.copy(kKV.data(), kNewSrc, newBytes);
            adapter.copy(vKV.data(), vNewSrc, newBytes);
            logger.debug("[MHA] KV put new segment: B=" + std::to_string(B) +
                         ", newSk=" + std::to_string(newSk) +
                         ", bytes=" + std::to_string(newBytes) +
                         ", prevOffsetBytes=" + std::to_string(prevBytes));
        }

        // Store into cache
        cache->put(kctx, kKV, vKV);
    }

    // Build effective attention mask: if caller didn't provide one, create causal mask
    Tensor usedMask;
    if (mask.data() == nullptr) {
        // 2D mask [Sq, Sk]: allow attending to all cached tokens (prevLen) and up to current index within new segment
        usedMask = Tensor({Sq, Sk}, DataType::FLOAT32);
        usedMask.allocate();
        // Fill mask with 0 for allowed, -inf for disallowed
        std::vector<float> m(static_cast<size_t>(Sq * Sk), 0.0f);
        for (int64_t s = 0; s < Sq; ++s) {
            int64_t allowed = prevLen + s + 1; // number of key positions allowed for this query
            if (allowed > Sk) allowed = Sk;
            // disallow positions t >= allowed
            for (int64_t t = allowed; t < Sk; ++t) {
                m[static_cast<size_t>(s * Sk + t)] = -std::numeric_limits<float>::infinity();
            }
        }
        usedMask.copyFromHost(m.data(), m.size() * sizeof(float));
    } else {
        usedMask = mask;
    }

    // Execute attention computation in 4D
    logger.debug("[MHA] Calling scaledDotProductAttention with q4=[" + std::to_string(B) + "," + std::to_string(Sq) + "," + std::to_string(numHeads_) + "," + std::to_string(headDim_) + "] k4=[" + std::to_string(B) + "," + std::to_string(Sk) + "," + std::to_string(numHeads_) + "," + std::to_string(headDim_) + "] v4=[" + std::to_string(B) + "," + std::to_string(Sk) + "," + std::to_string(numHeads_) + "," + std::to_string(headDim_) + "]");
    Tensor attnOut4 = scaledDotProductAttention(ctx, q4, k4, v4, usedMask); // [B,Sq,H,D]

    // Merge heads: [B,Sq,H,D] -> [B*Sq, H*D]
    Tensor merged = attnOut4.reshape({B * Sq, numHeads_ * headDim_});

    // Output projection back to embedDim
    Tensor output2D = merged.matmul(ctx, outputWeight_); // [B*Sq, E]
    if (hasBias_) {
        output2D = output2D.add(ctx, outputBias_);
    }

    // Reshape back to original rank
    Tensor output = is3D ? output2D.reshape({B, Sq, E}) : output2D;
    return output;
}

Tensor MultiHeadAttention::forwardWithSinks(Context& ctx, const Tensor& query,
                                           const Tensor& key, const Tensor& value,
                                           const Tensor& sinks, float scale,
                                           kvcache::Cache* cache) {
    (void)sinks; // suppress unused parameter warning for sinks
    (void)scale; // suppress unused parameter warning for scale
    // This is a simplified implementation, actually needs to handle sink tokens
    return forward(ctx, query, key, value, cache);
}

void MultiHeadAttention::initializeWeights(Context& ctx, const std::string& method) {
    (void)ctx; // currently unused
    (void)method; // suppress unused warning
    // Initialize all weight matrices
    queryWeight_.allocate();
    keyWeight_.allocate();
    valueWeight_.allocate();
    outputWeight_.allocate();
    if (hasBias_) {
        queryBias_.allocate();
        keyBias_.allocate();
        valueBias_.allocate();
        outputBias_.allocate();
    }
    // Xavier or other initialization can be applied here if needed
}

bool MultiHeadAttention::setWeights(Context& ctx,
                    const std::vector<float>& qW,
                    const std::vector<float>& kW,
                    const std::vector<float>& vW,
                    const std::vector<float>& oW,
                    const std::vector<float>* qB,
                    const std::vector<float>* kB,
                    const std::vector<float>* vB,
                    const std::vector<float>* oB) {
    // Basic size checks
    const auto qShape = queryWeight_.shape();
    const auto kShape = keyWeight_.shape();
    const auto vShape = valueWeight_.shape();
    const auto oShape = outputWeight_.shape();
    const size_t qExpected = static_cast<size_t>(qShape[0] * qShape[1]);
    const size_t kExpected = static_cast<size_t>(kShape[0] * kShape[1]);
    const size_t vExpected = static_cast<size_t>(vShape[0] * vShape[1]);
    const size_t oExpected = static_cast<size_t>(oShape[0] * oShape[1]);
    if (qW.size() != qExpected || kW.size() != kExpected || vW.size() != vExpected || oW.size() != oExpected) {
        return false;
    }
    // Attach backend and allocate
    auto *backend = ctx.getBackend();
    queryWeight_.setBackend(backend);
    keyWeight_.setBackend(backend);
    valueWeight_.setBackend(backend);
    outputWeight_.setBackend(backend);
    queryWeight_.allocate();
    keyWeight_.allocate();
    valueWeight_.allocate();
    outputWeight_.allocate();
    // Copy
    queryWeight_.copyFromHost(qW.data(), qW.size() * sizeof(float));
    keyWeight_.copyFromHost(kW.data(), kW.size() * sizeof(float));
    valueWeight_.copyFromHost(vW.data(), vW.size() * sizeof(float));
    outputWeight_.copyFromHost(oW.data(), oW.size() * sizeof(float));
    // Biases if provided and enabled
    if (hasBias_) {
        // query bias
        queryBias_.setBackend(backend);
        queryBias_.allocate();
        if (qB) {
            queryBias_.copyFromHost(qB->data(), qB->size() * sizeof(float));
        } else {
            // zero-init
            std::vector<float> zeros(static_cast<size_t>(queryBias_.numel()), 0.0f);
            queryBias_.copyFromHost(zeros.data(), zeros.size() * sizeof(float));
        }
        // key bias
        keyBias_.setBackend(backend);
        keyBias_.allocate();
        if (kB) {
            keyBias_.copyFromHost(kB->data(), kB->size() * sizeof(float));
        } else {
            std::vector<float> zeros(static_cast<size_t>(keyBias_.numel()), 0.0f);
            keyBias_.copyFromHost(zeros.data(), zeros.size() * sizeof(float));
        }
        // value bias
        valueBias_.setBackend(backend);
        valueBias_.allocate();
        if (vB) {
            valueBias_.copyFromHost(vB->data(), vB->size() * sizeof(float));
        } else {
            std::vector<float> zeros(static_cast<size_t>(valueBias_.numel()), 0.0f);
            valueBias_.copyFromHost(zeros.data(), zeros.size() * sizeof(float));
        }
        // output bias
        outputBias_.setBackend(backend);
        outputBias_.allocate();
        if (oB) {
            outputBias_.copyFromHost(oB->data(), oB->size() * sizeof(float));
        } else {
            std::vector<float> zeros(static_cast<size_t>(outputBias_.numel()), 0.0f);
            outputBias_.copyFromHost(zeros.data(), zeros.size() * sizeof(float));
        }
    }
    return true;
}

Tensor MultiHeadAttention::scaledDotProductAttention(Context& ctx, const Tensor& q, 
                                                   const Tensor& k, const Tensor& v,
                                                   const Tensor& mask) {
    (void)ctx; // currently unused in CPU reference implementation
    // q,k,v are 4D: [B, S, H, D]
    auto qs = q.shape();
    auto ks = k.shape();
    auto vs = v.shape();
    if (qs.size() != 4 || ks.size() != 4 || vs.size() != 4) {
        throw std::runtime_error("scaledDotProductAttention: expected 4D tensors [B,S,H,D]");
    }
    int64_t B = qs[0];
    int64_t Sq = qs[1];
    int64_t H = qs[2];
    int64_t D = qs[3];
    int64_t Sk = ks[1];
    if (ks[0] != B || ks[2] != H || ks[3] != D || vs[0] != B || vs[1] != Sk || vs[2] != H || vs[3] != D) {
        throw std::runtime_error("scaledDotProductAttention: q,k,v shape mismatch");
    }
    if (q.dtype() != DataType::FLOAT32 || k.dtype() != DataType::FLOAT32 || v.dtype() != DataType::FLOAT32) {
        throw std::runtime_error("scaledDotProductAttention: only FLOAT32 supported");
    }

    Tensor out({B, Sq, H, D}, DataType::FLOAT32);
    out.allocate();

    const float* qPtr = q.data<float>();
    const float* kPtr = k.data<float>();
    const float* vPtr = v.data<float>();
    float* outPtr = out.data<float>();

    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // Optional mask
    const bool hasMask = mask.data() != nullptr;
    std::vector<int64_t> ms;
    bool maskIsBool = false;
    const float* mF = nullptr;
    const bool* mB = nullptr;
    if (hasMask) {
        ms = mask.shape();
        maskIsBool = (mask.dtype() == DataType::BOOL);
        if (maskIsBool) {
            mB = mask.data<bool>();
        } else {
            mF = mask.data<float>();
        }
        // Supported mask shapes: [B,Sq,Sk] or [Sq,Sk]
        if (!(ms.size() == 3 || ms.size() == 2)) {
            throw std::runtime_error("scaledDotProductAttention: unsupported mask rank");
        }
        if (ms.size() == 3 && (ms[0] != B || ms[1] != Sq || ms[2] != Sk)) {
            throw std::runtime_error("scaledDotProductAttention: mask shape [B,Sq,Sk] mismatch");
        }
        if (ms.size() == 2 && (ms[0] != Sq || ms[1] != Sk)) {
            throw std::runtime_error("scaledDotProductAttention: mask shape [Sq,Sk] mismatch");
        }
    }

    // Helper lambdas for indexing
    auto idxQ = [&](int64_t b, int64_t s, int64_t h, int64_t d) -> int64_t {
        return (((b * Sq + s) * H + h) * D + d);
    };
    auto idxK = [&](int64_t b, int64_t s, int64_t h, int64_t d) -> int64_t {
        return (((b * Sk + s) * H + h) * D + d);
    };
    auto idxV = idxK;
    auto idxO = idxQ;

    // Compute attention for each batch and head
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            // For each query position s in Sq, compute scores against all key positions
            for (int64_t s = 0; s < Sq; ++s) {
                // 1) scores[t] = dot(q[b,s,h,:], k[b,t,h,:])
                std::vector<float> scores(static_cast<size_t>(Sk), 0.0f);
                for (int64_t t = 0; t < Sk; ++t) {
                    float dot = 0.0f;
                    for (int64_t d = 0; d < D; ++d) {
                        dot += qPtr[idxQ(b, s, h, d)] * kPtr[idxK(b, t, h, d)];
                    }
                    dot *= scale;
                    // Apply mask if exists
                    if (hasMask) {
                        if (ms.size() == 3) {
                            int64_t mi = (b * Sq + s) * Sk + t;
                            if (maskIsBool) {
                                if (!mB[mi]) dot = -std::numeric_limits<float>::infinity();
                            } else {
                                dot += mF[mi];
                            }
                        } else {
                            int64_t mi = s * Sk + t;
                            if (maskIsBool) {
                                if (!mB[mi]) dot = -std::numeric_limits<float>::infinity();
                            } else {
                                dot += mF[mi];
                            }
                        }
                    }
                    scores[static_cast<size_t>(t)] = dot;
                }

                // 2) stable softmax over scores
                float maxScore = -std::numeric_limits<float>::infinity();
                for (float sc : scores) maxScore = std::max(maxScore, sc);
                float sumExp = 0.0f;
                for (float sc : scores) sumExp += std::exp(sc - maxScore);
                if (!std::isfinite(sumExp) || sumExp == 0.0f) {
                    // Log anomaly but continue to avoid crash; output will be zeros
                    // Note: no logger here; keep minimal overhead
                }
                // 3) weighted sum of V
                for (int64_t d = 0; d < D; ++d) {
                    float acc = 0.0f;
                    for (int64_t t = 0; t < Sk; ++t) {
                        float w = std::exp(scores[static_cast<size_t>(t)] - maxScore);
                        if (sumExp > 0.0f && std::isfinite(w)) {
                            w /= sumExp;
                        } else {
                            w = 0.0f;
                        }
                        acc += w * vPtr[idxV(b, t, h, d)];
                    }
                    outPtr[idxO(b, s, h, d)] = acc;
                }
            }
        }
    }

    return out;
}

Tensor MultiHeadAttention::applyRotaryPositionEmbedding(Context& ctx, const Tensor& tensor, 
                                                      int64_t seqLen, int64_t offset) {
    (void)ctx; // CPU reference implementation doesn't rely on backend here
    auto s = tensor.shape();
    if (s.size() != 4) {
        throw std::runtime_error("applyRotaryPositionEmbedding: expected 4D tensor [B,S,H,D]");
    }
    int64_t B = s[0];
    int64_t S = s[1];
    int64_t H = s[2];
    int64_t D = s[3];

    Tensor out({B, S, H, D}, DataType::FLOAT32);
    out.allocate();

    const float* x = tensor.data<float>();
    float* outPtr = out.data<float>();

    // Precompute invFreq for first D/2 dims
    int64_t half = D / 2;
    std::vector<float> invFreq(static_cast<size_t>(half));
    for (int64_t i = 0; i < half; ++i) {
        invFreq[static_cast<size_t>(i)] = 1.0f / std::pow(10000.0f, static_cast<float>(i) / static_cast<float>(half));
    }

    auto idx = [&](int64_t b, int64_t s, int64_t h, int64_t d) -> int64_t {
        return (((b * S + s) * H + h) * D + d);
    };

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t sPos = 0; sPos < S; ++sPos) {
            int64_t pos = offset + sPos;
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t i = 0; i < half; ++i) {
                    float freq = invFreq[static_cast<size_t>(i)] * static_cast<float>(pos);
                    float c = std::cos(freq);
                    float s = std::sin(freq);
                    float x0 = x[idx(b, sPos, h, i)];
                    float x1 = x[idx(b, sPos, h, i + half)];
                    outPtr[idx(b, sPos, h, i)] = x0 * c - x1 * s;
                    outPtr[idx(b, sPos, h, i + half)] = x0 * s + x1 * c;
                }
            }
        }
    }

    return out;
}

Tensor attention(Context& ctx, const Tensor& query, const Tensor& key, 
                const Tensor& value, float scale, kvcache::Cache* cache) {
    (void)scale; // scale integrated into scaledDotProductAttention
    MultiHeadAttention mha(query.dim(-1), /*numHeads=*/1);
    return mha.forward(ctx, query, key, value, cache);
}

Tensor attentionWithSinks(Context& ctx, const Tensor& query, const Tensor& key,
                         const Tensor& value, const Tensor& sinks, float scale,
                         kvcache::Cache* cache) {
    (void)sinks;
    (void)scale;
    MultiHeadAttention mha(query.dim(-1), /*numHeads=*/1);
    return mha.forward(ctx, query, key, value, cache);
}

} // namespace nn
} // namespace ml
} // namespace duorou