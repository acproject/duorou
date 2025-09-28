#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <cstdarg>
#include <thread>
#include <algorithm>
#include <iostream>

// ggml headers
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpu.h"
#include "core/text_generator.h"
#include "extensions/ollama/ollama_model_manager.h"

// Simple logging helper
static void logi(const char * fmt, ...) {
    va_list args; va_start(args, fmt);
    vfprintf(stdout, fmt, args);
    fprintf(stdout, "\n");
    va_end(args);
}

// Load a tensor by name from a GGUF file into a contiguous float32 buffer
// Returns empty vector on failure
static std::vector<float> load_tensor_f32_from_gguf(const char * path, const char * name) {
    std::vector<float> out;

    // Build a ggml context with tensor data loaded so we can access type and dimensions
    struct ggml_context * ctx_data = nullptr;
    struct gguf_init_params params = { /* no_alloc */ false, /* ctx */ &ctx_data };
    gguf_context * ctx = gguf_init_from_file(path, params);
    if (!ctx) {
        logi("[ERR] gguf_init_from_file failed: %s", path);
        return out;
    }

    int64_t tidx = gguf_find_tensor(ctx, name);
    if (tidx < 0) {
        gguf_free(ctx);
        if (ctx_data) ggml_free(ctx_data);
        return out;
    }

    ggml_tensor * cur = ggml_get_tensor(ctx_data, name);
    if (!cur) {
        logi("[ERR] ggml_get_tensor failed: %s", name);
        gguf_free(ctx);
        if (ctx_data) ggml_free(ctx_data);
        return out;
    }

    const int64_t n = ggml_nelements(cur);
    out.resize((size_t)n);

    if (cur->type == GGML_TYPE_F32) {
        memcpy(out.data(), cur->data, sizeof(float) * (size_t)n);
    } else {
        // Convert from quantized type to float using ggml type traits
        const struct ggml_type_traits * tr = ggml_get_type_traits(cur->type);
        if (!tr || !tr->to_float) {
            logi("[ERR] no to_float for type=%d", (int)cur->type);
            out.clear();
        } else {
            tr->to_float(cur->data, out.data(), n);
            logi("[INFO] Dequantized tensor %s from type=%d to F32 (%lld elems)", name, (int)cur->type, (long long)n);
        }
    }

    gguf_free(ctx);
    if (ctx_data) ggml_free(ctx_data);

    return out;
}

// Build a minimal ggml graph to compute Q * K^T and softmax(QK^T / sqrt(d)) * V for one head
// For simplicity we assume contiguous f32, shapes: Q [D, T], K [D, T], V [D, T]
// Returns a newly computed ggml_tensor with shape [D, T]
static ggml_tensor * ggml_attention_simple(struct ggml_context * ctx, ggml_tensor * Q, ggml_tensor * K, ggml_tensor * V) {
    const int64_t D = Q->ne[0];
    const int64_t T = Q->ne[1];
    GGML_ASSERT(K->ne[0] == D && K->ne[1] == T);
    GGML_ASSERT(V->ne[0] == D && V->ne[1] == T);

    // scores = Q^T * K -> with ggml semantics: mul_mat(Q[K=D,N=T], K[K=D,M=T]) => [N, M] = [T, T]
    ggml_tensor * scores = ggml_mul_mat(ctx, Q, K); // [T, T]

    // scale by 1/sqrt(D)
    const float scale = 1.0f / sqrtf((float)D);
    scores = ggml_scale(ctx, scores, scale);

    // softmax along last dim (x dimension)
    scores = ggml_soft_max(ctx, scores);

    // output = scores * V  (math: [T, T] @ [T, D] -> [T, D])
    // In ggml semantics, we set A = V^T(contiguous) [T, D] and B = scores [T, T]
    // mul_mat(A[K=T,N=D], B[K=T,M=T]) => [N, M] = [D, T]
    ggml_tensor * Vt_cont = ggml_cont(ctx, ggml_transpose(ctx, V)); // [T, D]
    ggml_tensor * out = ggml_mul_mat(ctx, Vt_cont, scores); // [D, T]
    return out;
}

int main(int argc, char ** argv) {
    // prompt 快速路径：如果 argv[2] 存在且不是数字，则把它当作输入文本并直接用 TextGenerator 生成
    if (argc >= 3) {
        // 判断 argv[2] 是否为纯数字（seq_len），否则当作 prompt
        bool is_number = true;
        for (const char* p = argv[2]; *p; ++p) { if (*p < '0' || *p > '9') { is_number = false; break; } }
        if (!is_number) {
            const char* gguf_path = argv[1];
            std::string prompt = argv[2];

            // 让推理引擎自行选择后端（不要强制 llama.cpp），避免在不支持的架构上触发断言
            // （如 Qwen/Qwen2 系列通常走内部 forward 模式）

            try {
                using namespace duorou::extensions::ollama;
                GlobalModelManager::initialize(true);
                auto &manager = GlobalModelManager::getInstance();

                const std::string model_id = "cli_gguf"; // 简单固定ID
                if (!manager.registerModel(model_id, gguf_path)) {
                    std::cerr << "[ERR] registerModel failed: " << gguf_path << std::endl;
                    return 2;
                }
                if (!manager.loadModel(model_id)) {
                    std::cerr << "[ERR] loadModel failed: " << model_id << std::endl;
                    return 3;
                }

                InferenceRequest req;
                req.model_id = model_id;
                req.prompt = prompt;
                req.max_tokens = 64;
                req.temperature = 0.7f;
                req.top_p = 0.9f;

                auto resp = manager.generateText(req);
                if (!resp.success) {
                    std::cerr << "[ERR] generateText failed: " << resp.error_message << std::endl;
                    return 4;
                }

                std::cout << resp.generated_text << std::endl;
                return 0;
            } catch (const std::exception &e) {
                std::cerr << "[ERR] Exception in prompt inference path: " << e.what() << std::endl;
                return 5;
            }
        }
    }

    if (argc < 2) {
        fprintf(stderr, "Usage: %s </path/to/model.gguf> [seq_len or prompt]\n", argv[0]);
        return 1;
    }
    const char * gguf_path = argv[1];
    const int T = argc >= 3 ? std::max(1, atoi(argv[2])) : 4; // small seq length

    logi("[INFO] ggml_qwen2vl_test starting, model: %s, T=%d", gguf_path, T);

    // For a smoke test, we do not parse full Qwen. We will read the first layer's q_proj, k_proj, v_proj weights
    // and build toy Q,K,V from random input hidden state H [T, D]
    // NOTE: The exact tensor names vary by exporter. For Qwen2.5 models in llama.cpp format, names are like:
    // "blk.0.attn_q.weight" etc. We'll try a few common patterns and stop at the first that exists.

    const char * q_names[] = {
        "blk.0.attn_q.weight",
        "layers.0.attention.wq.weight",
        "model.layers.0.self_attn.q_proj.weight",
    };
    const char * k_names[] = {
        "blk.0.attn_k.weight",
        "layers.0.attention.wk.weight",
        "model.layers.0.self_attn.k_proj.weight",
    };
    const char * v_names[] = {
        "blk.0.attn_v.weight",
        "layers.0.attention.wv.weight",
        "model.layers.0.self_attn.v_proj.weight",
    };

    std::vector<float> Wq, Wk, Wv;
    const char * picked_q = nullptr, * picked_k = nullptr, * picked_v = nullptr;

    for (const char * nm : q_names) { Wq = load_tensor_f32_from_gguf(gguf_path, nm); if (!Wq.empty()) { picked_q = nm; break; } }
    for (const char * nm : k_names) { Wk = load_tensor_f32_from_gguf(gguf_path, nm); if (!Wk.empty()) { picked_k = nm; break; } }
    for (const char * nm : v_names) { Wv = load_tensor_f32_from_gguf(gguf_path, nm); if (!Wv.empty()) { picked_v = nm; break; } }

    if (!picked_q || !picked_k || !picked_v) {
        logi("[WARN] q/k/v projection weights not F32 or missing in %s, using random fallback for smoke test", gguf_path);
        if (!picked_q) picked_q = "(rand) q_proj";
        if (!picked_k) picked_k = "(rand) k_proj";
        if (!picked_v) picked_v = "(rand) v_proj";
        // Wq/Wk/Wv may be empty; make_weight() will generate small pseudo-random values for empty vectors
    }

    logi("[INFO] Using tensors: Q=%s, K=%s, V=%s", picked_q, picked_k, picked_v);

    // Infer hidden dim D and output D from Wq shape by dividing size by rows or cols is not available directly
    // We assume weights are [D_out, D_in] contiguous in gguf export for linear layers.
    const int64_t D_in = (int64_t)std::sqrt((double)Wq.size());
    if (D_in*D_in != (int64_t)Wq.size()) {
        logi("[WARN] Unable to infer square shape for Wq, falling back to D=128");
        // use small square just for compute path validation
        // and trim or pad buffers accordingly
    }
    // Clamp D to avoid huge allocations when model hidden size is large
    const int64_t D = std::min<int64_t>(D_in > 0 ? D_in : 128, 512);

    // Build ggml context
    const size_t mem_size = 128ull*1024*1024; // 128MB arena for safety
    std::vector<uint8_t> mem(mem_size);
    struct ggml_init_params iparams = { mem_size, mem.data(), /* no mem_pool */ false };
    ggml_context * ctx = ggml_init(iparams);
    if (!ctx) {
        logi("[ERR] ggml_init failed");
        return 3;
    }

    // Create random H [T, D]
    std::vector<float> H(T*D);
    for (int i = 0; i < T*D; ++i) { H[i] = (float) (sin(0.1*i) * 0.01 + 0.001*i); }

    ggml_tensor * H_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, T);
    memcpy(H_t->data, H.data(), sizeof(float)*H.size());

    // Create weight tensors as [D, D] for simplicity
    auto make_weight = [&](const std::vector<float> & W) {
        ggml_tensor * Wt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, D);
        size_t need = (size_t)D*D;
        if (!W.empty()) {
            std::vector<float> tmp(need, 0.0f);
            const size_t copy_n = std::min(need, W.size());
            memcpy(tmp.data(), W.data(), sizeof(float)*copy_n);
            memcpy(Wt->data, tmp.data(), sizeof(float)*need);
        } else {
            // fill with small pseudo-random values to avoid NaNs in softmax
            float * wp = (float *) Wt->data;
            for (size_t i = 0; i < need; ++i) {
                wp[i] = 0.01f * sinf((float)i * 0.0137f);
            }
        }
        return Wt;
    };

    ggml_tensor * Wq_t = make_weight(Wq);
    ggml_tensor * Wk_t = make_weight(Wk);
    ggml_tensor * Wv_t = make_weight(Wv);

    // Q = Wq * H, etc.  ggml expects mul_mat(A,B)=A*B with A [K, N], B [K, M] -> [N, M]
    ggml_tensor * Q = ggml_mul_mat(ctx, Wq_t, H_t); // [D, T]
    ggml_tensor * K = ggml_mul_mat(ctx, Wk_t, H_t); // [D, T]
    ggml_tensor * V = ggml_mul_mat(ctx, Wv_t, H_t); // [D, T]

    ggml_tensor * out = ggml_attention_simple(ctx, Q, K, V); // [D, T]

    // Build and compute graph
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_graph_compute_with_ctx(ctx, gf, /*n_threads*/ std::max(1u, std::thread::hardware_concurrency()));

    // Fetch output
    float * out_data = (float *) out->data;
    logi("[OK] Computed attention output. Dump first row (up to 8 vals):");
    const int dump_n = std::min<int64_t>(8, D);
    for (int i = 0; i < dump_n; ++i) {
        printf(" %8.5f", out_data[i]);
    }
    printf("\n");

    ggml_free(ctx);
    return 0;
}