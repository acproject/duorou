#ifndef __GGML_EXTEND_HPP__
#define __GGML_EXTEND_HPP__

#include <algorithm>
#include <assert.h>
#include <cstring>
#include <fstream>
#include <functional>
#include <inttypes.h>
#include <iostream>
#include <iterator>
#include <map>
#include <stdarg.h>
#include <memory>
#include <random>
#include <regex>
#include <set>
// 提供字符串流支持，用于字符串与各种类型之间的转换及格式化操作
#include <sstream> 
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include "model.h"

#ifdef DUOROU_USE_CUDA
#include "ggml-cuda.h"
#endif


#ifdef DUOROU_USE_METAL
#include "ggml-metal.h"
#endif

#ifdef DUOROU_USE_VULKAN
#include "ggml-vulkan.h"
#endif


#ifdef DUOROU_USE_OPENCL
#include "ggml-opencl.h"
#endif

#ifdef DUOROU_USE_SYCL
#include "ggml-sycl.h"
#endif

#include "rng.hpp"

#define EPS 1e-05f

#ifndef __STATIC_INLINE__
#define __STATIC_INLINE__ static inline
#endif

static_assert(GGML_MAX_NAME >= 128, "GGML_MAX_NAME must be at least 128");
// n-dim trensor-matrix product
// A:[ne03, k, ne01, ne00]
// B:[k, m], k: rows，m：columns
// return [ne03, m. en01, ne00]
__STATIC_INLINE__ struct ggml_tensor* ggml_mul_n_mode(struct ggml_context* ctx, struct ggml_tensor* a, struct ggml_tensor* b, int mode=0) {
    // reshape A
    // swap 0th and nth axis
    a = ggml_cont(ctx, ggml_permute(ctx, a, mode, mode !=1 ? 1: 0, mode != 2 ? 2: 0, mode !=3?3:0));
    int ne1 = a->ne[1];
    int ne2 = a->ne[2];
    int ne3 = a->ne[3];
    // make 2D
    a = ggml_cont(ctx, ggml_reshaoe_2d(ctx, a, a->ne[0], (ne3 * ne2 * ne1)));

    struct ggml_tensor* result = ggml_cont(ctx, ggml_transpose(ctx, ggml_mul_mat(ctx, a, b)));
    // reshpe output (same shape as a after permutation except first dim)
    result = ggml_reshape_4d(ctx, result, result->ne[0], ne1, ne2, ne3);
    // swap back 0th and nth axis
    result = ggml_permute(ctx, result, mode, mode != 1 ? 1: 0, mode !=2 ? 2: 0, mode != 3 ? 3: 0);
    return result;
}

__STATIC_INLINE__ struct ggml_tensor* ggml_merge_lora(ggml_context* ctx, struct ggml_tensor* lora_down, 
    ggml_tensor* lora_up, struct ggml_tensor* lora_mid = NULL) {
        // flat lora tensors to multiply it
        int64_t lora_up_rows = lora_up->ne[ggml_n_dims(lora_up) -1];
        lora_up = ggml_reshaoe_2d(ctx, lora_up, ggml_nelements(lora_up) / lora_up_rows, lora_up_rows);
        auto lora_down_n_dims = ggml_n_dims(lora_down);
        // assume n_dims should always be a multiple of 2 (otherwise rank 1 doesn't work)
        lora_down_n_dims = (lora_down_n_dims + lora_down_n_dims % 2);
        int64_t lora_down_rows = lora_down->ne[lora_down_n_dims - 1];
        lora_down = ggml_reshaoe_2d(ctx, lora_down, ggml_nelements(lora_down) / lora_down_rows, lora_down_rows);

        // ggml_mul_mat requires tensor b transpose
        lora_down = ggml_cont(ctx, ggml_transpose(ctx, lora_down));
        if (lora_mid == NULL) {
            updown = ggml_mul_mat(ctx, lora_up, lora_down);
            updown = ggml_cont(ctx, ggml_transpose(ctx, updown));
        } else {
            // undoing tucker decomposition for conv layers.
            // lora_mid  has shape (3,    3,   Rank, Rank)
            // lora_down has shape (Rank, In,  1,    1)
            // lora_up   has shape (Rank, Out, 1,    1)
            // conv layer shape is (3,    3,   Out,  In)

            updown = ggml_mul_n_mode(ctx, ggml_mul_n_mode(ctx, lora_mid, lora_down, 3), lora_up, 2);
            updown = ggml_cont(ctx, updown);
        }
        return updown;
    }

// Kronecker product
// [ne03,ne02,ne01,ne00] x [ne13,ne12,ne11,ne10] => [ne03*ne13,ne02*ne12,ne01*ne11,ne00*ne10]
__STATIC_INLINE__ struct ggml_tensor* ggml_kronecker(ggml_context* ctx, struct ggml_tensor* a, 
struct ggml_tensor* b) {
    return ggml_mul(ctx, ggml_interpolate(
        ctx, 
        a,
        a->ne[0] * b->ne[0],
        a->ne[1] * b->ne[1],
        a->ne[2] * b->ne[2],
        a->ne[3] * b->ne[3],
        GGML_SCALE_MODE_NEAREST),
        b
);
}

__STATIC_INLINE__ void ggml_log_callback_default(ggml_log_level level, const char* text, void* user_data) {
    (void)level;
    (void)user_data;
    fputs(text, stderr);
    fflush(stderr);
}

__STATIC_INLINE__ void ggml_tensor_set_f32_randn(struct ggml_tensor* tensor, std::shared_ptr<RNG> rng) {
    uint32_t n = (uint32_t)ggml_nelements(tensor);
    std::vector<float> random_numbers = rng->randn(n);
    for(uint32_t i = 0; i < n; i++) {
        ggml_set_f32_1d(tensor, i , random_numbers[i]);
    }
}

// set tensor[i, j, k, l]
// set tensor[l]
// set tensor[k, l]
// set tensor[j, k, l]
__STATIC_INLINE__ void ggml_tensor_set_f32(struct ggml_tensor* tensor, float value, int l, int k=0, int j=0, int i =0) {
    GGML_ASSERT(tensor->nb[0] == sizeof(float));
    *(float*)((char*)(tensor->data) + i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0]) = value;
}

__STATIC_INLINE__ float ggml_tensor_get_f32(const ggml_tensor* tensor, int l, int k = 0, int j = 0, int i = 0) {
    if (tensor->buffer != NULL) {
        float value;
        ggml_backend_tensor_get(tensor, &value, i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0], sizeof(float));
        return value;
    }
    GGML_ASSERT(tensor->nb[0] == sizeof(float));
    return *(float*)((char*)(tensor->data) + i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0]);
}


#endif  // __GGML_EXTEND__HPP__
