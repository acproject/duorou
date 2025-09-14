#ifndef ALGORITHM_FACTORY_H
#define ALGORITHM_FACTORY_H

// Minimal C-style factory to avoid C++ compilation issues

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct GGMLAttention GGMLAttention;
typedef struct GGMLFeedForward GGMLFeedForward;
typedef struct ROPEProcessor ROPEProcessor;

// Simple factory functions
GGMLAttention* createAttentionAlgorithm(const char* type);
GGMLFeedForward* createFeedForwardAlgorithm(const char* type);
ROPEProcessor* createPositionalEncodingAlgorithm(const char* type);

// Simple config structure
typedef struct {
    const char* attention_type;
    const char* feedforward_type;
    const char* positional_encoding_type;
    int use_optimized_attention;
    int use_kv_cache;
    int enable_parallel_heads;
} AlgorithmConfig;

// Default config initializer
static inline AlgorithmConfig getDefaultAlgorithmConfig() {
    AlgorithmConfig config;
    config.attention_type = "ggml_attention";
    config.feedforward_type = "swiglu";
    config.positional_encoding_type = "rope";
    config.use_optimized_attention = 1;
    config.use_kv_cache = 1;
    config.enable_parallel_heads = 1;
    return config;
}

#ifdef __cplusplus
}
#endif

#endif // ALGORITHM_FACTORY_H