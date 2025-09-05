#pragma once

#include "safetensors_parser_mmap.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <atomic>

namespace duorou {

// Configuration for the Qwen SafeTensors engine
struct QwenSafeTensorsConfig {
    std::string model_path;           // Path to model directory or single file
    int max_context_length = 2048;   // Maximum context length
    float temperature = 0.7f;        // Sampling temperature
    int top_k = 40;                   // Top-k sampling
    float top_p = 0.9f;              // Top-p (nucleus) sampling
    bool use_mmap = true;             // Use memory mapping (always true for this version)
    bool verbose = false;             // Verbose logging
    int num_threads = 4;              // Number of threads for inference
};

// Model architecture information
struct ModelArchitecture {
    uint32_t vocab_size = 0;
    uint32_t hidden_size = 0;
    uint32_t num_layers = 0;
    uint32_t num_attention_heads = 0;
    uint32_t num_key_value_heads = 0;
    uint32_t intermediate_size = 0;
    float rms_norm_eps = 1e-6f;
    uint32_t max_position_embeddings = 0;
    std::string model_type;
};

// Engine state
enum class EngineState {
    UNINITIALIZED,
    LOADING,
    READY,
    GENERATING,
    ERROR
};

// Memory-mapped Qwen SafeTensors inference engine
class QwenSafeTensorsEngineMmap {
public:
    explicit QwenSafeTensorsEngineMmap(const QwenSafeTensorsConfig& config);
    ~QwenSafeTensorsEngineMmap();
    
    // Model management
    bool loadModel();
    bool unloadModel();
    bool isModelLoaded() const;
    EngineState getState() const;
    
    // Model information
    const ModelArchitecture& getArchitecture() const { return architecture_; }
    std::string getModelInfo() const;
    
    // Tokenization
    std::vector<int> tokenize(const std::string& text) const;
    std::string detokenize(const std::vector<int>& tokens) const;
    
    // Text generation
    std::string generate(const std::string& prompt, int max_tokens = 100);
    std::vector<int> generateTokens(const std::vector<int>& input_tokens, int max_tokens = 100);
    
    // Configuration
    void setTemperature(float temperature) { config_.temperature = temperature; }
    void setTopK(int top_k) { config_.top_k = top_k; }
    void setTopP(float top_p) { config_.top_p = top_p; }
    
    // Quantization support
    bool supportsQuantization() const { return true; }
    std::vector<std::string> getSupportedQuantizations() const;
    
    // Memory usage
    size_t getMemoryUsage() const;
    size_t getModelSize() const;
    
    // Tensor access (for debugging/inspection)
    std::vector<std::string> getTensorNames() const;
    const TensorInfo* getTensorInfo(const std::string& name) const;
    const void* getTensorDataPtr(const std::string& name) const;
    
private:
    QwenSafeTensorsConfig config_;
    std::unique_ptr<SafeTensorsModelLoaderMmap> model_loader_;
    ModelArchitecture architecture_;
    std::atomic<EngineState> state_;
    mutable std::mutex mutex_;
    
    // Vocabulary and tokenization
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<int, std::string> reverse_vocab_;
    
    // Model weights (pointers to mapped memory)
    struct ModelWeights {
        const void* token_embeddings = nullptr;
        std::vector<const void*> layer_attention_weights;
        std::vector<const void*> layer_ffn_weights;
        std::vector<const void*> layer_norm_weights;
        const void* output_weights = nullptr;
    } weights_;
    
    // Private methods
    bool loadArchitecture();
    bool loadVocabulary();
    bool loadWeights();
    bool validateModel();
    
    // Inference helpers
    std::vector<float> forward(const std::vector<int>& tokens);
    int sampleToken(const std::vector<float>& logits);
    
    // Utility functions
    void log(const std::string& level, const std::string& message) const;
    void setState(EngineState state);
    
    // Tensor loading helpers
    template<typename T>
    const T* getTensorAs(const std::string& name) const {
        const void* ptr = getTensorDataPtr(name);
        return static_cast<const T*>(ptr);
    }
    
    // Memory management
    void cleanup();
};

} // namespace duorou