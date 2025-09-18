#pragma once

#include "text_processor.h"
#include "vocabulary.h"
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <cstdint>

namespace duorou {
namespace model {

// Model configuration structure
struct ModelConfig {
    std::string architecture;
    std::string tokenizer_type;
    size_t vocab_size;
    size_t context_length;
    size_t embedding_dim;
    size_t num_layers;
    size_t num_heads;
    double temperature;
    double top_p;
    int32_t top_k;
    
    ModelConfig() 
        : vocab_size(0), context_length(2048), embedding_dim(512), 
          num_layers(6), num_heads(8), temperature(0.8), 
          top_p(0.9), top_k(40) {}
};

// Model interface
class Model {
public:
    virtual ~Model() = default;
    
    // Core model operations
    virtual bool load(const std::string& modelPath) = 0;
    virtual bool isLoaded() const = 0;
    virtual void unload() = 0;
    
    // Text processing
    virtual std::vector<int32_t> encode(const std::string& text, bool addSpecial = true) = 0;
    virtual std::string decode(const std::vector<int32_t>& tokens) = 0;
    
    // Generation
    virtual std::vector<int32_t> generate(const std::vector<int32_t>& prompt, 
                                         size_t maxTokens = 100) = 0;
    virtual std::string generateText(const std::string& prompt, 
                                   size_t maxTokens = 100) = 0;
    
    // Model information
    virtual const ModelConfig& getConfig() const = 0;
    virtual const TextProcessor* getTokenizer() const = 0;
    virtual size_t getVocabSize() const = 0;
    virtual size_t getContextLength() const = 0;
    
    // Model metadata
    virtual std::string getModelName() const = 0;
    virtual std::string getModelVersion() const = 0;
    virtual std::map<std::string, std::string> getMetadata() const = 0;
};

// Base model implementation
class BaseModel : public Model {
public:
    BaseModel();
    virtual ~BaseModel();
    
    // Model operations
    bool load(const std::string& modelPath) override;
    bool isLoaded() const override;
    void unload() override;
    
    // Text processing
    std::vector<int32_t> encode(const std::string& text, bool addSpecial = true) override;
    std::string decode(const std::vector<int32_t>& tokens) override;
    
    // Generation (basic implementation)
    std::vector<int32_t> generate(const std::vector<int32_t>& prompt, 
                                 size_t maxTokens = 100) override;
    std::string generateText(const std::string& prompt, 
                           size_t maxTokens = 100) override;
    
    // Model information
    const ModelConfig& getConfig() const override;
    const TextProcessor* getTokenizer() const override;
    size_t getVocabSize() const override;
    size_t getContextLength() const override;
    
    // Model metadata
    std::string getModelName() const override;
    std::string getModelVersion() const override;
    std::map<std::string, std::string> getMetadata() const override;

protected:
    // Protected methods for derived classes
    virtual bool loadModel(const std::string& modelPath);
    virtual bool loadTokenizer(const std::string& tokenizerPath);
    virtual bool loadConfig(const std::string& configPath);
    
    // Generation helpers
    virtual int32_t sampleNext(const std::vector<int32_t>& context);
    virtual std::vector<double> computeLogits(const std::vector<int32_t>& context);
    virtual int32_t sampleFromLogits(const std::vector<double>& logits);
    
    // Configuration and state
    ModelConfig config_;
    std::unique_ptr<TextProcessor> tokenizer_;
    std::shared_ptr<Vocabulary> vocabulary_;
    bool loaded_;
    std::string modelPath_;
    std::string modelName_;
    std::string modelVersion_;
    std::map<std::string, std::string> metadata_;
};

// Model factory
class ModelFactory {
public:
    static std::unique_ptr<Model> createModel(const std::string& modelType);
    static std::unique_ptr<Model> loadModel(const std::string& modelPath);
    static std::vector<std::string> getSupportedModels();
    
    // Register custom model types
    using ModelCreator = std::function<std::unique_ptr<Model>()>;
    static void registerModel(const std::string& modelType, ModelCreator creator);

private:
    static std::map<std::string, ModelCreator> creators_;
};

} // namespace model
} // namespace duorou