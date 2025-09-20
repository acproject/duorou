#pragma once

#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <utility>
#include "text_processor.h"

namespace duorou {
namespace model {

// Forward declarations
class Vocabulary;

// Input types for multimodal processing
struct MultimodalInput {
    std::vector<uint8_t> data;  // Raw data (image, audio, etc.)
    std::string type;           // "image", "audio", "video", etc.
    std::string format;         // "jpeg", "png", "wav", etc.
};

struct TextInput {
    std::string text;
    bool addSpecial = true;
};

// Base model interface that all models should inherit from
class BaseModel {
public:
    virtual ~BaseModel() = default;
    
    // Core text processing methods
    virtual std::vector<int32_t> encode(const std::string& text, bool addSpecial = true) = 0;
    virtual std::string decode(const std::vector<int32_t>& ids) = 0;
    
    // Model information
    virtual std::string getModelType() const = 0;
    virtual size_t getVocabSize() const = 0;
    virtual const Vocabulary* getVocabulary() const = 0;
    
    // Configuration and initialization
    virtual bool initialize(const std::string& configPath) = 0;
    virtual bool isInitialized() const = 0;
    
protected:
    bool initialized_ = false;
    std::string modelType_;
    std::unique_ptr<Vocabulary> vocabulary_;
    std::unique_ptr<TextProcessor> tokenizer_;
};

// Multimodal processor interface for models that support multiple input types
class MultimodalProcessor {
public:
    virtual ~MultimodalProcessor() = default;
    
    // Process multimodal inputs (text + image/audio/video)
    virtual std::vector<int32_t> processMultimodal(
        const std::vector<TextInput>& textInputs,
        const std::vector<MultimodalInput>& multimodalInputs
    ) = 0;
    
    // Check if model supports specific input type
    virtual bool supportsInputType(const std::string& type) const = 0;
    
    // Get supported input types
    virtual std::vector<std::string> getSupportedInputTypes() const = 0;
};

// Text-only model interface
class TextModel : public BaseModel {
public:
    virtual ~TextModel() = default;
    
    // Text generation methods
    virtual std::vector<int32_t> generate(
        const std::vector<int32_t>& inputIds,
        size_t maxLength = 512,
        float temperature = 1.0f,
        float topP = 0.9f
    ) = 0;
    
    // Forward pass for inference
    virtual std::vector<float> forward(const std::vector<int32_t>& inputIds) = 0;
};

// Vision model interface
class VisionModel {
public:
    virtual ~VisionModel() = default;
    
    // Process image data
    virtual std::vector<float> processImage(const std::vector<uint8_t>& imageData) = 0;
    
    // Get image feature dimensions
    virtual std::pair<size_t, size_t> getImageFeatureDims() const = 0;
    
    // Check if model is loaded
    virtual bool isLoaded() const = 0;
};

// Image processor interface
class ImageProcessor {
public:
    virtual ~ImageProcessor() = default;
    
    // Process image data and return processed pixel values
    virtual std::vector<float> processImage(const std::vector<uint8_t>& imageData) = 0;
    
    // Get image dimensions from raw data
    virtual std::pair<size_t, size_t> getImageDimensions(const std::vector<uint8_t>& imageData) const = 0;
    
    // Check if image format is supported
    virtual bool isSupported(const std::string& format) const = 0;
};

// Factory function type for model creation
using ModelFactory = std::function<std::unique_ptr<BaseModel>(const std::string&)>;

// Model registry for different model types
class ModelRegistry {
public:
    static ModelRegistry& getInstance();
    
    // Register a model factory
    void registerModel(const std::string& modelType, ModelFactory factory);
    
    // Create a model instance
    std::unique_ptr<BaseModel> createModel(const std::string& modelType, const std::string& configPath);
    
    // Get available model types
    std::vector<std::string> getAvailableModelTypes() const;
    
private:
    std::unordered_map<std::string, ModelFactory> factories_;
};

} // namespace model
} // namespace duorou