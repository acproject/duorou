#pragma once

// Ensure this header is only parsed by a C++ compiler
#ifdef __cplusplus

#include "base_model.h"
#include "qwen_text_model.h"
#include "qwen_vision_model.h"
#include "qwen_image_processor.h"
#include "../ml/tensor.h"
#include "../ml/context.h"
#include "../ml/nn/attention.h"
#include "../kvcache/cache.h"
#include "../kvcache/wrapper.h"
#include "../extensions/ollama/gguf_parser.h"
#include <vector>
#include <string>
#include <memory>
#include <cstdint>

namespace duorou {
namespace model {

// Multimodal model configuration
struct QwenMultimodalConfig {
    // Text model config
    TextModelOptions textOptions;
    
    // Vision model config
    VisionModelOptions visionOptions;
    
    // Image processor config
    ImageProcessorConfig imageProcessorConfig;
    
    // Model paths
    std::string textModelPath;
    std::string visionModelPath;
    std::string configPath;
    
    // Special tokens
    int32_t imageTokenId = 151655;      // <|image|>
    int32_t videoTokenId = 151656;      // <|video|>
    int32_t visionStartId = 151652;     // <|vision_start|>
    int32_t visionEndId = 151653;       // <|vision_end|>
    int32_t visionPadId = 151654;       // <|vision_pad|>
    
    // Processing parameters
    size_t maxImageTokens = 256;
    size_t maxSequenceLength = 2048;
    bool useVisionPadding = true;
};

// Pixel values with grid information using ml::Tensor
struct PixelValues {
    duorou::ml::Tensor data;  // Shape: [channels, height, width] or [batch, channels, height, width]
    size_t height = 0;
    size_t width = 0;
    size_t channels = 3;
    
    // Grid information for patches
    size_t gridHeight = 0;
    size_t gridWidth = 0;
    size_t gridTemporal = 1;
    
    bool isValid() const { 
        return data.numel() > 0 && height > 0 && width > 0; 
    }
    
    size_t totalPatches() const { 
        return gridHeight * gridWidth * gridTemporal; 
    }
    
    // Convert from raw data
    static PixelValues fromRawData(const std::vector<float>& rawData, 
                                   size_t h, size_t w, size_t c = 3);
};

// Multimodal input for processing
struct MultimodalInputData {
    std::vector<TextInput> textInputs;
    std::vector<MultimodalInput> imageInputs;
    
    bool hasText() const { return !textInputs.empty(); }
    bool hasImages() const { return !imageInputs.empty(); }
    size_t totalInputs() const { return textInputs.size() + imageInputs.size(); }
};

// Main Qwen Multimodal Model
class QwenMultimodalModel : public BaseModel, public MultimodalProcessor {
public:
    QwenMultimodalModel();
    QwenMultimodalModel(const QwenMultimodalConfig& config);
    QwenMultimodalModel(const QwenMultimodalConfig& config, 
                        std::shared_ptr<Vocabulary> external_vocab);
    ~QwenMultimodalModel() override = default;
    
    // BaseModel interface implementation
    std::vector<int32_t> encode(const std::string& text, bool addSpecial = true) override;
    std::string decode(const std::vector<int32_t>& ids) override;
    std::string getModelType() const override { return "qwen-multimodal"; }
    size_t getVocabSize() const override;
    const Vocabulary* getVocabulary() const override;
    bool initialize(const std::string& configPath) override;
    bool isInitialized() const override { return initialized_; }
    
    // MultimodalProcessor interface implementation
    std::vector<int32_t> processMultimodal(
        const std::vector<TextInput>& textInputs,
        const std::vector<MultimodalInput>& multimodalInputs
    ) override;
    bool supportsInputType(const std::string& type) const override;
    std::vector<std::string> getSupportedInputTypes() const override;
    
    // Qwen-specific methods
    bool loadModel(const std::string& modelPath);
    void setConfig(const QwenMultimodalConfig& config);
    const QwenMultimodalConfig& getConfig() const { return config_; }
    
    // Process pixel values from images
    PixelValues processPixelValues(const std::vector<uint8_t>& imageData);
    std::vector<PixelValues> processMultipleImages(const std::vector<std::vector<uint8_t>>& imagesData);
    
    // Encode multimodal input with proper token arrangement
    std::vector<int32_t> encodeMultimodal(const MultimodalInputData& input);
    
    // Post-tokenize processing (add special tokens, padding, etc.)
    std::vector<int32_t> postTokenize(const std::vector<int32_t>& tokenIds);
    
    // Forward pass with multimodal inputs using ml::Tensor
    duorou::ml::Tensor forward(
        duorou::ml::Context& ctx,
        const duorou::ml::Tensor& inputIds,
        const std::vector<PixelValues>& pixelValues = {},
        duorou::kvcache::Cache* cache = nullptr
    );
    
    // Generate text with multimodal context
    std::vector<int32_t> generateMultimodal(
        const std::vector<int32_t>& inputIds,
        const std::vector<PixelValues>& pixelValues = {},
        size_t maxLength = 512,
        float temperature = 1.0f,
        float topP = 0.9f
    );
    
    // Get text and vision models (for advanced usage)
    QwenTextModel* getTextModel() { return textModel_.get(); }
    QwenVisionModel* getVisionModel() { return visionModel_.get(); }
    QwenImageProcessor* getImageProcessor() { return imageProcessor_.get(); }
    
    // Model state management
    bool saveModel(const std::string& savePath);
    bool loadFromCheckpoint(const std::string& checkpointPath);
    
private:
    QwenMultimodalConfig config_;
    
    // New: External vocabulary (optional)
    std::shared_ptr<Vocabulary> external_vocabulary_;
    
    // Component models
    std::unique_ptr<QwenTextModel> textModel_;
    std::unique_ptr<QwenVisionModel> visionModel_;
    std::unique_ptr<QwenImageProcessor> imageProcessor_;
    
    // ML framework components
    std::unique_ptr<duorou::ml::Context> mlContext_;
    std::unique_ptr<duorou::ml::nn::MultiHeadAttention> attention_;
    std::unique_ptr<duorou::kvcache::CacheWrapper> kvCache_;
    
    // GGUF model loader
    std::unique_ptr<duorou::extensions::ollama::GGUFParser> ggufParser_;
    
    // Helper methods
    bool initializeComponents();
    bool loadComponentModels();
    
    // ML framework integration
    bool initializeMLComponents();
    duorou::ml::Tensor convertToTensor(const std::vector<int32_t>& data);
    duorou::ml::Tensor convertToTensor(const std::vector<float>& data, 
                                       const std::vector<int64_t>& shape);
    std::vector<float> convertFromTensor(const duorou::ml::Tensor& tensor);
    
    // GGUF model loading
    bool loadGGUFModel(const std::string& modelPath);
    duorou::ml::Tensor loadTensorFromGGUF(const std::string& tensorName);
    
    // Token processing helpers
    std::vector<int32_t> insertImageTokens(
        const std::vector<int32_t>& textTokens,
        const std::vector<PixelValues>& pixelValues
    );
    
    std::vector<int32_t> addVisionTokens(
        const std::vector<int32_t>& tokens,
        size_t numImageTokens
    );
    
    // Vision feature processing
    std::vector<float> processVisionFeatures(const std::vector<PixelValues>& pixelValues);
    
    // Attention mask creation for multimodal inputs
    std::vector<float> createMultimodalAttentionMask(
        const std::vector<int32_t>& inputIds,
        const std::vector<size_t>& imageBounds
    );
    
    // Configuration loading and validation
    bool loadConfig(const std::string& configPath);
    bool validateConfig() const;
    
    // Utility methods
    std::vector<size_t> findImageTokenPositions(const std::vector<int32_t>& tokens);
    size_t calculateImageTokenCount(const PixelValues& pixelValues);
    
    // Special token management
    bool isSpecialToken(int32_t tokenId) const;
    std::vector<int32_t> getSpecialTokens() const;
};

// Factory function for creating Qwen multimodal models
// Factory functions
std::unique_ptr<BaseModel> createQwenMultimodalModel(const std::string& configPath);

// New: Factory function that accepts external vocabulary
std::unique_ptr<BaseModel> createQwenMultimodalModel(const std::string& configPath, 
                                                     std::shared_ptr<Vocabulary> external_vocab);

// Utility functions for multimodal processing
namespace MultimodalUtils {
    // Convert between different input formats
    MultimodalInputData createMultimodalInput(
        const std::string& text,
        const std::vector<std::vector<uint8_t>>& images
    );
    
    // Validate multimodal input
    bool validateMultimodalInput(const MultimodalInputData& input);
    
    // Calculate total token count for multimodal input
    size_t estimateTokenCount(
        const MultimodalInputData& input,
        const QwenMultimodalModel& model
    );
    
    // Image format utilities
    std::string detectImageFormat(const std::vector<uint8_t>& imageData);
    bool isSupportedImageFormat(const std::string& format);
    
    // Token sequence utilities
    std::vector<int32_t> mergeTokenSequences(
        const std::vector<std::vector<int32_t>>& sequences
    );
    
    std::vector<std::vector<int32_t>> splitTokenSequence(
        const std::vector<int32_t>& tokens,
        int32_t separatorToken
    );
}

} // namespace model
} // namespace duorou

#endif // __cplusplus