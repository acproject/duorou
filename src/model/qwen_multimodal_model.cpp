#include "qwen_multimodal_model.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>

namespace duorou {
namespace model {

// QwenMultimodalModel implementation
QwenMultimodalModel::QwenMultimodalModel() {
    config_ = QwenMultimodalConfig{};
    initializeMLComponents();
}

QwenMultimodalModel::QwenMultimodalModel(const QwenMultimodalConfig& config) 
    : config_(config) {
    initializeMLComponents();
}

bool QwenMultimodalModel::initialize(const std::string& configPath) {
    if (!loadConfig(configPath)) {
        std::cerr << "Failed to load config from: " << configPath << std::endl;
        return false;
    }
    
    if (!validateConfig()) {
        std::cerr << "Invalid configuration" << std::endl;
        return false;
    }
    
    if (!initializeComponents()) {
        std::cerr << "Failed to initialize components" << std::endl;
        return false;
    }
    
    initialized_ = true;
    modelType_ = "qwen-multimodal";
    
    return true;
}

bool QwenMultimodalModel::loadModel(const std::string& modelPath) {
    if (!initialized_) {
        std::cerr << "Model not initialized. Call initialize() first." << std::endl;
        return false;
    }
    
    return loadComponentModels();
}

std::vector<int32_t> QwenMultimodalModel::encode(const std::string& text, bool addSpecial) {
    if (!textModel_) {
        std::cerr << "Text model not initialized" << std::endl;
        return {};
    }
    
    return textModel_->encode(text, addSpecial);
}

std::string QwenMultimodalModel::decode(const std::vector<int32_t>& ids) {
    if (!textModel_) {
        std::cerr << "Text model not initialized" << std::endl;
        return "";
    }
    
    return textModel_->decode(ids);
}

size_t QwenMultimodalModel::getVocabSize() const {
    if (!textModel_) {
        return 0;
    }
    
    return textModel_->getVocabSize();
}

const Vocabulary* QwenMultimodalModel::getVocabulary() const {
    if (!textModel_) {
        return nullptr;
    }
    
    return textModel_->getVocabulary();
}

std::vector<int32_t> QwenMultimodalModel::processMultimodal(
    const std::vector<TextInput>& textInputs,
    const std::vector<MultimodalInput>& multimodalInputs) {
    
    MultimodalInputData input;
    input.textInputs = textInputs;
    input.imageInputs = multimodalInputs;
    
    return encodeMultimodal(input);
}

bool QwenMultimodalModel::supportsInputType(const std::string& type) const {
    return type == "text" || type == "image";
}

std::vector<std::string> QwenMultimodalModel::getSupportedInputTypes() const {
    return {"text", "image"};
}

PixelValues QwenMultimodalModel::processPixelValues(const std::vector<uint8_t>& imageData) {
    if (!imageProcessor_) {
        std::cerr << "Image processor not initialized" << std::endl;
        return {};
    }
    
    // Get image dimensions
    auto [width, height] = imageProcessor_->getImageDimensions(imageData);
    int channels = 3; // Assume RGB
    
    // Process image
    auto processedData = imageProcessor_->processImage(imageData);
    
    if (processedData.empty()) {
        std::cerr << "Failed to process image data" << std::endl;
        return {};
    }
    
    // Create PixelValues using fromRawData method
    PixelValues result = PixelValues::fromRawData(processedData, height, width, channels);
    
    // Calculate grid information
    const auto& config = imageProcessor_->getConfig();
    result.gridHeight = height / config.patchSize;
    result.gridWidth = width / config.patchSize;
    result.gridTemporal = 1; // For images
    
    return result;
}

std::vector<PixelValues> QwenMultimodalModel::processMultipleImages(
    const std::vector<std::vector<uint8_t>>& imagesData) {
    
    std::vector<PixelValues> results;
    results.reserve(imagesData.size());
    
    for (const auto& imageData : imagesData) {
        auto pixelValues = processPixelValues(imageData);
        if (pixelValues.isValid()) {
            results.push_back(pixelValues);
        }
    }
    
    return results;
}

std::vector<int32_t> QwenMultimodalModel::encodeMultimodal(const MultimodalInputData& input) {
    std::vector<int32_t> result;
    
    // Process images first to get pixel values
    std::vector<PixelValues> pixelValues;
    for (const auto& imageInput : input.imageInputs) {
        if (imageInput.type == "image") {
            auto pv = processPixelValues(imageInput.data);
            if (pv.isValid()) {
                pixelValues.push_back(pv);
            }
        }
    }
    
    // Process text inputs
    for (const auto& textInput : input.textInputs) {
        auto textTokens = encode(textInput.text, textInput.addSpecial);
        
        // Insert image tokens if there are images
        if (!pixelValues.empty()) {
            textTokens = insertImageTokens(textTokens, pixelValues);
        }
        
        result.insert(result.end(), textTokens.begin(), textTokens.end());
    }
    
    // Post-process tokens
    result = postTokenize(result);
    
    return result;
}

std::vector<int32_t> QwenMultimodalModel::postTokenize(const std::vector<int32_t>& tokenIds) {
    std::vector<int32_t> result = tokenIds;
    
    // Add special tokens, padding, etc. as needed
    // For now, return as is
    
    return result;
}

duorou::ml::Tensor QwenMultimodalModel::forward(
    duorou::ml::Context& ctx,
    const duorou::ml::Tensor& inputIds,
    const std::vector<PixelValues>& pixelValues,
    duorou::kvcache::Cache* cache) {
    
    if (!textModel_) {
        std::cerr << "Text model not initialized" << std::endl;
        return {};
    }
    
    // Process vision features if present
    std::vector<float> visionFeatures;
    if (!pixelValues.empty() && visionModel_) {
        visionFeatures = processVisionFeatures(pixelValues);
    }
    
    // Convert input tensor to vector for text model
    auto inputVector = convertFromTensor(inputIds);
    std::vector<int32_t> inputIds_vec(inputVector.begin(), inputVector.end());
    
    // Forward pass through text model
    auto textFeatures = textModel_->forward(inputIds_vec);
    
    // Convert text features to tensor
    std::vector<int64_t> shape = {static_cast<int64_t>(textFeatures.size())};
    auto resultTensor = convertToTensor(textFeatures, shape);
    
    // Combine text and vision features
    // For now, just return text features
    return resultTensor;
}

std::vector<int32_t> QwenMultimodalModel::generateMultimodal(
    const std::vector<int32_t>& inputIds,
    const std::vector<PixelValues>& pixelValues,
    size_t maxLength,
    float temperature,
    float topP) {
    
    if (!textModel_) {
        std::cerr << "Text model not initialized" << std::endl;
        return {};
    }
    
    // For now, delegate to text model
    return textModel_->generate(inputIds, maxLength, temperature, topP);
}

bool QwenMultimodalModel::saveModel(const std::string& savePath) {
    // TODO: Implement model saving
    return true;
}

bool QwenMultimodalModel::loadFromCheckpoint(const std::string& checkpointPath) {
    // TODO: Implement checkpoint loading
    return true;
}

bool QwenMultimodalModel::initializeComponents() {
    // Initialize text model
    textModel_ = std::make_unique<QwenTextModel>(config_.textOptions);
    if (!textModel_->initialize(config_.configPath)) {
        std::cerr << "Failed to initialize text model" << std::endl;
        return false;
    }
    
    // Initialize vision model
    visionModel_ = std::make_unique<QwenVisionModel>(config_.visionOptions);
    if (!visionModel_->initialize(config_.configPath)) {
        std::cerr << "Failed to initialize vision model" << std::endl;
        return false;
    }
    
    // Initialize image processor
    imageProcessor_ = std::make_unique<QwenImageProcessor>(config_.imageProcessorConfig);
    
    // Set up vocabulary and tokenizer from text model
    vocabulary_ = std::unique_ptr<Vocabulary>(const_cast<Vocabulary*>(textModel_->getVocabulary()));
    
    return true;
}

bool QwenMultimodalModel::loadComponentModels() {
    bool success = true;
    
    // Load text model
    if (!config_.textModelPath.empty()) {
        success &= textModel_->loadModel(config_.textModelPath);
    }
    
    // Load vision model
    if (!config_.visionModelPath.empty()) {
        success &= visionModel_->loadModel(config_.visionModelPath);
    }
    
    return success;
}

std::vector<int32_t> QwenMultimodalModel::insertImageTokens(
    const std::vector<int32_t>& textTokens,
    const std::vector<PixelValues>& pixelValues) {
    
    std::vector<int32_t> result;
    
    // Simple implementation: add image tokens at the beginning
    for (const auto& pv : pixelValues) {
        result.push_back(config_.visionStartId);
        
        // Add image token
        result.push_back(config_.imageTokenId);
        
        // Add vision padding tokens based on patch count
        size_t numImageTokens = calculateImageTokenCount(pv);
        for (size_t i = 0; i < numImageTokens; ++i) {
            result.push_back(config_.visionPadId);
        }
        
        result.push_back(config_.visionEndId);
    }
    
    // Add text tokens
    result.insert(result.end(), textTokens.begin(), textTokens.end());
    
    return result;
}

std::vector<int32_t> QwenMultimodalModel::addVisionTokens(
    const std::vector<int32_t>& tokens,
    size_t numImageTokens) {
    
    std::vector<int32_t> result = tokens;
    
    // Add vision tokens as needed
    // Implementation depends on specific model requirements
    
    return result;
}

std::vector<float> QwenMultimodalModel::processVisionFeatures(const std::vector<PixelValues>& pixelValues) {
    if (!visionModel_) {
        return {};
    }
    
    std::vector<float> allFeatures;
    
    for (const auto& pv : pixelValues) {
        // Convert tensor data to vector
        std::vector<float> tensorData = convertFromTensor(pv.data);
        
        // Convert float pixel values to uint8 for vision model
        std::vector<uint8_t> imageData(tensorData.size());
        for (size_t i = 0; i < tensorData.size(); ++i) {
            imageData[i] = static_cast<uint8_t>(std::clamp(tensorData[i] * 255.0f, 0.0f, 255.0f));
        }
        
        // Process through vision model
        auto features = visionModel_->processImage(imageData);
        allFeatures.insert(allFeatures.end(), features.begin(), features.end());
    }
    
    return allFeatures;
}

std::vector<float> QwenMultimodalModel::createMultimodalAttentionMask(
    const std::vector<int32_t>& inputIds,
    const std::vector<size_t>& imageBounds) {
    
    size_t seqLen = inputIds.size();
    std::vector<float> mask(seqLen * seqLen, 1.0f);
    
    // TODO: Implement proper attention mask for multimodal inputs
    
    return mask;
}

bool QwenMultimodalModel::loadConfig(const std::string& configPath) {
    // TODO: Load configuration from JSON file
    // For now, use default values
    return true;
}

bool QwenMultimodalModel::validateConfig() const {
    // Basic validation
    return config_.textOptions.hiddenSize > 0 && 
           config_.visionOptions.hiddenSize > 0 &&
           config_.imageProcessorConfig.imageSize > 0;
}

std::vector<size_t> QwenMultimodalModel::findImageTokenPositions(const std::vector<int32_t>& tokens) {
    std::vector<size_t> positions;
    
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] == config_.imageTokenId) {
            positions.push_back(i);
        }
    }
    
    return positions;
}

size_t QwenMultimodalModel::calculateImageTokenCount(const PixelValues& pixelValues) {
    // Calculate based on patch count
    size_t patchCount = pixelValues.totalPatches();
    
    // Use configured max or calculated count
    return std::min(patchCount, config_.maxImageTokens);
}

bool QwenMultimodalModel::isSpecialToken(int32_t tokenId) const {
    return tokenId == config_.imageTokenId ||
           tokenId == config_.videoTokenId ||
           tokenId == config_.visionStartId ||
           tokenId == config_.visionEndId ||
           tokenId == config_.visionPadId;
}

std::vector<int32_t> QwenMultimodalModel::getSpecialTokens() const {
    return {
        config_.imageTokenId,
        config_.videoTokenId,
        config_.visionStartId,
        config_.visionEndId,
        config_.visionPadId
    };
}

void QwenMultimodalModel::setConfig(const QwenMultimodalConfig& config) {
    config_ = config;
}

// Factory function
std::unique_ptr<BaseModel> createQwenMultimodalModel(const std::string& configPath) {
    auto model = std::make_unique<QwenMultimodalModel>();
    if (!model->initialize(configPath)) {
        return nullptr;
    }
    return std::move(model);
}

// MultimodalUtils namespace implementation
namespace MultimodalUtils {

MultimodalInputData createMultimodalInput(
    const std::string& text,
    const std::vector<std::vector<uint8_t>>& images) {
    
    MultimodalInputData input;
    
    // Add text input
    if (!text.empty()) {
        TextInput textInput;
        textInput.text = text;
        textInput.addSpecial = true;
        input.textInputs.push_back(textInput);
    }
    
    // Add image inputs
    for (const auto& imageData : images) {
        MultimodalInput imageInput;
        imageInput.data = imageData;
        imageInput.type = "image";
        imageInput.format = "unknown"; // Would detect in real implementation
        input.imageInputs.push_back(imageInput);
    }
    
    return input;
}

bool validateMultimodalInput(const MultimodalInputData& input) {
    // Basic validation
    if (!input.hasText() && !input.hasImages()) {
        return false;
    }
    
    // Validate text inputs
    for (const auto& textInput : input.textInputs) {
        if (textInput.text.empty()) {
            return false;
        }
    }
    
    // Validate image inputs
    for (const auto& imageInput : input.imageInputs) {
        if (imageInput.data.empty() || imageInput.type != "image") {
            return false;
        }
    }
    
    return true;
}

size_t estimateTokenCount(
    const MultimodalInputData& input,
    const QwenMultimodalModel& model) {
    
    size_t totalTokens = 0;
    
    // Estimate text tokens
    for (const auto& textInput : input.textInputs) {
        // Rough estimate: 1 token per 4 characters
        totalTokens += textInput.text.length() / 4;
    }
    
    // Estimate image tokens
    totalTokens += input.imageInputs.size() * model.getConfig().maxImageTokens;
    
    return totalTokens;
}

std::string detectImageFormat(const std::vector<uint8_t>& imageData) {
    if (imageData.size() < 8) return "unknown";
    
    // PNG signature
    if (imageData[0] == 0x89 && imageData[1] == 0x50 && imageData[2] == 0x4E && imageData[3] == 0x47) {
        return "png";
    }
    
    // JPEG signature
    if (imageData[0] == 0xFF && imageData[1] == 0xD8) {
        return "jpeg";
    }
    
    // BMP signature
    if (imageData[0] == 0x42 && imageData[1] == 0x4D) {
        return "bmp";
    }
    
    return "unknown";
}

bool isSupportedImageFormat(const std::string& format) {
    return format == "png" || format == "jpeg" || format == "jpg" || format == "bmp";
}

std::vector<int32_t> mergeTokenSequences(const std::vector<std::vector<int32_t>>& sequences) {
    std::vector<int32_t> merged;
    
    for (const auto& sequence : sequences) {
        merged.insert(merged.end(), sequence.begin(), sequence.end());
    }
    
    return merged;
}

std::vector<std::vector<int32_t>> splitTokenSequence(
    const std::vector<int32_t>& tokens,
    int32_t separatorToken) {
    
    std::vector<std::vector<int32_t>> sequences;
    std::vector<int32_t> currentSequence;
    
    for (int32_t token : tokens) {
        if (token == separatorToken) {
            if (!currentSequence.empty()) {
                sequences.push_back(currentSequence);
                currentSequence.clear();
            }
        } else {
            currentSequence.push_back(token);
        }
    }
    
    if (!currentSequence.empty()) {
        sequences.push_back(currentSequence);
    }
    
    return sequences;
}

} // namespace MultimodalUtils

// Add PixelValues::fromRawData implementation
PixelValues PixelValues::fromRawData(const std::vector<float>& rawData, 
                                     size_t h, size_t w, size_t c) {
    PixelValues result;
    result.height = h;
    result.width = w;
    result.channels = c;
    
    // Create tensor with shape [channels, height, width]
    std::vector<int64_t> shape = {static_cast<int64_t>(c), 
                                  static_cast<int64_t>(h), 
                                  static_cast<int64_t>(w)};
    result.data = duorou::ml::Tensor(shape, duorou::ml::DataType::FLOAT32);
    
    // Copy data to tensor
    result.data.copyFromHost(rawData.data(), rawData.size() * sizeof(float));
    
    return result;
}

// Add ML framework integration methods
bool QwenMultimodalModel::initializeMLComponents() {
    try {
        mlContext_ = std::make_unique<duorou::ml::Context>();
        kvCache_ = std::make_unique<duorou::kvcache::CacheWrapper>(duorou::kvcache::CacheType::CAUSAL);
        
        // Initialize attention with default parameters
        attention_ = std::make_unique<duorou::ml::nn::MultiHeadAttention>(
            768,  // embed_dim
            12,   // num_heads
            12,   // kv_heads
            true, // bias
            0.1f  // dropout
        );
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize ML components: " << e.what() << std::endl;
        return false;
    }
}

duorou::ml::Tensor QwenMultimodalModel::convertToTensor(const std::vector<int32_t>& data) {
    std::vector<int64_t> shape = {static_cast<int64_t>(data.size())};
    duorou::ml::Tensor tensor(shape, duorou::ml::DataType::INT32);
    tensor.copyFromHost(data.data(), data.size() * sizeof(int32_t));
    return tensor;
}

duorou::ml::Tensor QwenMultimodalModel::convertToTensor(const std::vector<float>& data, 
                                                        const std::vector<int64_t>& shape) {
    duorou::ml::Tensor tensor(shape, duorou::ml::DataType::FLOAT32);
    tensor.copyFromHost(data.data(), data.size() * sizeof(float));
    return tensor;
}

std::vector<float> QwenMultimodalModel::convertFromTensor(const duorou::ml::Tensor& tensor) {
    std::vector<float> result(tensor.numel());
    tensor.copyToHost(result.data(), result.size() * sizeof(float));
    return result;
}

bool QwenMultimodalModel::loadGGUFModel(const std::string& modelPath) {
    try {
        ggufParser_ = std::make_unique<duorou::extensions::ollama::GGUFParser>(true);
        return ggufParser_->parseFile(modelPath);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load GGUF model: " << e.what() << std::endl;
        return false;
    }
}

duorou::ml::Tensor QwenMultimodalModel::loadTensorFromGGUF(const std::string& tensorName) {
    if (!ggufParser_) {
        std::cerr << "GGUF parser not initialized" << std::endl;
        return duorou::ml::Tensor();
    }
    
    // Get tensor info from GGUF parser
    const auto* tensorInfo = ggufParser_->getTensorInfo(tensorName);
    if (!tensorInfo) {
        std::cerr << "Tensor not found: " << tensorName << std::endl;
        return duorou::ml::Tensor();
    }
    
    // Create tensor with appropriate shape and type
    std::vector<int64_t> shape;
    for (auto dim : tensorInfo->dimensions) {
        shape.push_back(static_cast<int64_t>(dim));
    }
    
    // For now, return empty tensor with correct shape
    // TODO: Implement actual tensor data loading from GGUF
    return duorou::ml::Tensor::zeros(shape, duorou::ml::DataType::FLOAT32);
}

} // namespace model
} // namespace duorou