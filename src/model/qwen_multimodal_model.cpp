#include "qwen_multimodal_model.h"
#include "tokenizer_factory.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include "../../third_party/llama.cpp/vendor/nlohmann/json.hpp"

#include "../ml/backend/backend.h"

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

// New: Constructor that accepts external vocabulary
QwenMultimodalModel::QwenMultimodalModel(const QwenMultimodalConfig& config, 
                                         std::shared_ptr<Vocabulary> external_vocab) 
    : config_(config), external_vocabulary_(external_vocab) {
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
        std::cerr << "[WARN] Model not initialized; proceeding with minimal GGUF-based setup" << std::endl;
        // Ensure text model exists for minimal inference
        if (!textModel_) {
            textModel_ = std::make_unique<QwenTextModel>(config_.textOptions);
        }
    }

    // If a GGUF file is passed in, prioritize initializing vocabulary and tokenizer from GGUF
    auto ends_with = [](const std::string& s, const std::string& suffix) {
        return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
    };

    bool gguf_inited = false;
    // First try by extension
    if (!modelPath.empty() && (ends_with(modelPath, ".gguf") || modelPath.find(".gguf") != std::string::npos)) {
        gguf_inited = loadGGUFModel(modelPath);
        if (!gguf_inited) {
            std::cerr << "[WARN] Failed to initialize tokenizer/vocab from GGUF: " << modelPath << std::endl;
        }
    } else {
        // Extension-less path (e.g., Ollama blob). Probe GGUF magic via parser.
        duorou::extensions::ollama::GGUFParser probe(/*use_mmap=*/true);
        if (probe.parseFile(modelPath)) {
            std::cout << "[DEBUG] Probed GGUF successfully for path (no .gguf extension): " << modelPath << std::endl;
            gguf_inited = loadGGUFModel(modelPath);
            if (!gguf_inited) {
                std::cerr << "[WARN] Failed to initialize tokenizer/vocab from probed GGUF: " << modelPath << std::endl;
            }
        } else if (!config_.textModelPath.empty()) {
            // Try config_.textModelPath as a GGUF source
            duorou::extensions::ollama::GGUFParser probe2(/*use_mmap=*/true);
            if (probe2.parseFile(config_.textModelPath)) {
                std::cout << "[DEBUG] Probed GGUF successfully for config path: " << config_.textModelPath << std::endl;
                gguf_inited = loadGGUFModel(config_.textModelPath);
                if (!gguf_inited) {
                    std::cerr << "[WARN] Failed to initialize tokenizer/vocab from probed GGUF: " << config_.textModelPath << std::endl;
                }
            }
        }
    }

    // Ensure text model is initialized so generate() will work
    if (textModel_) {
        if (external_vocabulary_) {
            if (!textModel_->initialize(config_.configPath, true)) {
                std::cerr << "[WARN] Failed to initialize text model with external vocabulary" << std::endl;
            }
        } else {
            if (!textModel_->initialize(config_.configPath)) {
                std::cerr << "[WARN] Failed to initialize text model without external vocabulary" << std::endl;
            }
        }
    } else {
        std::cerr << "[ERROR] textModel_ is null after minimal setup; cannot proceed" << std::endl;
        return false;
    }

    // In minimal setup, skip loading vision model/components to avoid null dereference
    return true;
}

std::vector<int32_t> QwenMultimodalModel::encode(const std::string& text, bool addSpecial) {
    std::cout << "[DEBUG] QwenMultimodalModel::encode called with text: '" << text 
              << "' (length: " << text.length() << " bytes, addSpecial: " << addSpecial << ")" << std::endl;
    
    // If external vocabulary exists, prioritize using tokenizer created based on external vocabulary
    if (external_vocabulary_) {
        std::cout << "[DEBUG] Using external vocabulary for encoding (vocab size: " 
                  << external_vocabulary_->size() << ")" << std::endl;
        
        if (tokenizer_) {
            std::cout << "[DEBUG] Tokenizer is available, using proper tokenization" << std::endl;
            std::vector<int32_t> result = tokenizer_->encode(text, addSpecial);
            std::cout << "[DEBUG] Tokenizer encoded " << result.size() << " tokens: ";
            for (size_t i = 0; i < std::min(result.size(), size_t(10)); ++i) {
                std::cout << result[i] << " ";
            }
            if (result.size() > 10) std::cout << "...";
            std::cout << std::endl;
            return result;
        } else {
            std::cout << "[WARN] Tokenizer is NULL! Falling back to text model encoding" << std::endl;
            
            // Fallback to text model instead of byte-level encoding
            // Byte-level encoding is incorrect for UTF-8 text like Chinese characters
            if (!textModel_) {
                std::cerr << "[ERROR] Both tokenizer and text model are unavailable" << std::endl;
                return {};
            }
            
            std::vector<int32_t> result = textModel_->encode(text, addSpecial);
            std::cout << "[DEBUG] Text model fallback encoded " << result.size() << " tokens" << std::endl;
            return result;
        }
    }
    
    std::cout << "[DEBUG] No external vocabulary, falling back to text model" << std::endl;
    
    // Fallback to text model
    if (!textModel_) {
        std::cerr << "[ERROR] Text model not initialized" << std::endl;
        return {};
    }
    
    std::vector<int32_t> result = textModel_->encode(text, addSpecial);
    std::cout << "[DEBUG] Text model encoded " << result.size() << " tokens" << std::endl;
    return result;
}

std::string QwenMultimodalModel::decode(const std::vector<int32_t>& ids) {
    // If external vocabulary exists, prioritize using tokenizer created based on external vocabulary
    if (external_vocabulary_) {
        std::cout << "[DEBUG] Using external vocabulary for decoding" << std::endl;
        if (tokenizer_) {
            return tokenizer_->decode(ids);
        }
        // Fallback: use external vocabulary for decoding
        std::string result;
        for (int32_t id : ids) {
            std::string token_text = external_vocabulary_->decode(id);
            if (!token_text.empty()) {
                result += token_text;
            }
        }
        return result;
    }
    
    // Fallback to text model
    if (!textModel_) {
        std::cerr << "Text model not initialized" << std::endl;
        return "";
    }
    
    return textModel_->decode(ids);
}

size_t QwenMultimodalModel::getVocabSize() const {
    // Prioritize using external vocabulary
    if (external_vocabulary_) {
        return external_vocabulary_->size();
    }
    // Fallback to text model's vocabulary
    if (textModel_) {
        return textModel_->getVocabSize();
    }
    return 0;
}

const Vocabulary* QwenMultimodalModel::getVocabulary() const {
    // Prioritize using external vocabulary
    if (external_vocabulary_) {
        return external_vocabulary_.get();
    }
    // Fallback to text model's vocabulary
    if (textModel_) {
        return textModel_->getVocabulary();
    }
    return nullptr;
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
    
    // Convert input tensor (expected INT32 token ids) to std::vector<int32_t>
    std::vector<int32_t> inputIds_vec;
    if (inputIds.numel() > 0) {
        if (inputIds.dtype() != duorou::ml::DataType::INT32) {
            std::cerr << "[WARN] QwenMultimodalModel::forward expected INT32 inputIds but got different dtype; attempting to interpret as INT32" << std::endl;
        }
        inputIds_vec.resize(static_cast<size_t>(inputIds.numel()));
        inputIds.copyToHost(inputIds_vec.data(), inputIds_vec.size() * sizeof(int32_t));
    }
    
    // Forward pass through text model (KV cache-enabled) to produce logits tensor directly
    auto logitsTensor = textModel_->forward(ctx, inputIds, cache);
    
    // Combine text and vision features if needed. For now, return text logits directly.
    return logitsTensor;
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
    // If external vocabulary is not yet provided, but the configured text model path is a GGUF file,
    // then prioritize initializing external_vocabulary_ and tokenizer_ from GGUF
    if (!external_vocabulary_) {
        const std::string& path = config_.textModelPath;
        if (!path.empty() && path.find(".gguf") != std::string::npos) {
            bool ok = loadGGUFModel(path);
            if (ok) {
                std::cout << "[DEBUG] Initialized external vocabulary from GGUF in initializeComponents, size="
                          << (external_vocabulary_ ? external_vocabulary_->size() : 0) << std::endl;
            } else {
                std::cerr << "[WARN] Failed to initialize from GGUF in initializeComponents: " << path << std::endl;
            }
        }
    }

    // If external vocabulary exists, no need to initialize text model's vocabulary
    if (external_vocabulary_) {
        // Use external vocabulary, create a text model without initializing vocabulary
        textModel_ = std::make_unique<QwenTextModel>(config_.textOptions);
        // Call textModel_->initialize() to set initialized_ flag, but skip vocabulary initialization
        if (!textModel_->initialize(config_.configPath, true)) {
            std::cerr << "[WARN] Failed to initialize text model with external vocabulary" << std::endl;
        }
        std::cout << "[DEBUG] Using external vocabulary with size: " << external_vocabulary_->size() << std::endl;
        
        // Create tokenizer based on external vocabulary (using Qwen architecture factory) ONLY if not already created
        if (!tokenizer_) {
            TokenizerFactoryOptions opts; // Allow future override through configuration
            tokenizer_ = createTextProcessorForArchitecture("qwen", external_vocabulary_, opts);
            if (!tokenizer_) {
                std::cerr << "[ERROR] Failed to create tokenizer with external vocabulary" << std::endl;
            } else {
                std::cout << "[DEBUG] Tokenizer created via architecture factory using external vocabulary" << std::endl;
            }
        } else {
            std::cout << "[DEBUG] Reusing tokenizer created from GGUF; skip architecture factory creation" << std::endl;
        }
    } else {
        // Only create text model instance, defer initialization to loadComponentModels stage, avoid fallback vocabulary initialization logs
        textModel_ = std::make_unique<QwenTextModel>(config_.textOptions);
        std::cout << "[DEBUG] Defer QwenTextModel initialization to loadComponentModels()" << std::endl;
    }
    
    // Initialize vision model
    visionModel_ = std::make_unique<QwenVisionModel>(config_.visionOptions);
    if (!visionModel_->initialize(config_.configPath)) {
        std::cerr << "Failed to initialize vision model" << std::endl;
        return false;
    }
    
    // Initialize image processor
    // TODO: Fix linking issue with QwenImageProcessor constructor
    // imageProcessor_ = std::make_unique<QwenImageProcessor>(config_.imageProcessorConfig);
    
    // Don't take ownership of text model's vocabulary here, avoid double free/dangling pointers
    // QwenMultimodalModel::getVocabulary() will safely return external_vocabulary_ or textModel_->getVocabulary()
    
    return true;
}

bool QwenMultimodalModel::loadComponentModels() {
    bool success = true;
    
    // Load text model
    if (!external_vocabulary_) {
        if (!textModel_->initialize(config_.configPath)) {
            std::cerr << "Failed to initialize text model" << std::endl;
            success = false;
        }
        if (!config_.textModelPath.empty()) {
            success &= textModel_->loadModel(config_.textModelPath);
        }
    } else {
        std::cout << "[DEBUG] Skipping textModel_ initialize/loadModel because external_vocabulary_ is provided" << std::endl;
        // Even when external vocabulary is provided, we still need to load real weights
        if (!config_.textModelPath.empty()) {
            bool ok = textModel_->loadModel(config_.textModelPath);
            if (!ok) {
                std::cerr << "[WARN] Failed to load text model weights from: " << config_.textModelPath << std::endl;
                success = false;
            } else {
                std::cout << "[DEBUG] Loaded text model weights from GGUF: " << config_.textModelPath << std::endl;
            }
        } else {
            std::cout << "[WARN] textModelPath is empty; using zero-initialized weights which may degrade output quality" << std::endl;
        }
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
    // Record config path
    config_.configPath = configPath;

    // Allow empty path to use defaults
    if (configPath.empty()) {
        return true;
    }

    std::ifstream in(configPath);
    if (!in.is_open()) {
        std::cerr << "[ERROR] Failed to open Qwen multimodal config: " << configPath << std::endl;
        return false;
    }

    try {
        nlohmann::json j;
        in >> j;

        // Model paths
        config_.textModelPath = j.value("text_model_path", config_.textModelPath);
        config_.visionModelPath = j.value("vision_model_path", config_.visionModelPath);

        // Text options
        if (j.contains("text_options") && j["text_options"].is_object()) {
            const auto& jt = j["text_options"];
            config_.textOptions.hiddenSize = jt.value("hidden_size", config_.textOptions.hiddenSize);
            config_.textOptions.numHeads = jt.value("num_heads", config_.textOptions.numHeads);
            config_.textOptions.numKVHeads = jt.value("num_kv_heads", config_.textOptions.numKVHeads);
            config_.textOptions.ropeDim = jt.value("rope_dim", config_.textOptions.ropeDim);
            config_.textOptions.originalContextLength = jt.value("original_context_length", config_.textOptions.originalContextLength);
            config_.textOptions.eps = jt.value("eps", config_.textOptions.eps);
            config_.textOptions.ropeBase = jt.value("rope_base", config_.textOptions.ropeBase);
            config_.textOptions.ropeScale = jt.value("rope_scale", config_.textOptions.ropeScale);
            config_.textOptions.blockCount = jt.value("block_count", config_.textOptions.blockCount);
            config_.textOptions.embeddingLength = jt.value("embedding_length", config_.textOptions.embeddingLength);
        }

        // Vision options
        if (j.contains("vision_options") && j["vision_options"].is_object()) {
            const auto& jv = j["vision_options"];
            config_.visionOptions.hiddenSize = jv.value("hidden_size", config_.visionOptions.hiddenSize);
            config_.visionOptions.numHeads = jv.value("num_heads", config_.visionOptions.numHeads);
            config_.visionOptions.numLayers = jv.value("num_layers", config_.visionOptions.numLayers);
            config_.visionOptions.patchSize = jv.value("patch_size", config_.visionOptions.patchSize);
            config_.visionOptions.imageSize = jv.value("image_size", config_.visionOptions.imageSize);
            config_.visionOptions.numChannels = jv.value("num_channels", config_.visionOptions.numChannels);
            config_.visionOptions.temporalPatchSize = jv.value("temporal_patch_size", config_.visionOptions.temporalPatchSize);
            config_.visionOptions.spatialMergeSize = jv.value("spatial_merge_size", config_.visionOptions.spatialMergeSize);
            config_.visionOptions.layerNormEps = jv.value("layer_norm_eps", config_.visionOptions.layerNormEps);
        }

        // Image processor config
        if (j.contains("image_processor") && j["image_processor"].is_object()) {
            const auto& jp = j["image_processor"];
            config_.imageProcessorConfig.imageSize = jp.value("image_size", config_.imageProcessorConfig.imageSize);
            config_.imageProcessorConfig.patchSize = jp.value("patch_size", config_.imageProcessorConfig.patchSize);
            config_.imageProcessorConfig.temporalPatchSize = jp.value("temporal_patch_size", config_.imageProcessorConfig.temporalPatchSize);
            config_.imageProcessorConfig.spatialMergeSize = jp.value("spatial_merge_size", config_.imageProcessorConfig.spatialMergeSize);
            config_.imageProcessorConfig.minPixels = jp.value("min_pixels", config_.imageProcessorConfig.minPixels);
            config_.imageProcessorConfig.maxPixels = jp.value("max_pixels", config_.imageProcessorConfig.maxPixels);
            config_.imageProcessorConfig.resampleMode = jp.value("resample_mode", config_.imageProcessorConfig.resampleMode);
            config_.imageProcessorConfig.doResize = jp.value("do_resize", config_.imageProcessorConfig.doResize);
            config_.imageProcessorConfig.doNormalize = jp.value("do_normalize", config_.imageProcessorConfig.doNormalize);
            config_.imageProcessorConfig.doConvertRgb = jp.value("do_convert_rgb", config_.imageProcessorConfig.doConvertRgb);

            if (jp.contains("mean") && jp["mean"].is_array()) {
                config_.imageProcessorConfig.mean.clear();
                for (const auto& v : jp["mean"]) {
                    config_.imageProcessorConfig.mean.push_back(v.get<float>());
                }
            }
            if (jp.contains("std") && jp["std"].is_array()) {
                config_.imageProcessorConfig.std.clear();
                for (const auto& v : jp["std"]) {
                    config_.imageProcessorConfig.std.push_back(v.get<float>());
                }
            }
        }

        // Special tokens
        if (j.contains("special_tokens") && j["special_tokens"].is_object()) {
            const auto& js = j["special_tokens"];
            config_.imageTokenId = js.value("image_token_id", config_.imageTokenId);
            config_.videoTokenId = js.value("video_token_id", config_.videoTokenId);
            config_.visionStartId = js.value("vision_start_id", config_.visionStartId);
            config_.visionEndId = js.value("vision_end_id", config_.visionEndId);
            config_.visionPadId = js.value("vision_pad_id", config_.visionPadId);
        }

        // Processing parameters
        if (j.contains("processing") && j["processing"].is_object()) {
            const auto& jp = j["processing"];
            config_.maxImageTokens = jp.value("max_image_tokens", config_.maxImageTokens);
            config_.maxSequenceLength = jp.value("max_sequence_length", config_.maxSequenceLength);
            config_.useVisionPadding = jp.value("use_vision_padding", config_.useVisionPadding);
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception while parsing Qwen multimodal config: " << e.what() << std::endl;
        return false;
    }
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

// New: Factory function that accepts an external vocabulary
std::unique_ptr<BaseModel> createQwenMultimodalModel(const std::string& configPath, 
                                                     std::shared_ptr<Vocabulary> external_vocab) {
    QwenMultimodalConfig config{};  // default config
    auto model = std::make_unique<QwenMultimodalModel>(config, external_vocab);
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
    
    // Backend-aware allocation: use current ML backend if available
    {
        duorou::ml::Backend *backend = duorou::ml::BackendManager::getInstance().getCurrentBackend();
        if (backend) {
            result.data.setBackend(backend);
        }
    }
    
    // Copy data to tensor (will allocate via backend if set)
    result.data.copyFromHost(rawData.data(), rawData.size() * sizeof(float));
    
    return result;
}

// Add ML framework integration methods
bool QwenMultimodalModel::initializeMLComponents() {
    try {
        mlContext_ = std::make_unique<duorou::ml::Context>();
        // Ensure Context uses the active backend (set by the engine)
        {
            duorou::ml::Backend *backend = duorou::ml::BackendManager::getInstance().getCurrentBackend();
            if (backend) {
                mlContext_->setBackend(backend);
            }
        }
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
    // Backend-aware allocation: prefer model Context backend, fallback to global current backend
    {
        duorou::ml::Backend *backend = nullptr;
        if (mlContext_) backend = mlContext_->getBackend();
        if (!backend) backend = duorou::ml::BackendManager::getInstance().getCurrentBackend();
        if (backend) tensor.setBackend(backend);
    }
    tensor.copyFromHost(data.data(), data.size() * sizeof(int32_t));
    return tensor;
}

duorou::ml::Tensor QwenMultimodalModel::convertToTensor(const std::vector<float>& data, 
                                                        const std::vector<int64_t>& shape) {
    duorou::ml::Tensor tensor(shape, duorou::ml::DataType::FLOAT32);
    // Backend-aware allocation: prefer model Context backend, fallback to global current backend
    {
        duorou::ml::Backend *backend = nullptr;
        if (mlContext_) backend = mlContext_->getBackend();
        if (!backend) backend = duorou::ml::BackendManager::getInstance().getCurrentBackend();
        if (backend) tensor.setBackend(backend);
    }
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
        // Parse GGUF file
        ggufParser_ = std::make_unique<duorou::extensions::ollama::GGUFParser>(true);
        if (!ggufParser_->parseFile(modelPath)) {
            std::cerr << "Failed to load GGUF model: parseFile failed for " << modelPath << std::endl;
            return false;
        }

        // Create Vocabulary from GGUF using unified factory (same as tokenizer_golden_test.cpp)
        auto vocab = createVocabularyFromGGUF(*ggufParser_);
        if (!vocab) {
            std::cerr << "[ERROR] Failed to create vocabulary from GGUF: " << modelPath << std::endl;
            return false;
        }
        external_vocabulary_ = std::move(vocab);
        std::cout << "[DEBUG] External vocabulary set from GGUF, size=" << external_vocabulary_->size() << std::endl;

        // Create TextProcessor (Tokenizer) from GGUF
        TokenizerFactoryOptions opts; // defaults; can be extended via config
        tokenizer_ = createTextProcessorFromGGUF(*ggufParser_, external_vocabulary_, opts);
        if (!tokenizer_) {
            std::cerr << "[ERROR] Failed to create tokenizer from GGUF" << std::endl;
            return false;
        }
        std::cout << "[DEBUG] Tokenizer created successfully from GGUF (vocab_size=" << tokenizer_->getVocabSize() << ")" << std::endl;

        // Optional: simple roundtrip sanity check for a short text
        const std::string sanity_text = "hello";
        auto sanity_ids = tokenizer_->encode(sanity_text, /*addSpecial=*/false);
        auto sanity_decoded = tokenizer_->decode(sanity_ids);
        std::cout << "[DEBUG] GGUF tokenizer sanity roundtrip: '" << sanity_text << "' -> ids(" << sanity_ids.size() << ") -> '" << sanity_decoded << "'" << std::endl;

        return true;
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

    const auto* tensorInfo = ggufParser_->getTensorInfo(tensorName);
    if (!tensorInfo) {
        std::cerr << "Tensor not found: " << tensorName << std::endl;
        return duorou::ml::Tensor();
    }

    // Build shape
    std::vector<int64_t> shape;
    shape.reserve(tensorInfo->dimensions.size());
    for (auto dim : tensorInfo->dimensions) {
        shape.push_back(static_cast<int64_t>(dim));
    }

    // Map GGML type to internal dtype
    auto mapDType = [](duorou::extensions::ollama::GGMLTensorType t) -> duorou::ml::DataType {
        using duorou::extensions::ollama::GGMLTensorType;
        switch (t) {
            case GGMLTensorType::F32: return duorou::ml::DataType::FLOAT32;
            case GGMLTensorType::F16: return duorou::ml::DataType::FLOAT16;
            default: return duorou::ml::DataType::INT8; // store raw bytes for quantized types
        }
    };

    duorou::ml::DataType dtype = mapDType(tensorInfo->type);

    // Select backend
    duorou::ml::Backend *backend = nullptr;
    if (mlContext_) backend = mlContext_->getBackend();
    if (!backend) backend = duorou::ml::BackendManager::getInstance().getCurrentBackend();

    size_t fileBytes = ggufParser_->getTensorSize(tensorName);

    // Create tensor and allocate
    duorou::ml::Tensor tensor(shape, dtype);
    if (backend) tensor.setBackend(backend);
    tensor.allocate(backend);

    // If bytes do not match allocation (e.g., quantized layouts), fall back to raw flat buffer
    if (tensor.nbytes() != fileBytes) {
        std::cout << "[INFO] GGUF tensor size mismatch for '" << tensorName
                  << "' (allocated=" << tensor.nbytes() << ", gguf=" << fileBytes
                  << ") — using flat INT8 buffer to preserve raw data" << std::endl;
        duorou::ml::Tensor raw({static_cast<int64_t>(fileBytes)}, duorou::ml::DataType::INT8);
        if (backend) raw.setBackend(backend);
        raw.allocate(backend);
        if (!ggufParser_->readTensorData(*tensorInfo, raw.data(), fileBytes)) {
            std::cerr << "[ERROR] Failed to read GGUF tensor data: " << tensorName << std::endl;
            return duorou::ml::Tensor();
        }
        return raw;
    }

    // Read data into allocated buffer
    if (!ggufParser_->readTensorData(*tensorInfo, tensor.data(), fileBytes)) {
        std::cerr << "[ERROR] Failed to read GGUF tensor data: " << tensorName << std::endl;
        return duorou::ml::Tensor();
    }

    return tensor;
}

} // namespace model
} // namespace duorou