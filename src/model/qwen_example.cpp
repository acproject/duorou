#include "qwen_multimodal_model.h"
#include "qwen_text_model.h"
#include "qwen_vision_model.h"
#include "qwen_image_processor.h"
#include <iostream>
#include <memory>

using namespace duorou::model;

int main() {
    std::cout << "Qwen Model C++ Wrapper Example\n";
    std::cout << "================================\n\n";

    // 1. Create text model
    std::cout << "1. Creating Qwen Text Model...\n";
    auto textModel = createQwenTextModel("config/qwen_text_config.json");
    if (textModel) {
        std::cout << "   ✓ Text model created successfully\n";
        std::cout << "   ✓ Model type: " << textModel->getModelType() << "\n";
        std::cout << "   ✓ Vocab size: " << textModel->getVocabSize() << "\n";
        
        // Example text encoding/decoding
        std::string inputText = "Hello, how are you?";
        auto tokens = textModel->encode(inputText);
        auto decoded = textModel->decode(tokens);
        std::cout << "   ✓ Encoded " << tokens.size() << " tokens, decoded: \"" << decoded << "\"\n";
    } else {
        std::cout << "   ✗ Failed to create text model\n";
    }

    // 2. Create vision model
    std::cout << "\n2. Creating Qwen Vision Model...\n";
    auto visionModel = createQwenVisionModel("config/qwen_vision_config.json");
    if (visionModel) {
        std::cout << "   ✓ Vision model created successfully\n";
        
        // Example image processing
        std::vector<uint8_t> dummyImageData(224 * 224 * 3, 128); // Dummy RGB image
        auto features = visionModel->processImage(dummyImageData);
        auto dims = visionModel->getImageFeatureDims();
        std::cout << "   ✓ Processed image: " << features.size() << " features\n";
        std::cout << "   ✓ Feature dimensions: " << dims.first << "x" << dims.second << "\n";
    } else {
        std::cout << "   ✗ Failed to create vision model\n";
    }

    // 3. Create image processor
    std::cout << "\n3. Creating Qwen Image Processor...\n";
    ImageProcessorConfig config;
    config.imageSize = 224;
    config.doNormalize = true;
    config.doResize = true;
    
    auto imageProcessor = std::make_unique<QwenImageProcessor>(config);
    if (imageProcessor) {
        std::cout << "   ✓ Image processor created successfully\n";
        
        // Example image processing
        std::vector<uint8_t> rawImageData(640 * 480 * 3, 100); // Dummy image
        auto processedImage = imageProcessor->processImage(rawImageData);
        auto dims = imageProcessor->getImageDimensions(rawImageData);
        std::cout << "   ✓ Processed image data: " << processedImage.size() << " floats\n";
        std::cout << "   ✓ Original dimensions: " << dims.first << "x" << dims.second << "\n";
    } else {
        std::cout << "   ✗ Failed to create image processor\n";
    }

    // 4. Create multimodal model
    std::cout << "\n4. Creating Qwen Multimodal Model...\n";
    auto multimodalModel = createQwenMultimodalModel("config/qwen_multimodal_config.json");
    if (multimodalModel) {
        std::cout << "   ✓ Multimodal model created successfully\n";
        std::cout << "   ✓ Model type: " << multimodalModel->getModelType() << "\n";
        
        // Example text processing
        std::string prompt = "Describe this image:";
        auto tokens = multimodalModel->encode(prompt);
        std::cout << "   ✓ Encoded prompt: " << tokens.size() << " tokens\n";
    } else {
        std::cout << "   ✗ Failed to create multimodal model\n";
    }

    std::cout << "\n================================\n";
    std::cout << "Qwen Model Example Completed!\n";
    
    return 0;
}