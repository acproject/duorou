#pragma once

// Ensure this header is only parsed by a C++ compiler
#ifdef __cplusplus

#include "base_model.h"
#include <vector>
#include <string>
#include <memory>
#include <cstdint>

namespace duorou {
namespace model {

// Image processing configuration
struct ImageProcessorConfig {
    size_t imageSize = 224;           // Target image size
    size_t patchSize = 14;            // Patch size for vision transformer
    size_t temporalPatchSize = 2;     // Temporal patch size for video
    size_t spatialMergeSize = 2;      // Spatial merge size
    size_t minPixels = 4 * 28 * 28;   // Minimum pixels
    size_t maxPixels = 16384 * 28 * 28; // Maximum pixels
    
    // Normalization parameters (ImageNet defaults)
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std = {0.229f, 0.224f, 0.225f};
    
    // Resizing parameters
    std::string resampleMode = "bicubic";
    bool doResize = true;
    bool doNormalize = true;
    bool doConvertRgb = true;
};

// Image data structure
struct ImageData {
    std::vector<float> pixelValues;
    size_t height = 0;
    size_t width = 0;
    size_t channels = 3;
    
    ImageData() = default;
    ImageData(size_t h, size_t w, size_t c = 3) 
        : height(h), width(w), channels(c) {
        pixelValues.resize(h * w * c);
    }
    
    size_t totalPixels() const { return height * width * channels; }
    bool isValid() const { return height > 0 && width > 0 && !pixelValues.empty(); }
};

// Resize result with grid information
struct ResizeResult {
    ImageData image;
    size_t gridHeight = 0;
    size_t gridWidth = 0;
    size_t gridTemporal = 1;
    
    size_t totalPatches() const { return gridHeight * gridWidth * gridTemporal; }
};

// Qwen Image Processor
class QwenImageProcessor : public ImageProcessor {
public:
    QwenImageProcessor();
    explicit QwenImageProcessor(const ImageProcessorConfig& config);
    ~QwenImageProcessor() override = default;
    
    // ImageProcessor interface implementation
    std::vector<float> processImage(const std::vector<uint8_t>& imageData) override;
    std::pair<size_t, size_t> getImageDimensions(const std::vector<uint8_t>& imageData) const override;
    bool isSupported(const std::string& format) const override;
    
    // Qwen-specific methods
    void setConfig(const ImageProcessorConfig& config);
    const ImageProcessorConfig& getConfig() const { return config_; }
    
    // Smart resize with aspect ratio preservation
    ResizeResult smartResize(const ImageData& image);
    
    // Process multiple images (for batch processing)
    std::vector<ResizeResult> processImages(const std::vector<std::vector<uint8_t>>& imagesData);
    
    // Convert raw image data to ImageData structure
    ImageData decodeImage(const std::vector<uint8_t>& imageData);
    
    // Normalize image with mean and std
    ImageData normalizeImage(const ImageData& image);
    
    // Convert to RGB if needed
    ImageData convertToRgb(const ImageData& image);
    
    // Resize image to target size
    ImageData resizeImage(const ImageData& image, size_t targetHeight, size_t targetWidth);
    
    // Calculate optimal resize dimensions
    std::pair<size_t, size_t> calculateResizeDimensions(
        size_t originalHeight, 
        size_t originalWidth,
        size_t minPixels,
        size_t maxPixels
    );
    
    // Create patches from image
    std::vector<float> createPatches(const ImageData& image);
    
private:
    ImageProcessorConfig config_;
    
    // Helper methods for image processing
    ImageData bilinearResize(const ImageData& image, size_t newHeight, size_t newWidth);
    ImageData bicubicResize(const ImageData& image, size_t newHeight, size_t newWidth);
    
    // Image format detection
    std::string detectImageFormat(const std::vector<uint8_t>& imageData) const;
    
    // Simple image decoders (for basic formats)
    ImageData decodePng(const std::vector<uint8_t>& imageData);
    ImageData decodeJpeg(const std::vector<uint8_t>& imageData);
    ImageData decodeBmp(const std::vector<uint8_t>& imageData);
    
    // Utility functions
    float cubicInterpolate(float p0, float p1, float p2, float p3, float t);
    float clamp(float value, float min, float max);
    
    // Grid calculation for patches
    std::tuple<size_t, size_t, size_t> calculateGrid(size_t height, size_t width);
};

// Factory function for creating Qwen image processors
std::unique_ptr<ImageProcessor> createQwenImageProcessor(const ImageProcessorConfig& config = {});

// Utility functions for image processing
namespace ImageUtils {
    // Convert between different color spaces
    std::vector<float> rgbToFloat(const std::vector<uint8_t>& rgb);
    std::vector<uint8_t> floatToRgb(const std::vector<float>& floatData);
    
    // Image statistics
    std::pair<float, float> calculateMeanStd(const std::vector<float>& data);
    
    // Aspect ratio utilities
    float calculateAspectRatio(size_t width, size_t height);
    std::pair<size_t, size_t> maintainAspectRatio(
        size_t originalWidth, 
        size_t originalHeight, 
        size_t targetSize
    );
    
    // Patch utilities
    size_t calculateNumPatches(size_t imageSize, size_t patchSize);
    std::vector<size_t> calculatePatchIndices(
        size_t height, 
        size_t width, 
        size_t patchSize
    );
}

} // namespace model
} // namespace duorou

#endif // __cplusplus