#include "qwen_image_processor.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstring>

namespace duorou {
namespace model {

// QwenImageProcessor implementation
QwenImageProcessor::QwenImageProcessor() {
    config_ = ImageProcessorConfig{};
}

QwenImageProcessor::QwenImageProcessor(const ImageProcessorConfig& config) 
    : config_(config) {
}

std::vector<float> QwenImageProcessor::processImage(const std::vector<uint8_t>& imageData) {
    // Decode image
    auto image = decodeImage(imageData);
    if (!image.isValid()) {
        std::cerr << "Failed to decode image" << std::endl;
        return {};
    }
    
    // Convert to RGB if needed
    if (config_.doConvertRgb) {
        image = convertToRgb(image);
    }
    
    // Smart resize
    auto resizeResult = smartResize(image);
    
    // Normalize
    if (config_.doNormalize) {
        resizeResult.image = normalizeImage(resizeResult.image);
    }
    
    // Create patches
    return createPatches(resizeResult.image);
}

std::pair<size_t, size_t> QwenImageProcessor::getImageDimensions(const std::vector<uint8_t>& imageData) const {
    // Simple dimension detection - in real implementation, would parse image headers
    // For now, return default dimensions
    return {config_.imageSize, config_.imageSize};
}

bool QwenImageProcessor::isSupported(const std::string& format) const {
    // Support common image formats
    return format == "png" || format == "jpg" || format == "jpeg" || format == "bmp";
}

ResizeResult QwenImageProcessor::smartResize(const ImageData& image) {
    ResizeResult result;
    
    // Calculate optimal dimensions
    auto [newHeight, newWidth] = calculateResizeDimensions(
        image.height, image.width, config_.minPixels, config_.maxPixels
    );
    
    // Resize image
    result.image = resizeImage(image, newHeight, newWidth);
    
    // Calculate grid
    auto [gridH, gridW, gridT] = calculateGrid(newHeight, newWidth);
    result.gridHeight = gridH;
    result.gridWidth = gridW;
    result.gridTemporal = gridT;
    
    return result;
}

std::vector<ResizeResult> QwenImageProcessor::processImages(const std::vector<std::vector<uint8_t>>& imagesData) {
    std::vector<ResizeResult> results;
    results.reserve(imagesData.size());
    
    for (const auto& imageData : imagesData) {
        auto image = decodeImage(imageData);
        if (image.isValid()) {
            if (config_.doConvertRgb) {
                image = convertToRgb(image);
            }
            auto resizeResult = smartResize(image);
            if (config_.doNormalize) {
                resizeResult.image = normalizeImage(resizeResult.image);
            }
            results.push_back(resizeResult);
        }
    }
    
    return results;
}

ImageData QwenImageProcessor::decodeImage(const std::vector<uint8_t>& imageData) {
    std::string format = detectImageFormat(imageData);
    
    if (format == "png") {
        return decodePng(imageData);
    } else if (format == "jpg" || format == "jpeg") {
        return decodeJpeg(imageData);
    } else if (format == "bmp") {
        return decodeBmp(imageData);
    }
    
    // Fallback: assume raw RGB data
    size_t totalPixels = imageData.size() / 3;
    size_t imageSize = static_cast<size_t>(std::sqrt(totalPixels));
    
    ImageData image(imageSize, imageSize, 3);
    for (size_t i = 0; i < imageData.size(); ++i) {
        image.pixelValues[i] = static_cast<float>(imageData[i]) / 255.0f;
    }
    
    return image;
}

ImageData QwenImageProcessor::normalizeImage(const ImageData& image) {
    ImageData normalized = image;
    
    // Apply normalization: (pixel - mean) / std
    for (size_t c = 0; c < image.channels; ++c) {
        float mean = (c < config_.mean.size()) ? config_.mean[c] : 0.5f;
        float std = (c < config_.std.size()) ? config_.std[c] : 0.5f;
        
        for (size_t h = 0; h < image.height; ++h) {
            for (size_t w = 0; w < image.width; ++w) {
                size_t idx = (h * image.width + w) * image.channels + c;
                normalized.pixelValues[idx] = (image.pixelValues[idx] - mean) / std;
            }
        }
    }
    
    return normalized;
}

ImageData QwenImageProcessor::convertToRgb(const ImageData& image) {
    // If already RGB, return as is
    if (image.channels == 3) {
        return image;
    }
    
    // Convert grayscale to RGB
    if (image.channels == 1) {
        ImageData rgb(image.height, image.width, 3);
        for (size_t h = 0; h < image.height; ++h) {
            for (size_t w = 0; w < image.width; ++w) {
                float gray = image.pixelValues[h * image.width + w];
                size_t rgbIdx = (h * image.width + w) * 3;
                rgb.pixelValues[rgbIdx] = gray;     // R
                rgb.pixelValues[rgbIdx + 1] = gray; // G
                rgb.pixelValues[rgbIdx + 2] = gray; // B
            }
        }
        return rgb;
    }
    
    // For other formats, return as is for now
    return image;
}

ImageData QwenImageProcessor::resizeImage(const ImageData& image, size_t targetHeight, size_t targetWidth) {
    if (image.height == targetHeight && image.width == targetWidth) {
        return image;
    }
    
    if (config_.resampleMode == "bicubic") {
        return bicubicResize(image, targetHeight, targetWidth);
    } else {
        return bilinearResize(image, targetHeight, targetWidth);
    }
}

std::pair<size_t, size_t> QwenImageProcessor::calculateResizeDimensions(
    size_t originalHeight, 
    size_t originalWidth,
    size_t minPixels,
    size_t maxPixels) {
    
    size_t totalPixels = originalHeight * originalWidth;
    
    // If within bounds, return original dimensions
    if (totalPixels >= minPixels && totalPixels <= maxPixels) {
        return {originalHeight, originalWidth};
    }
    
    // Calculate scale factor
    float scale = 1.0f;
    if (totalPixels < minPixels) {
        scale = std::sqrt(static_cast<float>(minPixels) / totalPixels);
    } else if (totalPixels > maxPixels) {
        scale = std::sqrt(static_cast<float>(maxPixels) / totalPixels);
    }
    
    size_t newHeight = static_cast<size_t>(originalHeight * scale);
    size_t newWidth = static_cast<size_t>(originalWidth * scale);
    
    // Ensure dimensions are multiples of patch size
    newHeight = (newHeight / config_.patchSize) * config_.patchSize;
    newWidth = (newWidth / config_.patchSize) * config_.patchSize;
    
    // Ensure minimum size
    newHeight = std::max(newHeight, config_.patchSize);
    newWidth = std::max(newWidth, config_.patchSize);
    
    return {newHeight, newWidth};
}

std::vector<float> QwenImageProcessor::createPatches(const ImageData& image) {
    size_t patchHeight = image.height / config_.patchSize;
    size_t patchWidth = image.width / config_.patchSize;
    size_t numPatches = patchHeight * patchWidth;
    size_t patchDim = config_.patchSize * config_.patchSize * image.channels;
    
    std::vector<float> patches(numPatches * patchDim);
    
    for (size_t ph = 0; ph < patchHeight; ++ph) {
        for (size_t pw = 0; pw < patchWidth; ++pw) {
            size_t patchIdx = ph * patchWidth + pw;
            
            for (size_t y = 0; y < config_.patchSize; ++y) {
                for (size_t x = 0; x < config_.patchSize; ++x) {
                    size_t imgY = ph * config_.patchSize + y;
                    size_t imgX = pw * config_.patchSize + x;
                    
                    for (size_t c = 0; c < image.channels; ++c) {
                        size_t imgIdx = (imgY * image.width + imgX) * image.channels + c;
                        size_t patchPixelIdx = (y * config_.patchSize + x) * image.channels + c;
                        size_t finalIdx = patchIdx * patchDim + patchPixelIdx;
                        
                        patches[finalIdx] = image.pixelValues[imgIdx];
                    }
                }
            }
        }
    }
    
    return patches;
}

ImageData QwenImageProcessor::bilinearResize(const ImageData& image, size_t newHeight, size_t newWidth) {
    ImageData resized(newHeight, newWidth, image.channels);
    
    float scaleY = static_cast<float>(image.height) / newHeight;
    float scaleX = static_cast<float>(image.width) / newWidth;
    
    for (size_t y = 0; y < newHeight; ++y) {
        for (size_t x = 0; x < newWidth; ++x) {
            float srcY = y * scaleY;
            float srcX = x * scaleX;
            
            size_t y1 = static_cast<size_t>(srcY);
            size_t x1 = static_cast<size_t>(srcX);
            size_t y2 = std::min(y1 + 1, image.height - 1);
            size_t x2 = std::min(x1 + 1, image.width - 1);
            
            float dy = srcY - y1;
            float dx = srcX - x1;
            
            for (size_t c = 0; c < image.channels; ++c) {
                float p11 = image.pixelValues[(y1 * image.width + x1) * image.channels + c];
                float p12 = image.pixelValues[(y1 * image.width + x2) * image.channels + c];
                float p21 = image.pixelValues[(y2 * image.width + x1) * image.channels + c];
                float p22 = image.pixelValues[(y2 * image.width + x2) * image.channels + c];
                
                float interpolated = p11 * (1 - dx) * (1 - dy) +
                                   p12 * dx * (1 - dy) +
                                   p21 * (1 - dx) * dy +
                                   p22 * dx * dy;
                
                resized.pixelValues[(y * newWidth + x) * image.channels + c] = interpolated;
            }
        }
    }
    
    return resized;
}

ImageData QwenImageProcessor::bicubicResize(const ImageData& image, size_t newHeight, size_t newWidth) {
    // For simplicity, fall back to bilinear for now
    // In a full implementation, would use proper bicubic interpolation
    return bilinearResize(image, newHeight, newWidth);
}

std::string QwenImageProcessor::detectImageFormat(const std::vector<uint8_t>& imageData) const {
    if (imageData.size() < 8) return "unknown";
    
    // PNG signature
    if (imageData[0] == 0x89 && imageData[1] == 0x50 && imageData[2] == 0x4E && imageData[3] == 0x47) {
        return "png";
    }
    
    // JPEG signature
    if (imageData[0] == 0xFF && imageData[1] == 0xD8) {
        return "jpg";
    }
    
    // BMP signature
    if (imageData[0] == 0x42 && imageData[1] == 0x4D) {
        return "bmp";
    }
    
    return "unknown";
}

ImageData QwenImageProcessor::decodePng(const std::vector<uint8_t>& imageData) {
    // Simplified PNG decoder - in real implementation would use libpng
    // For now, return a placeholder
    ImageData image(config_.imageSize, config_.imageSize, 3);
    std::fill(image.pixelValues.begin(), image.pixelValues.end(), 0.5f);
    return image;
}

ImageData QwenImageProcessor::decodeJpeg(const std::vector<uint8_t>& imageData) {
    // Simplified JPEG decoder - in real implementation would use libjpeg
    // For now, return a placeholder
    ImageData image(config_.imageSize, config_.imageSize, 3);
    std::fill(image.pixelValues.begin(), image.pixelValues.end(), 0.5f);
    return image;
}

ImageData QwenImageProcessor::decodeBmp(const std::vector<uint8_t>& imageData) {
    // Simplified BMP decoder
    // For now, return a placeholder
    ImageData image(config_.imageSize, config_.imageSize, 3);
    std::fill(image.pixelValues.begin(), image.pixelValues.end(), 0.5f);
    return image;
}

float QwenImageProcessor::cubicInterpolate(float p0, float p1, float p2, float p3, float t) {
    float a = -0.5f * p0 + 1.5f * p1 - 1.5f * p2 + 0.5f * p3;
    float b = p0 - 2.5f * p1 + 2.0f * p2 - 0.5f * p3;
    float c = -0.5f * p0 + 0.5f * p2;
    float d = p1;
    
    return a * t * t * t + b * t * t + c * t + d;
}

float QwenImageProcessor::clamp(float value, float min, float max) {
    return std::max(min, std::min(max, value));
}

std::tuple<size_t, size_t, size_t> QwenImageProcessor::calculateGrid(size_t height, size_t width) {
    size_t gridHeight = height / config_.patchSize;
    size_t gridWidth = width / config_.patchSize;
    size_t gridTemporal = 1; // For images, temporal dimension is 1
    
    return {gridHeight, gridWidth, gridTemporal};
}

void QwenImageProcessor::setConfig(const ImageProcessorConfig& config) {
    config_ = config;
}

// Factory function
std::unique_ptr<ImageProcessor> createQwenImageProcessor(const ImageProcessorConfig& config) {
    return std::make_unique<QwenImageProcessor>(config);
}

// ImageUtils namespace implementation
namespace ImageUtils {

std::vector<float> rgbToFloat(const std::vector<uint8_t>& rgb) {
    std::vector<float> floatData(rgb.size());
    for (size_t i = 0; i < rgb.size(); ++i) {
        floatData[i] = static_cast<float>(rgb[i]) / 255.0f;
    }
    return floatData;
}

std::vector<uint8_t> floatToRgb(const std::vector<float>& floatData) {
    std::vector<uint8_t> rgb(floatData.size());
    for (size_t i = 0; i < floatData.size(); ++i) {
        float clamped = std::max(0.0f, std::min(1.0f, floatData[i]));
        rgb[i] = static_cast<uint8_t>(clamped * 255.0f);
    }
    return rgb;
}

std::pair<float, float> calculateMeanStd(const std::vector<float>& data) {
    if (data.empty()) return {0.0f, 1.0f};
    
    float mean = 0.0f;
    for (float val : data) {
        mean += val;
    }
    mean /= data.size();
    
    float variance = 0.0f;
    for (float val : data) {
        float diff = val - mean;
        variance += diff * diff;
    }
    variance /= data.size();
    
    return {mean, std::sqrt(variance)};
}

float calculateAspectRatio(size_t width, size_t height) {
    return static_cast<float>(width) / static_cast<float>(height);
}

std::pair<size_t, size_t> maintainAspectRatio(
    size_t originalWidth, 
    size_t originalHeight, 
    size_t targetSize) {
    
    float aspectRatio = calculateAspectRatio(originalWidth, originalHeight);
    
    size_t newWidth, newHeight;
    if (aspectRatio > 1.0f) {
        // Landscape
        newWidth = targetSize;
        newHeight = static_cast<size_t>(targetSize / aspectRatio);
    } else {
        // Portrait or square
        newHeight = targetSize;
        newWidth = static_cast<size_t>(targetSize * aspectRatio);
    }
    
    return {newWidth, newHeight};
}

size_t calculateNumPatches(size_t imageSize, size_t patchSize) {
    return (imageSize / patchSize) * (imageSize / patchSize);
}

std::vector<size_t> calculatePatchIndices(
    size_t height, 
    size_t width, 
    size_t patchSize) {
    
    size_t patchHeight = height / patchSize;
    size_t patchWidth = width / patchSize;
    size_t numPatches = patchHeight * patchWidth;
    
    std::vector<size_t> indices(numPatches);
    for (size_t i = 0; i < numPatches; ++i) {
        indices[i] = i;
    }
    
    return indices;
}

} // namespace ImageUtils

} // namespace model
} // namespace duorou