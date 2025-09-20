#include "qwen25vl_model.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace duorou {
namespace model {

// Qwen25VLConfig 实现
bool Qwen25VLConfig::loadFromFile(const std::string &configPath) {
  // 简化实现，实际需要解析JSON配置文件
  return true;
}

bool Qwen25VLConfig::validate() const {
  return hiddenSize > 0 && numHeads > 0 && numLayers > 0 && vocabSize > 0;
}

// PixelValues 实现
PixelValues PixelValues::fromImageData(const std::vector<float> &imageData,
                                       size_t height, size_t width,
                                       size_t channels) {
  ml::Tensor tensor({static_cast<int64_t>(height), static_cast<int64_t>(width),
                     static_cast<int64_t>(channels)});
  tensor.allocate();

  // 复制数据
  float *tensorData = static_cast<float *>(tensor.data());
  std::copy(imageData.begin(), imageData.end(), tensorData);

  Grid grid(height, width, 1);
  return PixelValues(tensor, grid);
}

// QwenImageProcessor 实现
QwenImageProcessor::QwenImageProcessor(const Qwen25VLConfig &config)
    : config_(config) {}

PixelValues
QwenImageProcessor::processImage(const std::vector<uint8_t> &imageData,
                                 size_t width, size_t height, size_t channels) {
  // 简化的图像处理实现
  auto [newWidth, newHeight] = smartResize(height, width);

  // 转换为float并resize
  std::vector<float> resized =
      resize(imageData, width, height, channels, newWidth, newHeight);

  // 归一化
  std::vector<float> normalized = normalize(resized);

  return PixelValues::fromImageData(normalized, newHeight, newWidth, channels);
}

std::pair<size_t, size_t> QwenImageProcessor::smartResize(size_t height,
                                                          size_t width) const {
  // 简化的智能缩放实现
  size_t maxPixels = config_.maxPixels;
  size_t minPixels = config_.minPixels;

  size_t currentPixels = height * width;

  if (currentPixels <= minPixels) {
    return {width, height};
  }

  if (currentPixels > maxPixels) {
    double scale = std::sqrt(static_cast<double>(maxPixels) / currentPixels);
    return {static_cast<size_t>(width * scale),
            static_cast<size_t>(height * scale)};
  }

  return {width, height};
}

std::vector<float>
QwenImageProcessor::normalize(const std::vector<float> &pixels) const {
  std::vector<float> normalized(pixels.size());

  for (size_t i = 0; i < pixels.size(); ++i) {
    size_t channel = i % config_.numChannels;
    normalized[i] =
        (pixels[i] * config_.rescaleFactor - config_.imageMean[channel]) /
        config_.imageStd[channel];
  }

  return normalized;
}

std::vector<float>
QwenImageProcessor::resize(const std::vector<uint8_t> &imageData,
                           size_t origWidth, size_t origHeight, size_t channels,
                           size_t newWidth, size_t newHeight) const {
  // 简化的双线性插值resize实现
  std::vector<float> resized(newWidth * newHeight * channels);

  double xRatio = static_cast<double>(origWidth) / newWidth;
  double yRatio = static_cast<double>(origHeight) / newHeight;

  for (size_t y = 0; y < newHeight; ++y) {
    for (size_t x = 0; x < newWidth; ++x) {
      size_t origX = static_cast<size_t>(x * xRatio);
      size_t origY = static_cast<size_t>(y * yRatio);

      // 确保不越界
      origX = std::min(origX, origWidth - 1);
      origY = std::min(origY, origHeight - 1);

      for (size_t c = 0; c < channels; ++c) {
        size_t origIdx = (origY * origWidth + origX) * channels + c;
        size_t newIdx = (y * newWidth + x) * channels + c;
        resized[newIdx] = static_cast<float>(imageData[origIdx]);
      }
    }
  }

  return resized;
}

// VisionSelfAttention 实现
VisionSelfAttention::VisionSelfAttention(size_t hiddenSize, size_t numHeads,
                                         size_t headDim)
    : hiddenSize_(hiddenSize), numHeads_(numHeads), headDim_(headDim) {
  // 注意：这里需要在实际使用时通过Context创建Linear层
  // 暂时留空，在loadWeights时初始化
}

ml::Tensor VisionSelfAttention::forward(ml::Context &ctx,
                                        const ml::Tensor &hiddenStates,
                                        const ml::Tensor &cos,
                                        const ml::Tensor &sin,
                                        const ml::Tensor &mask) const {
  // 简化实现，返回输入
  return hiddenStates;
}

bool VisionSelfAttention::loadWeights(
    const std::map<std::string, ml::Tensor> &weights,
    const std::string &prefix) {
  // 简化实现
  return true;
}

ml::Tensor VisionSelfAttention::rotateHalf(ml::Context &ctx,
                                           const ml::Tensor &tensor) const {
  // 简化实现
  return tensor;
}

ml::Tensor VisionSelfAttention::applyRotaryEmbedding(
    ml::Context &ctx, const ml::Tensor &tensor, const ml::Tensor &cos,
    const ml::Tensor &sin) const {
  // 简化实现
  return tensor;
}

// VisionMLP 实现
VisionMLP::VisionMLP(size_t hiddenSize, size_t intermediateSize) {
  // 简化实现，实际需要初始化Linear层
}

ml::Tensor VisionMLP::forward(ml::Context &ctx,
                              const ml::Tensor &hiddenStates) const {
  // 简化实现
  return hiddenStates;
}

bool VisionMLP::loadWeights(const std::map<std::string, ml::Tensor> &weights,
                            const std::string &prefix) {
  return true;
}

// VisionLayer 实现
VisionLayer::VisionLayer(size_t hiddenSize, size_t numHeads,
                         size_t intermediateSize) {
  // 简化实现
}

ml::Tensor VisionLayer::forward(ml::Context &ctx,
                                const ml::Tensor &hiddenStates,
                                const ml::Tensor &cos, const ml::Tensor &sin,
                                const ml::Tensor &mask) const {
  return hiddenStates;
}

bool VisionLayer::loadWeights(const std::map<std::string, ml::Tensor> &weights,
                              const std::string &prefix) {
  return true;
}

// QwenVisionModel 实现
QwenVisionModel::QwenVisionModel(const Qwen25VLConfig &config)
    : config_(config) {
  // 简化实现
}

ml::Tensor QwenVisionModel::forward(ml::Context &ctx,
                                    const PixelValues &pixelValues) const {
  return pixelValues.data;
}

bool QwenVisionModel::loadWeights(
    const std::map<std::string, ml::Tensor> &weights) {
  return true;
}

ml::Tensor QwenVisionModel::createRotaryEmbedding(ml::Context &ctx,
                                                  size_t seqLen) const {
  // 简化实现
  return ml::Tensor({static_cast<int64_t>(seqLen),
                     static_cast<int64_t>(config_.visionHiddenSize)});
}

ml::Tensor
QwenVisionModel::createAttentionMask(ml::Context &ctx, size_t seqLen,
                                     const std::vector<size_t> &bounds) const {
  // 简化实现
  return ml::Tensor(
      {static_cast<int64_t>(seqLen), static_cast<int64_t>(seqLen)});
}

// 工厂函数
std::unique_ptr<Qwen25VLModel>
createQwen25VLModel(const std::string &configPath) {
  Qwen25VLConfig config;
  if (!configPath.empty()) {
    config.loadFromFile(configPath);
  }
  return std::make_unique<Qwen25VLModel>(config);
}

} // namespace model
} // namespace duorou