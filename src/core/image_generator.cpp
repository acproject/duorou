#include "image_generator.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <cstring>
#include <thread>
#include "../../third_party/stable-diffusion.cpp/stable-diffusion.h"

// Simple image processing functions (actual projects may need specialized image libraries)
#include "../../third_party/stable-diffusion.cpp/thirdparty/stb_image.h"
#include "../../third_party/stable-diffusion.cpp/thirdparty/stb_image_write.h"
#include "../../third_party/stable-diffusion.cpp/thirdparty/stb_image_resize.h"

namespace duorou {
namespace core {

ImageGenerator::ImageGenerator(sd_ctx_t* sd_ctx)
    : sd_ctx_(sd_ctx), total_generation_time_(0.0), generation_count_(0) {
    if (!sd_ctx_) {
        throw std::invalid_argument("SD context cannot be null");
    }
    
    // Initialize model information
    model_info_ = "Stable Diffusion Model";
    max_size_ = {1024, 1024};  // Default maximum size
    
    // Recommended image sizes
    recommended_sizes_ = {
        {512, 512},   // Standard square
        {768, 768},   // High quality square
        {512, 768},   // Portrait
        {768, 512},   // Landscape
        {1024, 1024}, // High resolution square
        {512, 1024},  // Long portrait
        {1024, 512}   // Long landscape
    };
}

ImageGenerator::~ImageGenerator() {
    // Clean up resources
}

ImageGenerationResult ImageGenerator::textToImage(const std::string& prompt, 
                                                 const ImageGenerationParams& params) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    ImageGenerationResult result;
    
    try {
        // Validate parameters
        if (!validateParams(params)) {
            result.error_message = "Invalid generation parameters";
            return result;
        }
        
        // Preprocess prompt
        std::string processed_prompt = preprocessPrompt(prompt);
        
        // Initialize random seed
        int64_t actual_seed = initializeRNG(params.seed);
        result.seed_used = actual_seed;
        
        // Convert sampler
        sample_method_t sampler = convertSampler(params.sampler);
        
        // Set image generation parameters
        sd_img_gen_params_t gen_params;
        sd_img_gen_params_init(&gen_params);
        
        gen_params.prompt = processed_prompt.c_str();
        gen_params.negative_prompt = params.negative_prompt.c_str();
        gen_params.clip_skip = params.clip_skip;
        gen_params.guidance.txt_cfg = params.cfg_scale;
        gen_params.width = params.width;
        gen_params.height = params.height;
        gen_params.sample_method = sampler;
        gen_params.sample_steps = params.steps;
        gen_params.seed = actual_seed;
        gen_params.batch_count = 1;
        
        // Call stable-diffusion.cpp to generate image
        sd_image_t* sd_image = generate_image(sd_ctx_, &gen_params);
        
        if (!sd_image) {
            result.error_message = "Failed to generate image";
            return result;
        }
        
        // Convert result
        result = convertSDImage(sd_image);
        
        // Clean up SD image
        if (sd_image->data) {
            free(sd_image->data);
        }
        free(sd_image);
        
    } catch (const std::exception& e) {
        result.error_message = std::string("Exception: ") + e.what();
        result.success = false;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    result.generation_time = duration.count() / 1000.0;
    
    // 更新统计信息
    total_generation_time_ += result.generation_time;
    generation_count_++;
    
    return result;
}

ImageGenerationResult ImageGenerator::textToImageWithProgress(const std::string& prompt,
                                                            ProgressCallback callback,
                                                            const ImageGenerationParams& params) {
    // 注意：stable-diffusion.cpp可能不直接支持进度回调
    // 这里提供一个基本实现，实际使用时可能需要修改库或使用其他方法
    
    if (callback) {
        // 模拟进度回调
        for (int i = 0; i <= params.steps; ++i) {
            float progress = static_cast<float>(i) / params.steps;
            callback(i, params.steps, progress);
            
            if (i < params.steps) {
                // 简单的延时模拟
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
    }
    
    return textToImage(prompt, params);
}

ImageGenerationResult ImageGenerator::imageToImage(const std::string& prompt,
                                                  const std::vector<uint8_t>& input_image,
                                                  int input_width,
                                                  int input_height,
                                                  const ImageGenerationParams& params) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    ImageGenerationResult result;
    
    try {
        // 验证参数
        if (!validateParams(params)) {
            result.error_message = "Invalid generation parameters";
            return result;
        }
        
        if (input_image.empty() || input_width <= 0 || input_height <= 0) {
            result.error_message = "Invalid input image";
            return result;
        }
        
        // 预处理提示词
        std::string processed_prompt = preprocessPrompt(prompt);
        
        // 初始化随机种子
        int64_t actual_seed = initializeRNG(params.seed);
        result.seed_used = actual_seed;
        
        // 转换采样器
        sample_method_t sampler = convertSampler(params.sampler);
        
        // 准备输入图像数据
        sd_image_t init_image;
        init_image.width = input_width;
        init_image.height = input_height;
        init_image.channel = 3;  // RGB
        init_image.data = const_cast<uint8_t*>(input_image.data());
        
        // 设置图像生成参数
        sd_img_gen_params_t gen_params;
        sd_img_gen_params_init(&gen_params);
        
        gen_params.prompt = processed_prompt.c_str();
        gen_params.negative_prompt = params.negative_prompt.c_str();
        gen_params.clip_skip = params.clip_skip;
        gen_params.guidance.img_cfg = params.cfg_scale;
        gen_params.width = params.width;
        gen_params.height = params.height;
        gen_params.sample_method = sampler;
        gen_params.sample_steps = params.steps;
        gen_params.strength = params.strength;
        gen_params.seed = actual_seed;
        gen_params.batch_count = 1;
        gen_params.init_image = init_image;
        
        // 调用stable-diffusion.cpp进行图像到图像生成
        sd_image_t* sd_image = generate_image(sd_ctx_, &gen_params);
        
        if (!sd_image) {
            result.error_message = "Failed to generate image";
            return result;
        }
        
        // 转换结果
        result = convertSDImage(sd_image);
        
        // 清理SD图像
        if (sd_image->data) {
            free(sd_image->data);
        }
        free(sd_image);
        
    } catch (const std::exception& e) {
        result.error_message = std::string("Exception: ") + e.what();
        result.success = false;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    result.generation_time = duration.count() / 1000.0;
    
    // 更新统计信息
    total_generation_time_ += result.generation_time;
    generation_count_++;
    
    return result;
}

bool ImageGenerator::canGenerate() const {
    return sd_ctx_ != nullptr;
}

std::string ImageGenerator::getModelInfo() const {
    return model_info_;
}

std::pair<int, int> ImageGenerator::getMaxImageSize() const {
    return max_size_;
}

std::vector<std::pair<int, int>> ImageGenerator::getRecommendedSizes() const {
    return recommended_sizes_;
}

bool ImageGenerator::validateParams(const ImageGenerationParams& params) const {
    // 验证图像尺寸
    if (!validateImageSize(params.width, params.height)) {
        return false;
    }
    
    // 验证步数
    if (params.steps < 1 || params.steps > 150) {
        return false;
    }
    
    // 验证CFG scale
    if (params.cfg_scale < 0.0f || params.cfg_scale > 30.0f) {
        return false;
    }
    
    // 验证强度
    if (params.strength < 0.0f || params.strength > 1.0f) {
        return false;
    }
    
    return true;
}

double ImageGenerator::estimateGenerationTime(const ImageGenerationParams& params) const {
    if (generation_count_ == 0) {
        // 基于经验的估算
        double base_time = 2.0;  // 基础时间（秒）
        double step_factor = params.steps / 20.0;  // 步数因子
        double size_factor = (params.width * params.height) / (512.0 * 512.0);  // 尺寸因子
        
        return base_time * step_factor * size_factor;
    } else {
        // 基于历史数据的估算
        double avg_time = total_generation_time_ / generation_count_;
        double step_factor = params.steps / 20.0;
        double size_factor = (params.width * params.height) / (512.0 * 512.0);
        
        return avg_time * step_factor * size_factor;
    }
}

bool ImageGenerator::saveImage(const ImageGenerationResult& result, 
                              const std::string& file_path, 
                              const std::string& format) {
    if (!result.success || result.image_data.empty()) {
        return false;
    }
    
    int success = 0;
    
    if (format == "png") {
        success = stbi_write_png(file_path.c_str(), 
                                result.width, 
                                result.height, 
                                result.channels, 
                                result.image_data.data(), 
                                result.width * result.channels);
    } else if (format == "jpg" || format == "jpeg") {
        success = stbi_write_jpg(file_path.c_str(), 
                                result.width, 
                                result.height, 
                                result.channels, 
                                result.image_data.data(), 
                                90);  // 质量90%
    } else if (format == "bmp") {
        success = stbi_write_bmp(file_path.c_str(), 
                                result.width, 
                                result.height, 
                                result.channels, 
                                result.image_data.data());
    }
    
    return success != 0;
}

std::vector<uint8_t> ImageGenerator::loadImage(const std::string& file_path,
                                              int& width,
                                              int& height,
                                              int& channels) {
    unsigned char* data = stbi_load(file_path.c_str(), &width, &height, &channels, 0);
    
    if (!data) {
        width = height = channels = 0;
        return {};
    }
    
    size_t data_size = width * height * channels;
    std::vector<uint8_t> result(data, data + data_size);
    
    stbi_image_free(data);
    
    return result;
}

std::vector<uint8_t> ImageGenerator::resizeImage(const std::vector<uint8_t>& image_data,
                                                 int input_width,
                                                 int input_height,
                                                 int output_width,
                                                 int output_height,
                                                 int channels) {
    if (image_data.empty() || input_width <= 0 || input_height <= 0 || 
        output_width <= 0 || output_height <= 0 || channels <= 0) {
        return {};
    }
    
    std::vector<uint8_t> output_data(output_width * output_height * channels);
    
    int success = stbir_resize_uint8(image_data.data(), input_width, input_height, 0,
                                    output_data.data(), output_width, output_height, 0,
                                    channels);
    
    if (!success) {
        return {};
    }
    
    return output_data;
}

int64_t ImageGenerator::initializeRNG(int64_t seed) {
    if (seed == -1) {
        std::random_device rd;
        seed = rd();
    }
    
    return seed;
}

std::string ImageGenerator::preprocessPrompt(const std::string& prompt) const {
    // 简单的提示词预处理
    std::string processed = prompt;
    
    // 移除多余的空格
    processed.erase(std::unique(processed.begin(), processed.end(),
                               [](char a, char b) { return a == ' ' && b == ' '; }),
                   processed.end());
    
    // 移除首尾空格
    processed.erase(0, processed.find_first_not_of(" \t\n\r"));
    processed.erase(processed.find_last_not_of(" \t\n\r") + 1);
    
    return processed;
}

sample_method_t ImageGenerator::convertSampler(const std::string& sampler_name) const {
    if (sampler_name == "euler_a") {
        return EULER_A;
    } else if (sampler_name == "euler") {
        return EULER;
    } else if (sampler_name == "heun") {
        return HEUN;
    } else if (sampler_name == "dpm2") {
        return DPM2;
    } else if (sampler_name == "dpm++2s_a") {
        return DPMPP2S_A;
    } else if (sampler_name == "dpm++2m") {
        return DPMPP2M;
    } else if (sampler_name == "dpm++2mv2") {
        return DPMPP2Mv2;
    } else if (sampler_name == "lcm") {
        return LCM;
    } else {
        return EULER_A;  // 默认采样器
    }
}

bool ImageGenerator::validateImageSize(int width, int height) const {
    // 检查尺寸是否为8的倍数（SD模型的要求）
    if (width % 8 != 0 || height % 8 != 0) {
        return false;
    }
    
    // 检查尺寸范围
    if (width < 64 || height < 64 || width > max_size_.first || height > max_size_.second) {
        return false;
    }
    
    // 检查总像素数（防止内存溢出）
    if (static_cast<long long>(width) * height > 2048 * 2048) {
        return false;
    }
    
    return true;
}

ImageGenerationResult ImageGenerator::convertSDImage(sd_image_t* sd_image) const {
    ImageGenerationResult result;
    
    if (!sd_image || !sd_image->data) {
        result.error_message = "Invalid SD image";
        return result;
    }
    
    result.width = sd_image->width;
    result.height = sd_image->height;
    result.channels = sd_image->channel;
    
    size_t data_size = result.width * result.height * result.channels;
    result.image_data.resize(data_size);
    
    std::memcpy(result.image_data.data(), sd_image->data, data_size);
    
    result.success = true;
    
    return result;
}

// ImageGeneratorFactory implementation
std::unique_ptr<ImageGenerator> ImageGeneratorFactory::create(sd_ctx_t* sd_ctx) {
    if (!sd_ctx) {
        return nullptr;
    }
    
    try {
        return std::make_unique<ImageGenerator>(sd_ctx);
    } catch (const std::exception& e) {
        std::cerr << "Failed to create ImageGenerator: " << e.what() << std::endl;
        return nullptr;
    }
}

} // namespace core
} // namespace duorou