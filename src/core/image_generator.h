#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <mutex>
#include "../../third_party/stable-diffusion.cpp/stable-diffusion.h"

namespace duorou {
namespace core {

/**
 * @brief 图像生成参数结构
 */
struct ImageGenerationParams {
    int width = 512;                    ///< 图像宽度
    int height = 512;                   ///< 图像高度
    int steps = 20;                     ///< 采样步数
    float cfg_scale = 7.5f;             ///< CFG引导强度
    int64_t seed = -1;                  ///< 随机种子，-1表示随机
    std::string negative_prompt;        ///< 负面提示词
    float strength = 0.75f;             ///< 图像到图像的强度
    std::string sampler = "euler_a";    ///< 采样器类型
    int clip_skip = -1;                 ///< CLIP跳过层数
    bool vae_tiling = false;            ///< VAE分块处理
    float control_strength = 1.0f;      ///< ControlNet强度
    
    ImageGenerationParams() = default;
};

/**
 * @brief 图像生成结果结构
 */
struct ImageGenerationResult {
    std::vector<uint8_t> image_data;    ///< 图像数据（RGB格式）
    int width;                          ///< 图像宽度
    int height;                         ///< 图像高度
    int channels;                       ///< 通道数
    bool success;                       ///< 是否成功生成
    std::string error_message;          ///< 错误信息
    double generation_time;             ///< 生成时间（秒）
    int64_t seed_used;                  ///< 实际使用的种子
    
    ImageGenerationResult() : width(0), height(0), channels(0), success(false), generation_time(0.0), seed_used(-1) {}
};

/**
 * @brief 生成进度回调函数类型
 * @param step 当前步数
 * @param total_steps 总步数
 * @param progress 进度百分比 (0.0-1.0)
 */
using ProgressCallback = std::function<void(int step, int total_steps, float progress)>;

/**
 * @brief 图像生成器类
 * 
 * 提供高级的图像生成接口，支持文本到图像、图像到图像等功能
 */
class ImageGenerator {
public:
    /**
     * @brief 构造函数
     * @param sd_ctx stable diffusion上下文指针
     */
    explicit ImageGenerator(sd_ctx_t* sd_ctx);
    
    /**
     * @brief 析构函数
     */
    ~ImageGenerator();
    
    /**
     * @brief 文本到图像生成
     * @param prompt 正面提示词
     * @param params 生成参数
     * @return 生成结果
     */
    ImageGenerationResult textToImage(const std::string& prompt, 
                                      const ImageGenerationParams& params = ImageGenerationParams());
    
    /**
     * @brief 带进度回调的文本到图像生成
     * @param prompt 正面提示词
     * @param callback 进度回调函数
     * @param params 生成参数
     * @return 生成结果
     */
    ImageGenerationResult textToImageWithProgress(const std::string& prompt,
                                                  ProgressCallback callback,
                                                  const ImageGenerationParams& params = ImageGenerationParams());
    
    /**
     * @brief 图像到图像生成
     * @param prompt 正面提示词
     * @param input_image 输入图像数据
     * @param input_width 输入图像宽度
     * @param input_height 输入图像高度
     * @param params 生成参数
     * @return 生成结果
     */
    ImageGenerationResult imageToImage(const std::string& prompt,
                                       const std::vector<uint8_t>& input_image,
                                       int input_width,
                                       int input_height,
                                       const ImageGenerationParams& params = ImageGenerationParams());
    
    /**
     * @brief 检查是否可以生成
     * @return 可以生成返回true
     */
    bool canGenerate() const;
    
    /**
     * @brief 获取模型信息
     * @return 模型信息字符串
     */
    std::string getModelInfo() const;
    
    /**
     * @brief 获取支持的最大图像尺寸
     * @return 最大尺寸
     */
    std::pair<int, int> getMaxImageSize() const;
    
    /**
     * @brief 获取推荐的图像尺寸列表
     * @return 尺寸列表
     */
    std::vector<std::pair<int, int>> getRecommendedSizes() const;
    
    /**
     * @brief 验证生成参数
     * @param params 生成参数
     * @return 参数有效返回true
     */
    bool validateParams(const ImageGenerationParams& params) const;
    
    /**
     * @brief 估算生成时间
     * @param params 生成参数
     * @return 估算时间（秒）
     */
    double estimateGenerationTime(const ImageGenerationParams& params) const;
    
    /**
     * @brief 保存图像到文件
     * @param result 生成结果
     * @param file_path 文件路径
     * @param format 图像格式（"png", "jpg", "bmp"）
     * @return 保存成功返回true
     */
    static bool saveImage(const ImageGenerationResult& result, 
                         const std::string& file_path, 
                         const std::string& format = "png");
    
    /**
     * @brief 从文件加载图像
     * @param file_path 文件路径
     * @param width 输出宽度
     * @param height 输出高度
     * @param channels 输出通道数
     * @return 图像数据
     */
    static std::vector<uint8_t> loadImage(const std::string& file_path,
                                          int& width,
                                          int& height,
                                          int& channels);
    
    /**
     * @brief 调整图像尺寸
     * @param image_data 输入图像数据
     * @param input_width 输入宽度
     * @param input_height 输入高度
     * @param output_width 输出宽度
     * @param output_height 输出高度
     * @param channels 通道数
     * @return 调整后的图像数据
     */
    static std::vector<uint8_t> resizeImage(const std::vector<uint8_t>& image_data,
                                            int input_width,
                                            int input_height,
                                            int output_width,
                                            int output_height,
                                            int channels);

private:
    /**
     * @brief 初始化随机数生成器
     * @param seed 随机种子
     * @return 实际使用的种子
     */
    int64_t initializeRNG(int64_t seed);
    
    /**
     * @brief 预处理提示词
     * @param prompt 原始提示词
     * @return 处理后的提示词
     */
    std::string preprocessPrompt(const std::string& prompt) const;
    
    /**
     * @brief 转换采样器名称
     * @param sampler_name 采样器名称
     * @return stable-diffusion.cpp的采样器枚举
     */
    sample_method_t convertSampler(const std::string& sampler_name) const;
    
    /**
     * @brief 验证图像尺寸
     * @param width 宽度
     * @param height 高度
     * @return 尺寸有效返回true
     */
    bool validateImageSize(int width, int height) const;
    
    /**
     * @brief 转换图像格式
     * @param sd_image stable-diffusion图像
     * @return 图像生成结果
     */
    ImageGenerationResult convertSDImage(sd_image_t* sd_image) const;
    
private:
    sd_ctx_t* sd_ctx_;                  ///< stable diffusion上下文指针
    mutable std::mutex mutex_;          ///< 线程安全互斥锁
    
    // 模型信息缓存
    std::string model_info_;            ///< 模型信息
    std::pair<int, int> max_size_;      ///< 最大图像尺寸
    std::vector<std::pair<int, int>> recommended_sizes_;  ///< 推荐尺寸
    
    // 性能统计
    mutable double total_generation_time_;  ///< 总生成时间
    mutable int generation_count_;          ///< 生成次数
};

/**
 * @brief 图像生成器工厂类
 */
class ImageGeneratorFactory {
public:
    /**
     * @brief 创建图像生成器
     * @param sd_ctx stable diffusion上下文指针
     * @return 图像生成器智能指针
     */
    static std::unique_ptr<ImageGenerator> create(sd_ctx_t* sd_ctx);
};

} // namespace core
} // namespace duorou