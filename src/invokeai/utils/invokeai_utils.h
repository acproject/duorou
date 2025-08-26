#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>

namespace duorou {
namespace invokeai {

// 图像格式枚举
enum class ImageFormat {
    PNG,
    JPEG,
    WEBP,
    BMP,
    TIFF
};

// 图像信息结构体
struct ImageInfo {
    int width = 0;
    int height = 0;
    int channels = 0;
    ImageFormat format = ImageFormat::PNG;
    size_t file_size = 0;
    std::string color_space;
    bool has_alpha = false;
};

// 系统信息结构体
struct SystemInfo {
    std::string platform;
    std::string cpu_info;
    std::string gpu_info;
    size_t total_memory_mb = 0;
    size_t available_memory_mb = 0;
    size_t gpu_memory_mb = 0;
    bool cuda_available = false;
    bool mps_available = false;
    std::string cuda_version;
    std::vector<std::string> available_devices;
};

// 模型验证结果
struct ModelValidationResult {
    bool is_valid = false;
    std::string model_type; // "checkpoint", "diffusers", "safetensors", "lora", "vae"
    std::string architecture; // "sd1.5", "sd2.0", "sdxl", "controlnet"
    std::string error_message;
    std::map<std::string, std::string> metadata;
    size_t file_size = 0;
    std::string checksum;
};

// 性能统计信息
struct PerformanceStats {
    double total_generation_time = 0.0;
    double model_load_time = 0.0;
    double inference_time = 0.0;
    double postprocess_time = 0.0;
    size_t memory_usage_mb = 0;
    size_t peak_memory_mb = 0;
    int total_generations = 0;
    double average_time_per_step = 0.0;
};

// 图像处理工具类
class ImageUtils {
public:
    // 图像信息获取
    static ImageInfo get_image_info(const std::string& image_path);
    static bool is_valid_image(const std::string& image_path);
    static std::vector<ImageFormat> get_supported_formats();
    
    // 图像转换
    static bool convert_format(const std::string& input_path, 
                              const std::string& output_path, 
                              ImageFormat target_format);
    static bool resize_image(const std::string& input_path, 
                           const std::string& output_path,
                           int target_width, int target_height, 
                           bool maintain_aspect_ratio = true);
    static bool crop_image(const std::string& input_path, 
                         const std::string& output_path,
                         int x, int y, int width, int height);
    
    // 图像处理
    static bool normalize_image(const std::string& input_path, 
                              const std::string& output_path);
    static bool apply_gaussian_blur(const std::string& input_path, 
                                   const std::string& output_path, 
                                   float sigma);
    static bool adjust_brightness(const std::string& input_path, 
                                const std::string& output_path, 
                                float factor);
    static bool adjust_contrast(const std::string& input_path, 
                              const std::string& output_path, 
                              float factor);
    
    // 图像分析
    static std::map<std::string, float> analyze_image_quality(const std::string& image_path);
    static bool detect_faces(const std::string& image_path);
    static std::vector<std::pair<int, int>> get_dominant_colors(const std::string& image_path, int num_colors = 5);
    
    // 图像合成
    static bool blend_images(const std::string& base_path, 
                           const std::string& overlay_path,
                           const std::string& output_path, 
                           float alpha = 0.5f);
    static bool apply_mask(const std::string& image_path, 
                         const std::string& mask_path,
                         const std::string& output_path);
    
    // 批量处理
    static bool batch_resize(const std::vector<std::string>& input_paths,
                           const std::string& output_dir,
                           int target_width, int target_height);
    static bool batch_convert(const std::vector<std::string>& input_paths,
                            const std::string& output_dir,
                            ImageFormat target_format);
};

// 模型工具类
class ModelUtils {
public:
    // 模型验证
    static ModelValidationResult validate_model(const std::string& model_path);
    static bool is_compatible_model(const std::string& model_path, const std::string& architecture);
    static std::string detect_model_architecture(const std::string& model_path);
    
    // 模型信息
    static std::map<std::string, std::string> extract_model_metadata(const std::string& model_path);
    static std::string calculate_model_checksum(const std::string& model_path);
    static size_t get_model_size(const std::string& model_path);
    
    // 模型转换
    static bool convert_checkpoint_to_diffusers(const std::string& checkpoint_path,
                                               const std::string& output_dir);
    static bool convert_diffusers_to_checkpoint(const std::string& diffusers_dir,
                                               const std::string& output_path);
    
    // 模型优化
    static bool optimize_model(const std::string& model_path, 
                             const std::string& output_path,
                             const std::string& precision = "fp16");
    static bool quantize_model(const std::string& model_path,
                             const std::string& output_path,
                             int bits = 8);
    
    // 模型搜索
    static std::vector<std::string> find_models_in_directory(const std::string& directory,
                                                            bool recursive = true);
    static std::vector<std::string> filter_models_by_type(const std::vector<std::string>& model_paths,
                                                         const std::string& model_type);
};

// 系统工具类
class SystemUtils {
public:
    // 系统信息
    static SystemInfo get_system_info();
    static bool is_cuda_available();
    static bool is_mps_available();
    static std::string get_optimal_device();
    
    // 内存管理
    static size_t get_available_memory();
    static size_t get_gpu_memory();
    static bool check_memory_requirements(size_t required_mb);
    static void optimize_memory_usage();
    
    // 性能监控
    static PerformanceStats get_performance_stats();
    static void reset_performance_stats();
    static void start_performance_monitoring();
    static void stop_performance_monitoring();
    
    // 设备管理
    static std::vector<std::string> get_available_devices();
    static bool set_device(const std::string& device);
    static std::string get_current_device();
    
    // 环境检查
    static bool check_dependencies();
    static std::vector<std::string> get_missing_dependencies();
    static bool verify_installation();
};

// 文件工具类
class FileUtils {
public:
    // 文件操作
    static bool file_exists(const std::string& path);
    static bool directory_exists(const std::string& path);
    static bool create_directory(const std::string& path);
    static bool remove_file(const std::string& path);
    static bool remove_directory(const std::string& path, bool recursive = false);
    
    // 文件信息
    static size_t get_file_size(const std::string& path);
    static std::string get_file_extension(const std::string& path);
    static std::string get_filename_without_extension(const std::string& path);
    static std::string get_directory_path(const std::string& path);
    
    // 路径操作
    static std::string join_paths(const std::string& path1, const std::string& path2);
    static std::string normalize_path(const std::string& path);
    static std::string get_absolute_path(const std::string& path);
    static std::string get_relative_path(const std::string& path, const std::string& base);
    
    // 文件搜索
    static std::vector<std::string> find_files(const std::string& directory,
                                              const std::string& pattern,
                                              bool recursive = true);
    static std::vector<std::string> list_directory(const std::string& directory,
                                                  bool include_subdirs = false);
    
    // 文件复制和移动
    static bool copy_file(const std::string& source, const std::string& destination);
    static bool move_file(const std::string& source, const std::string& destination);
    static bool copy_directory(const std::string& source, const std::string& destination);
    
    // 临时文件
    static std::string create_temp_file(const std::string& prefix = "invokeai_",
                                       const std::string& suffix = ".tmp");
    static std::string create_temp_directory(const std::string& prefix = "invokeai_");
    static void cleanup_temp_files();
};

// 配置工具类
class ConfigUtils {
public:
    // 配置文件操作
    static bool load_config(const std::string& config_path, 
                          std::map<std::string, std::string>& config);
    static bool save_config(const std::string& config_path, 
                          const std::map<std::string, std::string>& config);
    
    // 默认配置
    static std::map<std::string, std::string> get_default_config();
    static void apply_default_config(std::map<std::string, std::string>& config);
    
    // 配置验证
    static bool validate_config(const std::map<std::string, std::string>& config);
    static std::vector<std::string> get_config_errors(const std::map<std::string, std::string>& config);
    
    // 环境变量
    static std::string get_env_var(const std::string& name, const std::string& default_value = "");
    static bool set_env_var(const std::string& name, const std::string& value);
    
    // 配置路径
    static std::string get_config_directory();
    static std::string get_models_directory();
    static std::string get_cache_directory();
    static std::string get_output_directory();
};

// 日志工具类
class LogUtils {
public:
    enum class LogLevel {
        DEBUG,
        INFO,
        WARNING,
        ERROR,
        CRITICAL
    };
    
    // 日志记录
    static void log(LogLevel level, const std::string& message);
    static void debug(const std::string& message);
    static void info(const std::string& message);
    static void warning(const std::string& message);
    static void error(const std::string& message);
    static void critical(const std::string& message);
    
    // 日志配置
    static void set_log_level(LogLevel level);
    static void set_log_file(const std::string& file_path);
    static void enable_console_logging(bool enabled);
    static void enable_file_logging(bool enabled);
    
    // 性能日志
    static void log_performance(const std::string& operation, double duration);
    static void log_memory_usage(const std::string& context, size_t memory_mb);
};

// 工具函数
std::string format_file_size(size_t bytes);
std::string format_duration(double seconds);
std::string format_memory_size(size_t bytes);
std::string generate_uuid();
std::string get_timestamp_string();
bool is_valid_uuid(const std::string& uuid);
std::vector<std::string> split_string(const std::string& str, char delimiter);
std::string join_strings(const std::vector<std::string>& strings, const std::string& delimiter);
std::string trim_string(const std::string& str);
std::string to_lower(const std::string& str);
std::string to_upper(const std::string& str);

} // namespace invokeai
} // namespace duorou