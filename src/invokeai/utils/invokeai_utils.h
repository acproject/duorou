#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>

namespace duorou {
namespace invokeai {

// Image format enumeration
enum class ImageFormat {
    PNG,    // PNG format
    JPEG,   // JPEG format
    WEBP,   // WebP format
    BMP,
    TIFF
};

// Image data structure
struct ImageInfo {
    int width = 0;          // Image width
    int height = 0;         // Image height
    int channels = 0;       // Number of channels
    ImageFormat format = ImageFormat::PNG; // Image format
    size_t file_size = 0;
    std::string color_space;
    bool has_alpha = false;
};

// System information structure
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

// Model validation result
struct ModelValidationResult {
    bool is_valid = false;
    std::string model_type; // "checkpoint", "diffusers", "safetensors", "lora", "vae"
    std::string architecture; // "sd1.5", "sd2.0", "sdxl", "controlnet"
    std::string error_message;
    std::map<std::string, std::string> metadata;
    size_t file_size = 0;
    std::string checksum;
};

// Performance statistics
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

// Image processing utility class
class ImageUtils {
public:
    // Image information retrieval
    static ImageInfo get_image_info(const std::string& image_path);
    static bool is_valid_image(const std::string& image_path);
    static std::vector<ImageFormat> get_supported_formats();
    
    // Image conversion
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
    
    // Image processing
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
    
    // Image analysis
    static std::map<std::string, float> analyze_image_quality(const std::string& image_path);
    static bool detect_faces(const std::string& image_path);
    static std::vector<std::pair<int, int>> get_dominant_colors(const std::string& image_path, int num_colors = 5);
    
    // Image composition
    static bool blend_images(const std::string& base_path, 
                           const std::string& overlay_path,
                           const std::string& output_path, 
                           float alpha = 0.5f);
    static bool apply_mask(const std::string& image_path, 
                         const std::string& mask_path,
                         const std::string& output_path);
    
    // Batch processing
    static bool batch_resize(const std::vector<std::string>& input_paths,
                           const std::string& output_dir,
                           int target_width, int target_height);
    static bool batch_convert(const std::vector<std::string>& input_paths,
                            const std::string& output_dir,
                            ImageFormat target_format);
};

// Model utility class
class ModelUtils {
public:
    // Model validation
    static ModelValidationResult validate_model(const std::string& model_path);
    static bool is_compatible_model(const std::string& model_path, const std::string& architecture);
    static std::string detect_model_architecture(const std::string& model_path);
    
    // Model information
    static std::map<std::string, std::string> extract_model_metadata(const std::string& model_path);
    static std::string calculate_model_checksum(const std::string& model_path);
    static size_t get_model_size(const std::string& model_path);
    
    // Model conversion
    static bool convert_checkpoint_to_diffusers(const std::string& checkpoint_path,
                                               const std::string& output_dir);
    static bool convert_diffusers_to_checkpoint(const std::string& diffusers_dir,
                                               const std::string& output_path);
    
    // Model optimization
    static bool optimize_model(const std::string& model_path, 
                             const std::string& output_path,
                             const std::string& precision = "fp16");
    static bool quantize_model(const std::string& model_path,
                             const std::string& output_path,
                             int bits = 8);
    
    // Model search
    static std::vector<std::string> find_models_in_directory(const std::string& directory,
                                                            bool recursive = true);
    static std::vector<std::string> filter_models_by_type(const std::vector<std::string>& model_paths,
                                                         const std::string& model_type);
};

// System utility class
class SystemUtils {
public:
    // System information
    static SystemInfo get_system_info();
    static bool is_cuda_available();
    static bool is_mps_available();
    static std::string get_optimal_device();
    
    // Memory management
    static size_t get_available_memory();
    static size_t get_gpu_memory();
    static bool check_memory_requirements(size_t required_mb);
    static void optimize_memory_usage();
    
    // Performance monitoring
    static PerformanceStats get_performance_stats();
    static void reset_performance_stats();
    static void start_performance_monitoring();
    static void stop_performance_monitoring();
    
    // Device management
    static std::vector<std::string> get_available_devices();
    static bool set_device(const std::string& device);
    static std::string get_current_device();
    
    // Environment check
    static bool check_dependencies();
    static std::vector<std::string> get_missing_dependencies();
    static bool verify_installation();
};

// File utility class
class FileUtils {
public:
    // File operations
    static bool file_exists(const std::string& path);
    static bool directory_exists(const std::string& path);
    static bool create_directory(const std::string& path);
    static bool remove_file(const std::string& path);
    static bool remove_directory(const std::string& path, bool recursive = false);
    
    // File information
    static size_t get_file_size(const std::string& path);
    static std::string get_file_extension(const std::string& path);
    static std::string get_filename_without_extension(const std::string& path);
    static std::string get_directory_path(const std::string& path);
    
    // Path operations
    static std::string join_paths(const std::string& path1, const std::string& path2);
    static std::string normalize_path(const std::string& path);
    static std::string get_absolute_path(const std::string& path);
    static std::string get_relative_path(const std::string& path, const std::string& base);
    
    // File search
    static std::vector<std::string> find_files(const std::string& directory,
                                              const std::string& pattern,
                                              bool recursive = true);
    static std::vector<std::string> list_directory(const std::string& directory,
                                                  bool include_subdirs = false);
    
    // File copy and move
    static bool copy_file(const std::string& source, const std::string& destination);
    static bool move_file(const std::string& source, const std::string& destination);
    static bool copy_directory(const std::string& source, const std::string& destination);
    
    // Temporary files
    static std::string create_temp_file(const std::string& prefix = "invokeai_",
                                       const std::string& suffix = ".tmp");
    static std::string create_temp_directory(const std::string& prefix = "invokeai_");
    static void cleanup_temp_files();
};

// Configuration utility class
class ConfigUtils {
public:
    // Configuration file operations
    static bool load_config(const std::string& config_path, 
                          std::map<std::string, std::string>& config);
    static bool save_config(const std::string& config_path, 
                          const std::map<std::string, std::string>& config);
    
    // Default configuration
    static std::map<std::string, std::string> get_default_config();
    static void apply_default_config(std::map<std::string, std::string>& config);
    
    // Configuration validation
    static bool validate_config(const std::map<std::string, std::string>& config);
    static std::vector<std::string> get_config_errors(const std::map<std::string, std::string>& config);
    
    // Environment variables
    static std::string get_env_var(const std::string& name, const std::string& default_value = "");
    static bool set_env_var(const std::string& name, const std::string& value);
    
    // Configuration paths
    static std::string get_config_directory();
    static std::string get_models_directory();
    static std::string get_cache_directory();
    static std::string get_output_directory();
};

// Log utility class
class LogUtils {
public:
    enum class LogLevel {
        DEBUG,
        INFO,
        WARNING,
        ERROR,
        CRITICAL
    };
    
    // Log recording
    static void log(LogLevel level, const std::string& message);
    static void debug(const std::string& message);
    static void info(const std::string& message);
    static void warning(const std::string& message);
    static void error(const std::string& message);
    static void critical(const std::string& message);
    
    // Log configuration
    static void set_log_level(LogLevel level);
    static void set_log_file(const std::string& file_path);
    static void enable_console_logging(bool enabled);
    static void enable_file_logging(bool enabled);
    
    // Performance logging
    static void log_performance(const std::string& operation, double duration);
    static void log_memory_usage(const std::string& context, size_t memory_mb);
};

// Utility functions
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