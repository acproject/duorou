#ifndef DUOROU_EXTENSIONS_OLLAMA_COMPATIBILITY_CHECKER_H
#define DUOROU_EXTENSIONS_OLLAMA_COMPATIBILITY_CHECKER_H

#include <string>
#include <unordered_map>
#include <vector>
#include <functional>

namespace duorou {
namespace extensions {
namespace ollama {

// 支持的模型架构枚举
enum class ModelArchitecture {
    LLAMA,
    MISTRAL,
    GEMMA,
    QWEN,
    PHI,
    CODELLAMA,
    DEEPSEEK,
    UNKNOWN
};

// 架构兼容性信息
struct ArchitectureInfo {
    ModelArchitecture arch;
    std::string ollama_name;     // Ollama中的架构名称
    std::string llama_cpp_name;  // llama.cpp中的架构名称
    std::vector<std::string> aliases;  // 别名列表
    bool supported;              // 是否支持
    std::string version;         // 支持的版本
    std::unordered_map<std::string, std::string> parameter_mapping;  // 参数映射
};

// 模型兼容性检查结果
struct CompatibilityResult {
    bool is_compatible;
    ModelArchitecture detected_arch;
    std::string arch_name;
    std::vector<std::string> warnings;
    std::vector<std::string> errors;
    std::unordered_map<std::string, std::string> required_modifications;
};

// 参数转换规则
struct ParameterConversionRule {
    std::string ollama_param;
    std::string llama_cpp_param;
    std::function<std::string(const std::string&)> converter;
    bool required;
};

// Ollama兼容性检查器
class CompatibilityChecker {
public:
    CompatibilityChecker();
    ~CompatibilityChecker();

    // 架构检测和兼容性检查
    CompatibilityResult checkCompatibility(const std::string& model_path);
    CompatibilityResult checkCompatibilityFromGGUF(const std::string& gguf_path);
    
    // 架构映射
    ModelArchitecture detectArchitecture(const std::string& arch_name);
    std::string mapToLlamaCppArchitecture(const std::string& ollama_arch);
    std::string mapToOllamaArchitecture(const std::string& llama_cpp_arch);
    
    // 参数转换
    std::unordered_map<std::string, std::string> convertParameters(
        const std::unordered_map<std::string, std::string>& ollama_params,
        ModelArchitecture arch);
    
    // 架构信息查询
    const ArchitectureInfo* getArchitectureInfo(ModelArchitecture arch) const;
    const ArchitectureInfo* getArchitectureInfo(const std::string& arch_name) const;
    std::vector<ModelArchitecture> getSupportedArchitectures() const;
    
    // 配置
    void setVerbose(bool verbose) { verbose_ = verbose; }
    void setStrictMode(bool strict) { strict_mode_ = strict; }
    
    // 注册自定义架构
    bool registerArchitecture(const ArchitectureInfo& arch_info);
    bool registerParameterConversion(ModelArchitecture arch, const ParameterConversionRule& rule);

private:
    // 初始化内置架构信息
    void initializeBuiltinArchitectures();
    void initializeParameterConversions();
    
    // GGUF文件分析
    std::string extractArchitectureFromGGUF(const std::string& gguf_path);
    std::unordered_map<std::string, std::string> extractMetadataFromGGUF(const std::string& gguf_path);
    
    // 兼容性检查辅助方法
    bool checkArchitectureSupport(ModelArchitecture arch, CompatibilityResult& result);
    bool checkParameterCompatibility(const std::unordered_map<std::string, std::string>& params,
                                   ModelArchitecture arch, CompatibilityResult& result);
    bool checkVersionCompatibility(const std::string& version, ModelArchitecture arch, CompatibilityResult& result);
    
    // 辅助方法
    std::string normalizeArchitectureName(const std::string& name);
    void log(const std::string& level, const std::string& message);
    bool fileExists(const std::string& path);

private:
    bool verbose_;
    bool strict_mode_;
    
    // 架构信息存储
    std::unordered_map<ModelArchitecture, ArchitectureInfo> architectures_;
    std::unordered_map<std::string, ModelArchitecture> name_to_arch_;
    
    // 参数转换规则
    std::unordered_map<ModelArchitecture, std::vector<ParameterConversionRule>> conversion_rules_;
    
    // 支持的架构列表
    std::vector<ModelArchitecture> supported_architectures_;
};

// 辅助函数
std::string architectureToString(ModelArchitecture arch);
ModelArchitecture stringToArchitecture(const std::string& arch_str);
bool isArchitectureSupported(ModelArchitecture arch);

} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // DUOROU_EXTENSIONS_OLLAMA_COMPATIBILITY_CHECKER_H