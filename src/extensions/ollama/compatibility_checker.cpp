#include "compatibility_checker.h"

#include <fstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <filesystem>

namespace duorou {
namespace extensions {
namespace ollama {

// 辅助函数实现
std::string architectureToString(ModelArchitecture arch) {
    switch (arch) {
        case ModelArchitecture::LLAMA: return "llama";
        case ModelArchitecture::MISTRAL: return "mistral";
        case ModelArchitecture::GEMMA: return "gemma";
        case ModelArchitecture::QWEN: return "qwen";
        case ModelArchitecture::PHI: return "phi";
        case ModelArchitecture::CODELLAMA: return "codellama";
        case ModelArchitecture::DEEPSEEK: return "deepseek";
        default: return "unknown";
    }
}

ModelArchitecture stringToArchitecture(const std::string& arch_str) {
    std::string lower_str = arch_str;
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(), ::tolower);
    
    if (lower_str == "llama" || lower_str == "llama2" || lower_str == "llama3") {
        return ModelArchitecture::LLAMA;
    } else if (lower_str == "mistral") {
        return ModelArchitecture::MISTRAL;
    } else if (lower_str == "gemma" || lower_str == "gemma2") {
        return ModelArchitecture::GEMMA;
    } else if (lower_str == "qwen" || lower_str == "qwen2") {
        return ModelArchitecture::QWEN;
    } else if (lower_str == "phi" || lower_str == "phi3") {
        return ModelArchitecture::PHI;
    } else if (lower_str == "codellama") {
        return ModelArchitecture::CODELLAMA;
    } else if (lower_str == "deepseek") {
        return ModelArchitecture::DEEPSEEK;
    }
    
    return ModelArchitecture::UNKNOWN;
}

bool isArchitectureSupported(ModelArchitecture arch) {
    return arch != ModelArchitecture::UNKNOWN;
}

// CompatibilityChecker 实现
CompatibilityChecker::CompatibilityChecker() : verbose_(false), strict_mode_(false) {
    initializeBuiltinArchitectures();
    initializeParameterConversions();
}

CompatibilityChecker::~CompatibilityChecker() = default;

CompatibilityResult CompatibilityChecker::checkCompatibility(const std::string& model_path) {
    CompatibilityResult result;
    result.is_compatible = false;
    result.detected_arch = ModelArchitecture::UNKNOWN;
    
    if (!fileExists(model_path)) {
        result.errors.push_back("Model file not found: " + model_path);
        return result;
    }
    
    // 检查文件扩展名
    std::string extension = std::filesystem::path(model_path).extension().string();
    if (extension == ".gguf") {
        return checkCompatibilityFromGGUF(model_path);
    }
    
    result.errors.push_back("Unsupported model file format: " + extension);
    return result;
}

CompatibilityResult CompatibilityChecker::checkCompatibilityFromGGUF(const std::string& gguf_path) {
    CompatibilityResult result;
    result.is_compatible = false;
    result.detected_arch = ModelArchitecture::UNKNOWN;
    
    try {
        // 从GGUF文件提取架构信息
        std::string arch_name = extractArchitectureFromGGUF(gguf_path);
        if (arch_name.empty()) {
            result.errors.push_back("Failed to extract architecture from GGUF file");
            return result;
        }
        
        result.detected_arch = detectArchitecture(arch_name);
        result.arch_name = arch_name;
        
        // 检查架构支持
        if (!checkArchitectureSupport(result.detected_arch, result)) {
            return result;
        }
        
        // 提取和检查参数
        auto metadata = extractMetadataFromGGUF(gguf_path);
        if (!checkParameterCompatibility(metadata, result.detected_arch, result)) {
            return result;
        }
        
        // 检查版本兼容性
        auto version_it = metadata.find("version");
        std::string version = (version_it != metadata.end()) ? version_it->second : "";
        if (!checkVersionCompatibility(version, result.detected_arch, result)) {
            return result;
        }
        
        result.is_compatible = true;
        
    } catch (const std::exception& e) {
        result.errors.push_back("Error checking GGUF compatibility: " + std::string(e.what()));
    }
    
    return result;
}

ModelArchitecture CompatibilityChecker::detectArchitecture(const std::string& arch_name) {
    std::string normalized = normalizeArchitectureName(arch_name);
    
    auto it = name_to_arch_.find(normalized);
    if (it != name_to_arch_.end()) {
        return it->second;
    }
    
    // 尝试模糊匹配
    return stringToArchitecture(normalized);
}

std::string CompatibilityChecker::mapToLlamaCppArchitecture(const std::string& ollama_arch) {
    ModelArchitecture arch = detectArchitecture(ollama_arch);
    const ArchitectureInfo* info = getArchitectureInfo(arch);
    
    if (info && !info->llama_cpp_name.empty()) {
        return info->llama_cpp_name;
    }
    
    return ollama_arch; // 如果没有映射，返回原名称
}

std::string CompatibilityChecker::mapToOllamaArchitecture(const std::string& llama_cpp_arch) {
    // 反向查找
    for (const auto& pair : architectures_) {
        const ArchitectureInfo& info = pair.second;
        if (info.llama_cpp_name == llama_cpp_arch) {
            return info.ollama_name;
        }
    }
    
    return llama_cpp_arch; // 如果没有映射，返回原名称
}

std::unordered_map<std::string, std::string> CompatibilityChecker::convertParameters(
    const std::unordered_map<std::string, std::string>& ollama_params,
    ModelArchitecture arch) {
    
    std::unordered_map<std::string, std::string> converted_params;
    
    auto rules_it = conversion_rules_.find(arch);
    if (rules_it == conversion_rules_.end()) {
        // 没有转换规则，直接返回原参数
        return ollama_params;
    }
    
    const auto& rules = rules_it->second;
    
    for (const auto& rule : rules) {
        auto param_it = ollama_params.find(rule.ollama_param);
        if (param_it != ollama_params.end()) {
            std::string converted_value = param_it->second;
            if (rule.converter) {
                converted_value = rule.converter(param_it->second);
            }
            converted_params[rule.llama_cpp_param] = converted_value;
        } else if (rule.required) {
            if (verbose_) {
                log("WARNING", "Required parameter missing: " + rule.ollama_param);
            }
        }
    }
    
    // 复制未转换的参数
    for (const auto& param : ollama_params) {
        bool found = false;
        for (const auto& rule : rules) {
            if (rule.ollama_param == param.first) {
                found = true;
                break;
            }
        }
        if (!found) {
            converted_params[param.first] = param.second;
        }
    }
    
    return converted_params;
}

const ArchitectureInfo* CompatibilityChecker::getArchitectureInfo(ModelArchitecture arch) const {
    auto it = architectures_.find(arch);
    return (it != architectures_.end()) ? &it->second : nullptr;
}

const ArchitectureInfo* CompatibilityChecker::getArchitectureInfo(const std::string& arch_name) const {
    ModelArchitecture arch = const_cast<CompatibilityChecker*>(this)->detectArchitecture(arch_name);
    return getArchitectureInfo(arch);
}

std::vector<ModelArchitecture> CompatibilityChecker::getSupportedArchitectures() const {
    return supported_architectures_;
}

bool CompatibilityChecker::registerArchitecture(const ArchitectureInfo& arch_info) {
    architectures_[arch_info.arch] = arch_info;
    name_to_arch_[normalizeArchitectureName(arch_info.ollama_name)] = arch_info.arch;
    
    // 注册别名
    for (const auto& alias : arch_info.aliases) {
        name_to_arch_[normalizeArchitectureName(alias)] = arch_info.arch;
    }
    
    if (arch_info.supported) {
        auto it = std::find(supported_architectures_.begin(), supported_architectures_.end(), arch_info.arch);
        if (it == supported_architectures_.end()) {
            supported_architectures_.push_back(arch_info.arch);
        }
    }
    
    return true;
}

bool CompatibilityChecker::registerParameterConversion(ModelArchitecture arch, const ParameterConversionRule& rule) {
    conversion_rules_[arch].push_back(rule);
    return true;
}

void CompatibilityChecker::initializeBuiltinArchitectures() {
    // Llama架构
    ArchitectureInfo llama_info;
    llama_info.arch = ModelArchitecture::LLAMA;
    llama_info.ollama_name = "llama";
    llama_info.llama_cpp_name = "llama";
    llama_info.aliases = {"llama2", "llama3", "llama3.1", "llama3.2"};
    llama_info.supported = true;
    llama_info.version = "1.0";
    registerArchitecture(llama_info);
    
    // Mistral架构
    ArchitectureInfo mistral_info;
    mistral_info.arch = ModelArchitecture::MISTRAL;
    mistral_info.ollama_name = "mistral";
    mistral_info.llama_cpp_name = "llama"; // Mistral使用llama架构
    mistral_info.aliases = {"mistral-7b", "mixtral"};
    mistral_info.supported = true;
    mistral_info.version = "1.0";
    registerArchitecture(mistral_info);
    
    // Gemma架构
    ArchitectureInfo gemma_info;
    gemma_info.arch = ModelArchitecture::GEMMA;
    gemma_info.ollama_name = "gemma";
    gemma_info.llama_cpp_name = "gemma";
    gemma_info.aliases = {"gemma2", "gemma-2b", "gemma-7b"};
    gemma_info.supported = true;
    gemma_info.version = "1.0";
    registerArchitecture(gemma_info);
    
    // Qwen架构
    ArchitectureInfo qwen_info;
    qwen_info.arch = ModelArchitecture::QWEN;
    qwen_info.ollama_name = "qwen";
    qwen_info.llama_cpp_name = "qwen2";
    qwen_info.aliases = {"qwen2", "qwen2.5"};
    qwen_info.supported = true;
    qwen_info.version = "1.0";
    registerArchitecture(qwen_info);
    
    // Phi架构
    ArchitectureInfo phi_info;
    phi_info.arch = ModelArchitecture::PHI;
    phi_info.ollama_name = "phi";
    phi_info.llama_cpp_name = "phi3";
    phi_info.aliases = {"phi3", "phi-3"};
    phi_info.supported = true;
    phi_info.version = "1.0";
    registerArchitecture(phi_info);
}

void CompatibilityChecker::initializeParameterConversions() {
    // Llama参数转换规则
    ParameterConversionRule temp_rule;
    temp_rule.ollama_param = "temperature";
    temp_rule.llama_cpp_param = "temp";
    temp_rule.converter = nullptr; // 直接复制
    temp_rule.required = false;
    registerParameterConversion(ModelArchitecture::LLAMA, temp_rule);
    
    ParameterConversionRule top_p_rule;
    top_p_rule.ollama_param = "top_p";
    top_p_rule.llama_cpp_param = "top_p";
    top_p_rule.converter = nullptr;
    top_p_rule.required = false;
    registerParameterConversion(ModelArchitecture::LLAMA, top_p_rule);
    
    // 可以为其他架构添加更多转换规则
}

std::string CompatibilityChecker::extractArchitectureFromGGUF(const std::string& gguf_path) {
    // 简化的GGUF架构提取实现
    // 在实际项目中应该使用专业的GGUF解析库
    
    std::ifstream file(gguf_path, std::ios::binary);
    if (!file.is_open()) {
        return "";
    }
    
    // 读取GGUF头部
    char magic[4];
    file.read(magic, 4);
    
    if (std::string(magic, 4) != "GGUF") {
        return "";
    }
    
    // 这里应该实现完整的GGUF解析
    // 暂时返回一个默认值
    return "llama";
}

std::unordered_map<std::string, std::string> CompatibilityChecker::extractMetadataFromGGUF(const std::string& gguf_path) {
    std::unordered_map<std::string, std::string> metadata;
    
    // 简化的GGUF元数据提取实现
    // 在实际项目中应该使用专业的GGUF解析库
    
    metadata["architecture"] = "llama";
    metadata["version"] = "1.0";
    
    return metadata;
}

bool CompatibilityChecker::checkArchitectureSupport(ModelArchitecture arch, CompatibilityResult& result) {
    const ArchitectureInfo* info = getArchitectureInfo(arch);
    
    if (!info) {
        result.errors.push_back("Unknown architecture: " + architectureToString(arch));
        return false;
    }
    
    if (!info->supported) {
        result.errors.push_back("Unsupported architecture: " + info->ollama_name);
        return false;
    }
    
    return true;
}

bool CompatibilityChecker::checkParameterCompatibility(
    const std::unordered_map<std::string, std::string>& params,
    ModelArchitecture arch, CompatibilityResult& result) {
    
    // 检查必需参数
    auto rules_it = conversion_rules_.find(arch);
    if (rules_it != conversion_rules_.end()) {
        for (const auto& rule : rules_it->second) {
            if (rule.required && params.find(rule.ollama_param) == params.end()) {
                result.warnings.push_back("Missing required parameter: " + rule.ollama_param);
            }
        }
    }
    
    return true;
}

bool CompatibilityChecker::checkVersionCompatibility(
    const std::string& version, ModelArchitecture arch, CompatibilityResult& result) {
    
    const ArchitectureInfo* info = getArchitectureInfo(arch);
    if (!info) {
        return false;
    }
    
    // 简单的版本检查
    if (!version.empty() && version != info->version) {
        result.warnings.push_back("Version mismatch: expected " + info->version + ", got " + version);
    }
    
    return true;
}

std::string CompatibilityChecker::normalizeArchitectureName(const std::string& name) {
    std::string normalized = name;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);
    return normalized;
}

void CompatibilityChecker::log(const std::string& level, const std::string& message) {
    if (verbose_) {
        std::cout << "[" << level << "] CompatibilityChecker: " << message << std::endl;
    }
}

bool CompatibilityChecker::fileExists(const std::string& path) {
    try {
        return std::filesystem::exists(path) && std::filesystem::is_regular_file(path);
    } catch (const std::exception&) {
        return false;
    }
}

} // namespace ollama
} // namespace extensions
} // namespace duorou