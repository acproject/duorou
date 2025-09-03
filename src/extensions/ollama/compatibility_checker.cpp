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
    
    std::cout << "[DEBUG] CompatibilityChecker::checkCompatibility called with: " << model_path << std::endl;
    
    if (!fileExists(model_path)) {
        std::cout << "[DEBUG] Model file not found: " << model_path << std::endl;
        result.errors.push_back("Model file not found: " + model_path);
        return result;
    }
    
    // 检查文件扩展名
    std::string extension = std::filesystem::path(model_path).extension().string();
    std::cout << "[DEBUG] File extension: '" << extension << "'" << std::endl;
    
    // Ollama blob文件没有扩展名，但是GGUF格式，直接尝试作为GGUF处理
    if (extension == ".gguf" || extension.empty()) {
        std::cout << "[DEBUG] Treating as GGUF file" << std::endl;
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
        std::cout << "[DEBUG] Extracted architecture from GGUF: '" << arch_name << "'" << std::endl;
        
        if (arch_name.empty()) {
            result.errors.push_back("Failed to extract architecture from GGUF file");
            return result;
        }
        
        result.detected_arch = detectArchitecture(arch_name);
        result.arch_name = arch_name;
        
        std::cout << "[DEBUG] Detected architecture: " << static_cast<int>(result.detected_arch) << " (" << architectureToString(result.detected_arch) << ")" << std::endl;
        
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
    // 特殊映射：qwen25vl -> qwen2vl
    if (ollama_arch == "qwen25vl") {
        std::cout << "[DEBUG] Mapping qwen25vl to qwen2vl for llama.cpp compatibility" << std::endl;
        return "qwen2vl";
    }
    
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
    qwen_info.aliases = {"qwen2", "qwen2.5", "qwen25vl"};
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
    std::ifstream file(gguf_path, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "[DEBUG] Failed to open GGUF file: " << gguf_path << std::endl;
        return "";
    }
    
    // 读取GGUF头部
    char magic[4];
    file.read(magic, 4);
    
    std::cout << "[DEBUG] GGUF magic: " << std::string(magic, 4) << std::endl;
    
    if (std::string(magic, 4) != "GGUF") {
        std::cout << "[DEBUG] Invalid GGUF magic" << std::endl;
        return "";
    }
    
    // 读取版本号
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    std::cout << "[DEBUG] GGUF version: " << version << std::endl;
    
    // 读取tensor数量
    uint64_t tensor_count;
    file.read(reinterpret_cast<char*>(&tensor_count), sizeof(tensor_count));
    std::cout << "[DEBUG] Tensor count: " << tensor_count << std::endl;
    
    // 读取metadata键值对数量
    uint64_t metadata_kv_count;
    file.read(reinterpret_cast<char*>(&metadata_kv_count), sizeof(metadata_kv_count));
    std::cout << "[DEBUG] Metadata KV count: " << metadata_kv_count << std::endl;
    
    // 解析metadata键值对
    for (uint64_t i = 0; i < metadata_kv_count; ++i) {
        std::cout << "[DEBUG] Processing metadata KV pair " << i << "/" << metadata_kv_count << std::endl;
        
        // 读取key长度
        uint64_t key_len;
        file.read(reinterpret_cast<char*>(&key_len), sizeof(key_len));
        std::cout << "[DEBUG] Key length: " << key_len << std::endl;
        
        // 读取key
        std::string key(key_len, '\0');
        file.read(&key[0], key_len);
        std::cout << "[DEBUG] Key: '" << key << "'" << std::endl;
        
        // 读取value类型
        uint32_t value_type;
        file.read(reinterpret_cast<char*>(&value_type), sizeof(value_type));
        std::cout << "[DEBUG] Value type: " << value_type << std::endl;
        
        // 如果是架构相关的key，读取value
        if (key == "general.architecture") {
            std::cout << "[DEBUG] Found general.architecture key!" << std::endl;
            if (value_type == 8) { // GGUF_TYPE_STRING
                uint64_t value_len;
                file.read(reinterpret_cast<char*>(&value_len), sizeof(value_len));
                std::cout << "[DEBUG] Architecture value length: " << value_len << std::endl;
                
                std::string architecture(value_len, '\0');
                file.read(&architecture[0], value_len);
                std::cout << "[DEBUG] Architecture value: '" << architecture << "'" << std::endl;
                
                return architecture;
            } else {
                std::cout << "[DEBUG] Unexpected value type for architecture: " << value_type << std::endl;
            }
        } else {
            // 跳过其他值
            skipGGUFValue(file, value_type);
        }
    }
    
    return "";
}

std::unordered_map<std::string, std::string> CompatibilityChecker::extractMetadataFromGGUF(const std::string& gguf_path) {
    std::unordered_map<std::string, std::string> metadata;
    
    std::ifstream file(gguf_path, std::ios::binary);
    if (!file.is_open()) {
        return metadata;
    }
    
    // 读取GGUF头部
    char magic[4];
    file.read(magic, 4);
    
    if (std::string(magic, 4) != "GGUF") {
        return metadata;
    }
    
    // 读取版本号
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    
    // 读取tensor数量
    uint64_t tensor_count;
    file.read(reinterpret_cast<char*>(&tensor_count), sizeof(tensor_count));
    
    // 读取metadata键值对数量
    uint64_t metadata_kv_count;
    file.read(reinterpret_cast<char*>(&metadata_kv_count), sizeof(metadata_kv_count));
    
    // 解析metadata键值对
    for (uint64_t i = 0; i < metadata_kv_count; ++i) {
        // 读取key长度
        uint64_t key_len;
        file.read(reinterpret_cast<char*>(&key_len), sizeof(key_len));
        
        // 读取key
        std::string key(key_len, '\0');
        file.read(&key[0], key_len);
        
        // 读取value类型
        uint32_t value_type;
        file.read(reinterpret_cast<char*>(&value_type), sizeof(value_type));
        
        // 读取value并存储
        std::string value = readGGUFValue(file, value_type);
        if (!value.empty()) {
            metadata[key] = value;
        }
    }
    
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

void CompatibilityChecker::skipGGUFValue(std::ifstream& file, uint32_t value_type) {
    switch (value_type) {
        case 0: // GGUF_TYPE_UINT8
            file.seekg(1, std::ios::cur);
            break;
        case 1: // GGUF_TYPE_INT8
            file.seekg(1, std::ios::cur);
            break;
        case 2: // GGUF_TYPE_UINT16
            file.seekg(2, std::ios::cur);
            break;
        case 3: // GGUF_TYPE_INT16
            file.seekg(2, std::ios::cur);
            break;
        case 4: // GGUF_TYPE_UINT32
            file.seekg(4, std::ios::cur);
            break;
        case 5: // GGUF_TYPE_INT32
            file.seekg(4, std::ios::cur);
            break;
        case 6: // GGUF_TYPE_FLOAT32
            file.seekg(4, std::ios::cur);
            break;
        case 7: // GGUF_TYPE_BOOL
            file.seekg(1, std::ios::cur);
            break;
        case 8: // GGUF_TYPE_STRING
            {
                uint64_t len;
                file.read(reinterpret_cast<char*>(&len), sizeof(len));
                file.seekg(len, std::ios::cur);
            }
            break;
        case 9: // GGUF_TYPE_ARRAY
            {
                uint32_t array_type;
                file.read(reinterpret_cast<char*>(&array_type), sizeof(array_type));
                uint64_t array_len;
                file.read(reinterpret_cast<char*>(&array_len), sizeof(array_len));
                for (uint64_t i = 0; i < array_len; ++i) {
                    skipGGUFValue(file, array_type);
                }
            }
            break;
        case 10: // GGUF_TYPE_UINT64
            file.seekg(8, std::ios::cur);
            break;
        case 11: // GGUF_TYPE_INT64
            file.seekg(8, std::ios::cur);
            break;
        case 12: // GGUF_TYPE_FLOAT64
            file.seekg(8, std::ios::cur);
            break;
        default:
            // Unknown type, skip 8 bytes as fallback
            file.seekg(8, std::ios::cur);
            break;
    }
}

std::string CompatibilityChecker::readGGUFValue(std::ifstream& file, uint32_t value_type) {
    switch (value_type) {
        case 0: // GGUF_TYPE_UINT8
            {
                uint8_t val;
                file.read(reinterpret_cast<char*>(&val), sizeof(val));
                return std::to_string(val);
            }
        case 1: // GGUF_TYPE_INT8
            {
                int8_t val;
                file.read(reinterpret_cast<char*>(&val), sizeof(val));
                return std::to_string(val);
            }
        case 2: // GGUF_TYPE_UINT16
            {
                uint16_t val;
                file.read(reinterpret_cast<char*>(&val), sizeof(val));
                return std::to_string(val);
            }
        case 3: // GGUF_TYPE_INT16
            {
                int16_t val;
                file.read(reinterpret_cast<char*>(&val), sizeof(val));
                return std::to_string(val);
            }
        case 4: // GGUF_TYPE_UINT32
            {
                uint32_t val;
                file.read(reinterpret_cast<char*>(&val), sizeof(val));
                return std::to_string(val);
            }
        case 5: // GGUF_TYPE_INT32
            {
                int32_t val;
                file.read(reinterpret_cast<char*>(&val), sizeof(val));
                return std::to_string(val);
            }
        case 6: // GGUF_TYPE_FLOAT32
            {
                float val;
                file.read(reinterpret_cast<char*>(&val), sizeof(val));
                return std::to_string(val);
            }
        case 7: // GGUF_TYPE_BOOL
            {
                uint8_t val;
                file.read(reinterpret_cast<char*>(&val), sizeof(val));
                return val ? "true" : "false";
            }
        case 8: // GGUF_TYPE_STRING
            {
                uint64_t len;
                file.read(reinterpret_cast<char*>(&len), sizeof(len));
                std::string val(len, '\0');
                file.read(&val[0], len);
                return val;
            }
        case 10: // GGUF_TYPE_UINT64
            {
                uint64_t val;
                file.read(reinterpret_cast<char*>(&val), sizeof(val));
                return std::to_string(val);
            }
        case 11: // GGUF_TYPE_INT64
            {
                int64_t val;
                file.read(reinterpret_cast<char*>(&val), sizeof(val));
                return std::to_string(val);
            }
        case 12: // GGUF_TYPE_FLOAT64
            {
                double val;
                file.read(reinterpret_cast<char*>(&val), sizeof(val));
                return std::to_string(val);
            }
        default:
            // For unsupported types like arrays, skip and return empty
            skipGGUFValue(file, value_type);
            return "";
    }
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