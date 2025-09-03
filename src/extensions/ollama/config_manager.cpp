#include "config_manager.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <regex>
#include <iomanip>

namespace duorou {
namespace extensions {
namespace ollama {

// 静态成员初始化
std::unordered_map<std::string, ConfigValidator> ConfigManager::validators_;

ConfigManager::ConfigManager(bool verbose) : verbose_(verbose) {
    initializeValidators();
}

ConfigManager::~ConfigManager() = default;

bool ConfigManager::registerArchitecture(const ArchitectureConfig& config) {
    if (config.architecture.empty()) {
        log("ERROR", "Architecture name cannot be empty");
        return false;
    }
    
    architectures_[config.architecture] = config;
    
    // 注册配置项验证器
    for (const auto& [key, item] : config.items) {
        if (item.validator) {
            validators_[config.architecture + "." + key] = item.validator;
        }
    }
    
    log("INFO", "Registered architecture: " + config.architecture);
    return true;
}

bool ConfigManager::loadConfig(const std::string& config_path, const std::string& architecture) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        log("ERROR", "Cannot open config file: " + config_path);
        return false;
    }
    
    std::string content((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
    file.close();
    
    return importFromJson(architecture, content);
}

bool ConfigManager::saveConfig(const std::string& config_path, const std::string& architecture) const {
    std::ofstream file(config_path);
    if (!file.is_open()) {
        const_cast<ConfigManager*>(this)->log("ERROR", "Cannot create config file: " + config_path);
        return false;
    }
    
    std::string json_content = exportToJson(architecture);
    file << json_content;
    file.close();
    
    const_cast<ConfigManager*>(this)->log("INFO", "Saved config for " + architecture + " to " + config_path);
    return true;
}

bool ConfigManager::setConfig(const std::string& architecture, const std::string& key, const ConfigValue& value) {
    if (!hasArchitecture(architecture)) {
        log("ERROR", "Architecture not registered: " + architecture);
        return false;
    }
    
    // 验证配置值
    std::string validator_key = architecture + "." + key;
    if (validators_.find(validator_key) != validators_.end()) {
        if (!validators_[validator_key](value)) {
            log("ERROR", "Validation failed for " + validator_key);
            return false;
        }
    }
    
    configs_[architecture][key] = value;
    log("INFO", "Set config " + architecture + "." + key);
    return true;
}

const ConfigValue* ConfigManager::getConfig(const std::string& architecture, const std::string& key) const {
    auto arch_it = configs_.find(architecture);
    if (arch_it == configs_.end()) {
        return nullptr;
    }
    
    auto config_it = arch_it->second.find(key);
    if (config_it == arch_it->second.end()) {
        return nullptr;
    }
    
    return &config_it->second;
}

bool ConfigManager::removeConfig(const std::string& architecture, const std::string& key) {
    auto arch_it = configs_.find(architecture);
    if (arch_it == configs_.end()) {
        return false;
    }
    
    auto config_it = arch_it->second.find(key);
    if (config_it == arch_it->second.end()) {
        return false;
    }
    
    arch_it->second.erase(config_it);
    log("INFO", "Removed config " + architecture + "." + key);
    return true;
}

bool ConfigManager::validateConfig(const std::string& architecture) const {
    if (!hasArchitecture(architecture)) {
        return false;
    }
    
    const auto& arch_config = architectures_.at(architecture);
    auto configs_it = configs_.find(architecture);
    
    if (configs_it == configs_.end()) {
        return true;
    }
    
    const auto& configs = configs_it->second;
    
    // 检查必需的配置项
    for (const auto& [key, item] : arch_config.items) {
        if (item.required && configs.find(key) == configs.end()) {
            const_cast<ConfigManager*>(this)->log("ERROR", "Required config missing: " + architecture + "." + key);
            return false;
        }
    }
    
    return true;
}

std::vector<std::string> ConfigManager::getConfigKeys(const std::string& architecture) const {
    std::vector<std::string> keys;
    auto it = configs_.find(architecture);
    if (it != configs_.end()) {
        for (const auto& [key, value] : it->second) {
            keys.push_back(key);
        }
    }
    return keys;
}

bool ConfigManager::hasArchitecture(const std::string& architecture) const {
    return architectures_.find(architecture) != architectures_.end();
}

std::vector<std::string> ConfigManager::getRegisteredArchitectures() const {
    std::vector<std::string> architectures;
    for (const auto& [arch, config] : architectures_) {
        architectures.push_back(arch);
    }
    return architectures;
}

bool ConfigManager::applyConfigMapping(const std::string& source_arch, const std::string& target_arch) {
    // 简化实现
    return true;
}

bool ConfigManager::mergeConfig(const std::string& architecture, 
                               const std::unordered_map<std::string, ConfigValue>& other_configs,
                               bool overwrite) {
    if (!hasArchitecture(architecture)) {
        log("ERROR", "Architecture not registered: " + architecture);
        return false;
    }
    
    for (const auto& [key, value] : other_configs) {
        if (overwrite || configs_[architecture].find(key) == configs_[architecture].end()) {
            setConfig(architecture, key, value);
        }
    }
    
    return true;
}

bool ConfigManager::resetToDefaults(const std::string& architecture) {
    if (!hasArchitecture(architecture)) {
        return false;
    }
    
    configs_[architecture].clear();
    
    const auto& arch_config = architectures_[architecture];
    for (const auto& [key, item] : arch_config.items) {
        configs_[architecture][key] = item.default_value;
    }
    
    log("INFO", "Reset " + architecture + " to defaults");
    return true;
}

std::string ConfigManager::exportToJson(const std::string& architecture) const {
    auto it = configs_.find(architecture);
    if (it == configs_.end()) {
        return "{}";
    }
    
    std::ostringstream oss;
    oss << "{\n";
    bool first = true;
    for (const auto& [key, value] : it->second) {
        if (!first) oss << ",\n";
        oss << "  \"" << key << "\": " << configValueToJson(value);
        first = false;
    }
    oss << "\n}";
    return oss.str();
}

bool ConfigManager::importFromJson(const std::string& architecture, const std::string& json_str) {
    if (!hasArchitecture(architecture)) {
        log("ERROR", "Architecture not registered: " + architecture);
        return false;
    }
    
    // 简化的JSON解析
    std::regex json_pattern("\"([^\"]+)\"\\s*:\\s*([^,}]+)");
    std::sregex_iterator iter(json_str.begin(), json_str.end(), json_pattern);
    std::sregex_iterator end;
    
    for (; iter != end; ++iter) {
        std::string key = (*iter)[1].str();
        std::string value_str = (*iter)[2].str();
        
        // 移除前后空白
        value_str.erase(0, value_str.find_first_not_of(" \t\n\r"));
        value_str.erase(value_str.find_last_not_of(" \t\n\r") + 1);
        
        try {
            ConfigValue value = parseJsonValue(value_str);
            setConfig(architecture, key, value);
        } catch (const std::exception& e) {
            log("ERROR", "Failed to parse JSON value for key " + key + ": " + e.what());
        }
    }
    
    return true;
}

duorou::extensions::ollama::ArchitectureConfig duorou::extensions::ollama::ConfigManager::createStandardConfig(const std::string& architecture) {
    ArchitectureConfig config(architecture);
    
    if (architecture == "llama") {
        config.items["context_length"] = ConfigItem("context_length", 2048, "Context window size", true);
        config.items["batch_size"] = ConfigItem("batch_size", 512, "Batch size", false);
        config.items["temperature"] = ConfigItem("temperature", 0.7f, "Sampling temperature", false);
        config.required_keys = {"context_length"};
    } else if (architecture == "mistral") {
        config.items["context_length"] = ConfigItem("context_length", 4096, "Context window size", true);
        config.items["sliding_window"] = ConfigItem("sliding_window", 4096, "Sliding window size", false);
        config.required_keys = {"context_length"};
    }
    
    return config;
}

std::unordered_map<std::string, duorou::extensions::ollama::ConfigValidator> duorou::extensions::ollama::ConfigManager::createValidators() {
    std::unordered_map<std::string, ConfigValidator> validators;
    
    validators["*.context_length"] = [](const ConfigValue& value) {
        if (std::holds_alternative<int32_t>(value)) {
            int32_t val = std::get<int32_t>(value);
            return val > 0 && val <= 32768;
        }
        return false;
    };
    
    validators["*.temperature"] = [](const ConfigValue& value) {
        if (std::holds_alternative<float>(value)) {
            float val = std::get<float>(value);
            return val >= 0.0f && val <= 2.0f;
        }
        return false;
    };
    
    return validators;
}

bool duorou::extensions::ollama::ConfigManager::parseConfigFile(const std::string& content, const std::string& architecture) {
    return importFromJson(architecture, content);
}

std::string duorou::extensions::ollama::ConfigManager::generateConfigFile(const std::string& architecture) const {
    return exportToJson(architecture);
}

duorou::extensions::ollama::ConfigValue duorou::extensions::ollama::ConfigManager::parseConfigValue(const std::string& value_str, const duorou::extensions::ollama::ConfigValue& expected_type) const {
    return parseJsonValue(value_str);
}

std::string duorou::extensions::ollama::ConfigManager::configValueToString(const ConfigValue& value) const {
    return configValueToJson(value);
}

duorou::extensions::ollama::ConfigValue duorou::extensions::ollama::ConfigManager::parseJsonValue(const std::string& json_value) const {
    std::string trimmed = json_value;
    trimmed.erase(0, trimmed.find_first_not_of(" \t\n\r"));
    trimmed.erase(trimmed.find_last_not_of(" \t\n\r") + 1);
    
    if (trimmed == "true") {
        return true;
    } else if (trimmed == "false") {
        return false;
    } else if (trimmed.front() == '"' && trimmed.back() == '"') {
        return trimmed.substr(1, trimmed.length() - 2);
    } else if (trimmed.find('.') != std::string::npos) {
        return std::stof(trimmed);
    } else {
        return std::stoi(trimmed);
    }
}

std::string duorou::extensions::ollama::ConfigManager::configValueToJson(const ConfigValue& value) const {
    return std::visit([](const auto& v) -> std::string {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, bool>) {
            return v ? "true" : "false";
        } else if constexpr (std::is_same_v<T, std::string>) {
            return "\"" + v + "\"";
        } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
            std::string result = "[";
            for (size_t i = 0; i < v.size(); ++i) {
                if (i > 0) result += ", ";
                result += "\"" + v[i] + "\"";
            }
            result += "]";
            return result;
        } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(6) << v;
            return oss.str();
        } else {
            return std::to_string(v);
        }
    }, value);
}

std::string duorou::extensions::ollama::ConfigManager::normalizeKey(const std::string& architecture, const std::string& key) const {
    return architecture + "." + key;
}

void duorou::extensions::ollama::ConfigManager::initializeValidators() {
    validators_ = createValidators();
}

void duorou::extensions::ollama::ConfigManager::log(const std::string& level, const std::string& message) {
    if (verbose_) {
        std::cout << "[ConfigManager][" << level << "] " << message << std::endl;
    }
}

} // namespace ollama
} // namespace extensions
} // namespace duorou