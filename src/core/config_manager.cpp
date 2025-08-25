#include "config_manager.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>

namespace duorou {
namespace core {

// 简单的JSON解析器（用于基本配置）
class SimpleJsonParser {
public:
    static bool parseValue(const std::string& json, const std::string& key, ConfigValue& value) {
        // 查找键
        std::string search_key = "\"" + key + "\"";
        size_t key_pos = json.find(search_key);
        if (key_pos == std::string::npos) {
            return false;
        }
        
        // 查找冒号
        size_t colon_pos = json.find(':', key_pos);
        if (colon_pos == std::string::npos) {
            return false;
        }
        
        // 跳过空白字符
        size_t value_start = colon_pos + 1;
        while (value_start < json.length() && std::isspace(json[value_start])) {
            value_start++;
        }
        
        if (value_start >= json.length()) {
            return false;
        }
        
        // 解析值
        char first_char = json[value_start];
        
        if (first_char == '"') {
            // 字符串值
            size_t end_quote = json.find('"', value_start + 1);
            if (end_quote == std::string::npos) {
                return false;
            }
            value = json.substr(value_start + 1, end_quote - value_start - 1);
            return true;
        } else if (first_char == 't' || first_char == 'f') {
            // 布尔值
            if (json.substr(value_start, 4) == "true") {
                value = true;
                return true;
            } else if (json.substr(value_start, 5) == "false") {
                value = false;
                return true;
            }
        } else if (std::isdigit(first_char) || first_char == '-') {
            // 数字值
            size_t value_end = value_start;
            bool has_dot = false;
            
            while (value_end < json.length()) {
                char c = json[value_end];
                if (std::isdigit(c) || (c == '-' && value_end == value_start)) {
                    value_end++;
                } else if (c == '.' && !has_dot) {
                    has_dot = true;
                    value_end++;
                } else {
                    break;
                }
            }
            
            std::string number_str = json.substr(value_start, value_end - value_start);
            try {
                if (has_dot) {
                    value = std::stod(number_str);
                } else {
                    value = std::stoi(number_str);
                }
                return true;
            } catch (const std::exception&) {
                return false;
            }
        }
        
        return false;
    }
    
    static std::string generateJson(const std::unordered_map<std::string, ConfigValue>& config_map) {
        std::ostringstream oss;
        oss << "{\n";
        
        bool first = true;
        for (const auto& pair : config_map) {
            if (!first) {
                oss << ",\n";
            }
            first = false;
            
            oss << "  \"" << pair.first << "\": ";
            
            std::visit([&oss](const auto& value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, std::string>) {
                    oss << "\"" << value << "\"";
                } else if constexpr (std::is_same_v<T, bool>) {
                    oss << (value ? "true" : "false");
                } else {
                    oss << value;
                }
            }, pair.second);
        }
        
        oss << "\n}";
        return oss.str();
    }
};

// ConfigManager实现
ConfigManager::ConfigManager()
    : initialized_(false)
    , modified_(false) {
}

ConfigManager::~ConfigManager() {
    if (modified_ && !config_file_path_.empty()) {
        saveConfig();
    }
}

bool ConfigManager::initialize() {
    if (initialized_) {
        return true;
    }
    
    std::cout << "ConfigManager::initialize: Starting initialization" << std::endl;
    std::lock_guard<std::mutex> lock(mutex_);
    std::cout << "ConfigManager::initialize: Acquired mutex lock" << std::endl;
    
    try {
        // 设置默认配置文件路径
        std::cout << "ConfigManager::initialize: Getting default config path" << std::endl;
        config_file_path_ = getDefaultConfigPath();
        std::cout << "ConfigManager::initialize: Config path: " << config_file_path_ << std::endl;
        
        // 创建默认配置
        std::cout << "ConfigManager::initialize: Creating default config" << std::endl;
        createDefaultConfig();
        std::cout << "ConfigManager::initialize: Default config created" << std::endl;
        
        // 尝试加载现有配置文件
        std::cout << "ConfigManager::initialize: Checking if config file exists" << std::endl;
        if (std::filesystem::exists(config_file_path_)) {
            std::cout << "ConfigManager::initialize: Config file exists, loading" << std::endl;
            if (!loadConfig(config_file_path_)) {
                std::cout << "Failed to load existing config, using defaults" << std::endl;
            }
        } else {
            std::cout << "ConfigManager::initialize: Config file doesn't exist, saving default" << std::endl;
            // 保存默认配置
            if (!saveConfigInternal(config_file_path_)) {
                std::cerr << "Failed to save default config" << std::endl;
                return false;
            }
            std::cout << "ConfigManager::initialize: Default config saved" << std::endl;
        }
        
        initialized_ = true;
        modified_ = false;
        
        std::cout << "ConfigManager initialized successfully" << std::endl;
        std::cout << "Config file: " << config_file_path_ << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize ConfigManager: " << e.what() << std::endl;
        return false;
    }
}

bool ConfigManager::loadConfig(const std::string& config_path) {
    try {
        std::ifstream file(config_path);
        if (!file.is_open()) {
            std::cerr << "Cannot open config file: " << config_path << std::endl;
            return false;
        }
        
        // 读取文件内容
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        file.close();
        
        // 解析JSON配置
        if (!parseJsonConfig(content)) {
            std::cerr << "Failed to parse config file: " << config_path << std::endl;
            return false;
        }
        
        config_file_path_ = config_path;
        modified_ = false;
        
        std::cout << "Config loaded from: " << config_path << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading config: " << e.what() << std::endl;
        return false;
    }
}

bool ConfigManager::saveConfig(const std::string& config_path) const {
    std::cout << "saveConfig: Starting with path: " << config_path << std::endl;
    std::lock_guard<std::mutex> lock(mutex_);
    std::cout << "saveConfig: Acquired mutex lock" << std::endl;
    
    return saveConfigInternal(config_path);
}

bool ConfigManager::saveConfigInternal(const std::string& config_path) const {
    std::string path = config_path.empty() ? config_file_path_ : config_path;
    std::cout << "saveConfigInternal: Using path: " << path << std::endl;
    if (path.empty()) {
        std::cerr << "No config file path specified" << std::endl;
        return false;
    }
    
    try {
        std::cout << "saveConfigInternal: Creating directories" << std::endl;
        // 确保目录存在
        std::filesystem::path file_path(path);
        std::filesystem::create_directories(file_path.parent_path());
        std::cout << "saveConfigInternal: Directories created" << std::endl;
        
        std::cout << "saveConfigInternal: Generating JSON content" << std::endl;
        // 生成JSON内容
        std::string json_content = generateJsonConfig();
        std::cout << "saveConfigInternal: JSON content generated" << std::endl;
        
        std::cout << "saveConfigInternal: Opening file for writing" << std::endl;
        // 写入文件
        std::ofstream file(path);
        if (!file.is_open()) {
            std::cerr << "Cannot create config file: " << path << std::endl;
            return false;
        }
        std::cout << "saveConfigInternal: File opened successfully" << std::endl;
        
        std::cout << "saveConfigInternal: Writing content to file" << std::endl;
        file << json_content;
        std::cout << "saveConfigInternal: Content written" << std::endl;
        
        std::cout << "saveConfigInternal: Closing file" << std::endl;
        file.close();
        std::cout << "saveConfigInternal: File closed" << std::endl;
        
        std::cout << "Config saved to: " << path << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error saving config: " << e.what() << std::endl;
        return false;
    }
}

std::string ConfigManager::getString(const std::string& key, const std::string& default_value) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = config_map_.find(key);
    if (it != config_map_.end()) {
        if (std::holds_alternative<std::string>(it->second)) {
            return std::get<std::string>(it->second);
        }
    }
    
    return default_value;
}

int ConfigManager::getInt(const std::string& key, int default_value) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = config_map_.find(key);
    if (it != config_map_.end()) {
        if (std::holds_alternative<int>(it->second)) {
            return std::get<int>(it->second);
        }
    }
    
    return default_value;
}

double ConfigManager::getDouble(const std::string& key, double default_value) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = config_map_.find(key);
    if (it != config_map_.end()) {
        if (std::holds_alternative<double>(it->second)) {
            return std::get<double>(it->second);
        }
    }
    
    return default_value;
}

bool ConfigManager::getBool(const std::string& key, bool default_value) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = config_map_.find(key);
    if (it != config_map_.end()) {
        if (std::holds_alternative<bool>(it->second)) {
            return std::get<bool>(it->second);
        }
    }
    
    return default_value;
}

void ConfigManager::setString(const std::string& key, const std::string& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_map_[key] = value;
    modified_ = true;
}

void ConfigManager::setInt(const std::string& key, int value) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_map_[key] = value;
    modified_ = true;
}

void ConfigManager::setDouble(const std::string& key, double value) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_map_[key] = value;
    modified_ = true;
}

void ConfigManager::setBool(const std::string& key, bool value) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_map_[key] = value;
    modified_ = true;
}

bool ConfigManager::hasKey(const std::string& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_map_.find(key) != config_map_.end();
}

bool ConfigManager::removeKey(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = config_map_.find(key);
    if (it != config_map_.end()) {
        config_map_.erase(it);
        modified_ = true;
        return true;
    }
    
    return false;
}

std::vector<std::string> ConfigManager::getAllKeys() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::string> keys;
    for (const auto& pair : config_map_) {
        keys.push_back(pair.first);
    }
    
    return keys;
}

void ConfigManager::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    config_map_.clear();
    modified_ = true;
}

std::string ConfigManager::getDefaultConfigPath() const {
    // 获取用户配置目录
    std::string config_dir;
    
#ifdef _WIN32
    const char* appdata = std::getenv("APPDATA");
    if (appdata) {
        config_dir = std::string(appdata) + "/Duorou";
    } else {
        config_dir = "./config";
    }
#else
    const char* home = std::getenv("HOME");
    if (home) {
        config_dir = std::string(home) + "/.config/duorou";
    } else {
        config_dir = "./config";
    }
#endif
    
    return config_dir + "/config.json";
}

bool ConfigManager::parseJsonConfig(const std::string& content) {
    // 清空现有配置
    config_map_.clear();
    
    // 解析基本的JSON键值对
    std::istringstream iss(content);
    std::string line;
    
    while (std::getline(iss, line)) {
        // 移除前后空白字符
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);
        
        // 跳过空行和注释
        if (line.empty() || line[0] == '{' || line[0] == '}' || line[0] == '/' || line[0] == '#') {
            continue;
        }
        
        // 查找冒号
        size_t colon_pos = line.find(':');
        if (colon_pos == std::string::npos) {
            continue;
        }
        
        // 提取键
        std::string key_part = line.substr(0, colon_pos);
        key_part.erase(0, key_part.find_first_not_of(" \t"));
        key_part.erase(key_part.find_last_not_of(" \t") + 1);
        
        // 移除引号
        if (key_part.front() == '"' && key_part.back() == '"') {
            key_part = key_part.substr(1, key_part.length() - 2);
        }
        
        // 解析值
        ConfigValue value;
        if (SimpleJsonParser::parseValue(content, key_part, value)) {
            config_map_[key_part] = value;
        }
    }
    
    return true;
}

std::string ConfigManager::generateJsonConfig() const {
    return SimpleJsonParser::generateJson(config_map_);
}

void ConfigManager::createDefaultConfig() {
    // 应用程序基本设置
    config_map_["app.name"] = std::string("Duorou");
    config_map_["app.version"] = std::string("1.0.0");
    config_map_["app.language"] = std::string("zh_CN");
    
    // 日志设置
    config_map_["log.level"] = std::string("INFO");
    config_map_["log.console_output"] = true;
    config_map_["log.file_output"] = true;
    config_map_["log.max_file_size"] = 10; // MB
    
    // 模型设置
    config_map_["model.memory_limit"] = 4096; // MB
    config_map_["model.auto_unload"] = true;
    config_map_["model.default_language_model"] = std::string("");
    config_map_["model.default_diffusion_model"] = std::string("");
    
    // 工作流设置
    config_map_["workflow.worker_threads"] = 0; // 0表示自动检测
    config_map_["workflow.max_queue_size"] = 100;
    config_map_["workflow.task_timeout"] = 300; // 秒
    
    // UI设置
    config_map_["ui.theme"] = std::string("default");
    config_map_["ui.window_width"] = 1200;
    config_map_["ui.window_height"] = 800;
    config_map_["ui.remember_window_state"] = true;
    
    // 性能设置
    config_map_["performance.gpu_acceleration"] = true;
    config_map_["performance.cpu_threads"] = 0; // 0表示自动检测
    config_map_["performance.memory_optimization"] = true;
}

} // namespace core
} // namespace duorou