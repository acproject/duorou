#ifndef DUOROU_EXTENSIONS_OLLAMA_CONFIG_MANAGER_H
#define DUOROU_EXTENSIONS_OLLAMA_CONFIG_MANAGER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <variant>
#include <functional>

namespace duorou {
namespace extensions {
namespace ollama {

// 配置值类型
using ConfigValue = std::variant<bool, int32_t, int64_t, float, double, std::string, std::vector<std::string>>;

// 配置验证函数类型
using ConfigValidator = std::function<bool(const ConfigValue&)>;

// 配置项定义
struct ConfigItem {
    std::string key;                    // 配置键名
    ConfigValue default_value;          // 默认值
    std::string description;            // 描述信息
    ConfigValidator validator;          // 验证函数
    bool required;                      // 是否必需
    std::vector<std::string> aliases;   // 别名列表
    
    ConfigItem() = default;
    ConfigItem(const std::string& k, const ConfigValue& def_val, 
               const std::string& desc, bool req = false)
        : key(k), default_value(def_val), description(desc), required(req) {}
};

// 架构特定配置
struct ArchitectureConfig {
    std::string architecture;                           // 架构名称
    std::unordered_map<std::string, ConfigItem> items; // 配置项映射
    std::vector<std::string> required_keys;             // 必需的配置键
    std::unordered_map<std::string, std::string> key_mappings; // 键名映射
    
    ArchitectureConfig() = default;
    explicit ArchitectureConfig(const std::string& arch) : architecture(arch) {}
};

// 配置管理器类
class ConfigManager {
public:
    explicit ConfigManager(bool verbose = false);
    ~ConfigManager();

    /**
     * 注册架构配置
     * @param config 架构配置
     * @return 成功返回true
     */
    bool registerArchitecture(const ArchitectureConfig& config);
    
    /**
     * 加载配置文件
     * @param config_path 配置文件路径
     * @param architecture 目标架构
     * @return 成功返回true
     */
    bool loadConfig(const std::string& config_path, const std::string& architecture);
    
    /**
     * 保存配置文件
     * @param config_path 配置文件路径
     * @param architecture 架构名称
     * @return 成功返回true
     */
    bool saveConfig(const std::string& config_path, const std::string& architecture) const;
    
    /**
     * 设置配置值
     * @param architecture 架构名称
     * @param key 配置键
     * @param value 配置值
     * @return 成功返回true
     */
    bool setConfig(const std::string& architecture, const std::string& key, const ConfigValue& value);
    
    /**
     * 获取配置值
     * @param architecture 架构名称
     * @param key 配置键
     * @return 配置值指针，不存在返回nullptr
     */
    const ConfigValue* getConfig(const std::string& architecture, const std::string& key) const;
    
    /**
     * 获取配置值（带默认值）
     * @param architecture 架构名称
     * @param key 配置键
     * @param default_value 默认值
     * @return 配置值
     */
    template<typename T>
    T getConfigOr(const std::string& architecture, const std::string& key, const T& default_value) const {
        const auto* value = getConfig(architecture, key);
        if (value && std::holds_alternative<T>(*value)) {
            return std::get<T>(*value);
        }
        return default_value;
    }
    
    /**
     * 移除配置项
     * @param architecture 架构名称
     * @param key 配置键
     * @return 成功返回true
     */
    bool removeConfig(const std::string& architecture, const std::string& key);
    
    /**
     * 验证架构配置
     * @param architecture 架构名称
     * @return 验证通过返回true
     */
    bool validateConfig(const std::string& architecture) const;
    
    /**
     * 获取架构的所有配置键
     * @param architecture 架构名称
     * @return 配置键列表
     */
    std::vector<std::string> getConfigKeys(const std::string& architecture) const;
    
    /**
     * 检查架构是否已注册
     * @param architecture 架构名称
     * @return 已注册返回true
     */
    bool hasArchitecture(const std::string& architecture) const;
    
    /**
     * 获取所有已注册的架构
     * @return 架构名称列表
     */
    std::vector<std::string> getRegisteredArchitectures() const;
    
    /**
     * 应用配置映射
     * @param source_arch 源架构
     * @param target_arch 目标架构
     * @return 成功返回true
     */
    bool applyConfigMapping(const std::string& source_arch, const std::string& target_arch);
    
    /**
     * 合并配置
     * @param architecture 架构名称
     * @param other_configs 其他配置映射
     * @param overwrite 是否覆盖现有配置
     * @return 成功返回true
     */
    bool mergeConfig(const std::string& architecture, 
                     const std::unordered_map<std::string, ConfigValue>& other_configs,
                     bool overwrite = false);
    
    /**
     * 重置架构配置为默认值
     * @param architecture 架构名称
     * @return 成功返回true
     */
    bool resetToDefaults(const std::string& architecture);
    
    /**
     * 导出配置为JSON字符串
     * @param architecture 架构名称
     * @return JSON字符串
     */
    std::string exportToJson(const std::string& architecture) const;
    
    /**
     * 从JSON字符串导入配置
     * @param architecture 架构名称
     * @param json_str JSON字符串
     * @return 成功返回true
     */
    bool importFromJson(const std::string& architecture, const std::string& json_str);
    
    /**
     * 设置详细输出模式
     * @param verbose 是否启用详细输出
     */
    void setVerbose(bool verbose) { verbose_ = verbose; }
    
    /**
     * 创建标准架构配置
     * @param architecture 架构名称
     * @return 架构配置
     */
    static ArchitectureConfig createStandardConfig(const std::string& architecture);
    
    /**
     * 创建配置验证器
     * @return 常用验证器映射
     */
    static std::unordered_map<std::string, ConfigValidator> createValidators();

private:
    /**
     * 解析配置文件
     * @param content 文件内容
     * @param architecture 架构名称
     * @return 成功返回true
     */
    bool parseConfigFile(const std::string& content, const std::string& architecture);
    
    /**
     * 生成配置文件内容
     * @param architecture 架构名称
     * @return 配置文件内容
     */
    std::string generateConfigFile(const std::string& architecture) const;
    
    /**
     * 解析配置值
     * @param value_str 值字符串
     * @param expected_type 期望类型
     * @return 解析后的配置值
     */
    ConfigValue parseConfigValue(const std::string& value_str, const ConfigValue& expected_type) const;
    
    /**
     * 配置值转字符串
     * @param value 配置值
     * @return 字符串表示
     */
    std::string configValueToString(const ConfigValue& value) const;
    
    /**
     * 解析JSON值
     * @param json_value JSON值字符串
     * @return 配置值
     */
    ConfigValue parseJsonValue(const std::string& json_value) const;
    
    /**
     * 配置值转JSON
     * @param value 配置值
     * @return JSON字符串
     */
    std::string configValueToJson(const ConfigValue& value) const;
    
    /**
     * 规范化配置键
     * @param architecture 架构名称
     * @param key 原始键名
     * @return 规范化后的键名
     */
    std::string normalizeKey(const std::string& architecture, const std::string& key) const;
    
    /**
     * 初始化验证器
     */
    void initializeValidators();
    
    /**
     * 日志输出
     * @param level 日志级别
     * @param message 日志消息
     */
    void log(const std::string& level, const std::string& message);

private:
    std::unordered_map<std::string, ArchitectureConfig> architectures_;     // 架构配置映射
    std::unordered_map<std::string, std::unordered_map<std::string, ConfigValue>> configs_; // 实际配置值
    bool verbose_;                                                          // 详细输出模式
    
    // 静态验证器
    static std::unordered_map<std::string, ConfigValidator> validators_;
};

// 辅助函数
namespace config_utils {
    /**
     * 检查配置值类型
     * @param value 配置值
     * @param type_name 类型名称
     * @return 类型匹配返回true
     */
    bool checkValueType(const ConfigValue& value, const std::string& type_name);
    
    /**
     * 转换配置值类型
     * @param value 源配置值
     * @param target_type 目标类型示例
     * @return 转换后的配置值
     */
    ConfigValue convertValue(const ConfigValue& value, const ConfigValue& target_type);
    
    /**
     * 创建范围验证器
     * @param min_val 最小值
     * @param max_val 最大值
     * @return 验证器函数
     */
    template<typename T>
    ConfigValidator createRangeValidator(const T& min_val, const T& max_val) {
        return [min_val, max_val](const ConfigValue& value) -> bool {
            if (std::holds_alternative<T>(value)) {
                T val = std::get<T>(value);
                return val >= min_val && val <= max_val;
            }
            return false;
        };
    }
    
    /**
     * 创建枚举验证器
     * @param valid_values 有效值列表
     * @return 验证器函数
     */
    template<typename T>
    ConfigValidator createEnumValidator(const std::vector<T>& valid_values) {
        return [valid_values](const ConfigValue& value) -> bool {
            if (std::holds_alternative<T>(value)) {
                T val = std::get<T>(value);
                return std::find(valid_values.begin(), valid_values.end(), val) != valid_values.end();
            }
            return false;
        };
    }
}

} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // DUOROU_EXTENSIONS_OLLAMA_CONFIG_MANAGER_H