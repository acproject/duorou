#pragma once

#include <string>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <variant>
#include <vector>

namespace duorou {
namespace core {

/**
 * @brief 配置值类型
 */
using ConfigValue = std::variant<std::string, int, double, bool>;

/**
 * @brief 配置管理器类
 * 
 * 负责应用程序配置的加载、保存和管理
 */
class ConfigManager {
public:
    /**
     * @brief 构造函数
     */
    ConfigManager();
    
    /**
     * @brief 析构函数
     */
    ~ConfigManager();
    
    /**
     * @brief 初始化配置管理器
     * @return 成功返回true，失败返回false
     */
    bool initialize();
    
    /**
     * @brief 加载配置文件
     * @param config_path 配置文件路径
     * @return 成功返回true，失败返回false
     */
    bool loadConfig(const std::string& config_path);
    
    /**
     * @brief 保存配置到文件
     * @param config_path 配置文件路径
     * @return 成功返回true，失败返回false
     */
    bool saveConfig(const std::string& config_path = "") const;
    
    /**
     * @brief 获取字符串配置值
     * @param key 配置键
     * @param default_value 默认值
     * @return 配置值
     */
    std::string getString(const std::string& key, const std::string& default_value = "") const;
    
    /**
     * @brief 获取整数配置值
     * @param key 配置键
     * @param default_value 默认值
     * @return 配置值
     */
    int getInt(const std::string& key, int default_value = 0) const;
    
    /**
     * @brief 获取浮点数配置值
     * @param key 配置键
     * @param default_value 默认值
     * @return 配置值
     */
    double getDouble(const std::string& key, double default_value = 0.0) const;
    
    /**
     * @brief 获取布尔配置值
     * @param key 配置键
     * @param default_value 默认值
     * @return 配置值
     */
    bool getBool(const std::string& key, bool default_value = false) const;
    
    /**
     * @brief 设置字符串配置值
     * @param key 配置键
     * @param value 配置值
     */
    void setString(const std::string& key, const std::string& value);
    
    /**
     * @brief 设置整数配置值
     * @param key 配置键
     * @param value 配置值
     */
    void setInt(const std::string& key, int value);
    
    /**
     * @brief 设置浮点数配置值
     * @param key 配置键
     * @param value 配置值
     */
    void setDouble(const std::string& key, double value);
    
    /**
     * @brief 设置布尔配置值
     * @param key 配置键
     * @param value 配置值
     */
    void setBool(const std::string& key, bool value);
    
    /**
     * @brief 检查配置键是否存在
     * @param key 配置键
     * @return 存在返回true，不存在返回false
     */
    bool hasKey(const std::string& key) const;
    
    /**
     * @brief 删除配置项
     * @param key 配置键
     * @return 成功返回true，失败返回false
     */
    bool removeKey(const std::string& key);
    
    /**
     * @brief 获取所有配置键
     * @return 配置键列表
     */
    std::vector<std::string> getAllKeys() const;
    
    /**
     * @brief 清空所有配置
     */
    void clear();
    
    /**
     * @brief 获取默认配置文件路径
     * @return 默认配置文件路径
     */
    std::string getDefaultConfigPath() const;
    
private:
    /**
     * @brief 解析JSON配置文件
     * @param content 文件内容
     * @return 成功返回true，失败返回false
     */
    bool parseJsonConfig(const std::string& content);
    
    /**
     * @brief 生成JSON配置字符串
     * @return JSON配置字符串
     */
    std::string generateJsonConfig() const;
    
    /**
     * @brief 创建默认配置
     */
    void createDefaultConfig();
    
    /**
     * @brief 保存配置到文件（内部方法，不加锁）
     * @param config_path 配置文件路径
     * @return 成功返回true，失败返回false
     */
    bool saveConfigInternal(const std::string& config_path) const;
    
private:
    std::unordered_map<std::string, ConfigValue> config_map_;  ///< 配置映射表
    std::string config_file_path_;                             ///< 配置文件路径
    mutable std::mutex mutex_;                                 ///< 线程安全互斥锁
    bool initialized_;                                         ///< 是否已初始化
    bool modified_;                                            ///< 配置是否已修改
};

} // namespace core
} // namespace duorou