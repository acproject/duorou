#pragma once

#include <string>
#include <memory>
#include <fstream>
#include <mutex>
#include <chrono>

namespace duorou {
namespace core {

/**
 * @brief 日志级别枚举
 */
enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3,
    FATAL = 4
};

/**
 * @brief 日志管理器类
 * 
 * 提供线程安全的日志记录功能，支持多种日志级别和输出目标
 */
class Logger {
public:
    /**
     * @brief 构造函数
     */
    Logger();
    
    /**
     * @brief 析构函数
     */
    ~Logger();
    
    /**
     * @brief 初始化日志系统
     * @return 成功返回true，失败返回false
     */
    bool initialize();
    
    /**
     * @brief 设置日志级别
     * @param level 日志级别
     */
    void setLogLevel(LogLevel level);
    
    /**
     * @brief 设置日志文件路径
     * @param file_path 日志文件路径
     * @return 成功返回true，失败返回false
     */
    bool setLogFile(const std::string& file_path);
    
    /**
     * @brief 启用/禁用控制台输出
     * @param enable 是否启用控制台输出
     */
    void setConsoleOutput(bool enable);
    
    /**
     * @brief 记录调试信息
     * @param message 日志消息
     */
    void debug(const std::string& message);
    
    /**
     * @brief 记录一般信息
     * @param message 日志消息
     */
    void info(const std::string& message);
    
    /**
     * @brief 记录警告信息
     * @param message 日志消息
     */
    void warning(const std::string& message);
    
    /**
     * @brief 记录错误信息
     * @param message 日志消息
     */
    void error(const std::string& message);
    
    /**
     * @brief 记录致命错误信息
     * @param message 日志消息
     */
    void fatal(const std::string& message);
    
    /**
     * @brief 刷新日志缓冲区
     */
    void flush();
    
    /**
     * @brief 获取默认日志文件路径
     * @return 默认日志文件路径
     */
    std::string getDefaultLogPath() const;
    
private:
    /**
     * @brief 写入日志
     * @param level 日志级别
     * @param message 日志消息
     */
    void writeLog(LogLevel level, const std::string& message);
    
    /**
     * @brief 获取当前时间戳字符串
     * @return 时间戳字符串
     */
    std::string getCurrentTimestamp() const;
    
    /**
     * @brief 获取日志级别字符串表示
     * @param level 日志级别
     * @return 级别字符串
     */
    std::string getLevelString(LogLevel level) const;
    
private:
    LogLevel current_level_;        ///< 当前日志级别
    bool console_output_;           ///< 是否输出到控制台
    bool file_output_;              ///< 是否输出到文件
    std::string log_file_path_;     ///< 日志文件路径
    std::unique_ptr<std::ofstream> log_file_;  ///< 日志文件流
    mutable std::mutex mutex_;      ///< 线程安全互斥锁
    bool initialized_;              ///< 是否已初始化
};

} // namespace core
} // namespace duorou