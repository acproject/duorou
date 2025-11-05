#pragma once

#include <string>
#include <memory>
#include <fstream>
#include <mutex>
#include <chrono>

namespace duorou {
namespace core {

/**
 * @brief Log level enumeration
 */
enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    LOAD_ERROR = 3,
    FATAL = 4
};

/**
 * @brief Logger manager class
 * 
 * Provides thread-safe logging functionality with support for multiple log levels and output targets
 */
class Logger {
public:
    /**
     * @brief Constructor
     */
    Logger();
    
    /**
     * @brief Destructor
     */
    ~Logger();
    
    /**
     * @brief Initialize logging system
     * @return Returns true on success, false on failure
     */
    bool initialize();
    
    /**
     * @brief Set log level
     * @param level Log level
     */
    void setLogLevel(LogLevel level);
    
    /**
     * @brief Set log file path
     * @param file_path Log file path
     * @return Returns true on success, false on failure
     */
    bool setLogFile(const std::string& file_path);
    
    /**
     * @brief Enable/disable console output
     * @param enable Whether to enable console output
     */
    void setConsoleOutput(bool enable);
    
    /**
     * @brief Log debug information
     * @param message Log message
     */
    void debug(const std::string& message);
    
    /**
     * @brief Log general information
     * @param message Log message
     */
    void info(const std::string& message);
    
    /**
     * @brief Log warning information
     * @param message Log message
     */
    void warning(const std::string& message);
    
    /**
     * @brief Log error information
     * @param message Log message
     */
    void error(const std::string& message);
    
    /**
     * @brief Log fatal error information
     * @param message Log message
     */
    void fatal(const std::string& message);
    
    /**
     * @brief Flush log buffer
     */
    void flush();
    
    /**
     * @brief Get default log file path
     * @return Default log file path
     */
    std::string getDefaultLogPath() const;
    
private:
    /**
     * @brief Write log
     * @param level Log level
     * @param message Log message
     */
    void writeLog(LogLevel level, const std::string& message);
    
    /**
     * @brief Get current timestamp string
     * @return Timestamp string
     */
    std::string getCurrentTimestamp() const;
    
    /**
     * @brief Get log level string representation
     * @param level Log level
     * @return Level string
     */
    std::string getLevelString(LogLevel level) const;
    
private:
    LogLevel current_level_;        ///< Current log level
    bool console_output_;           ///< Whether to output to console
    bool file_output_;              ///< Whether to output to file
    std::string log_file_path_;     ///< Log file path
    std::unique_ptr<std::ofstream> log_file_;  ///< Log file stream
    mutable std::mutex mutex_;      ///< Thread-safe mutex
    bool initialized_;              ///< Whether initialized
};

} // namespace core
} // namespace duorou