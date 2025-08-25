#include "logger.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <ctime>

namespace duorou {
namespace core {

Logger::Logger()
    : current_level_(LogLevel::INFO)
    , console_output_(true)
    , file_output_(false)
    , initialized_(false) {
}

Logger::~Logger() {
    flush();
    if (log_file_ && log_file_->is_open()) {
        log_file_->close();
    }
}

bool Logger::initialize() {
    if (initialized_) {
        return true;
    }
    
    std::cout << "Logger::initialize: Starting initialization" << std::endl;
    
    try {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            std::cout << "Logger::initialize: Acquired mutex lock" << std::endl;
            
            // 默认启用控制台输出，文件输出将在配置加载后设置
            std::cout << "Logger::initialize: Setting default output options" << std::endl;
            console_output_ = true;
            // file_output_ 保持默认值 false，等待配置管理器设置
            
            initialized_ = true;
            std::cout << "Logger::initialize: Set initialized flag to true" << std::endl;
        } // mutex锁在这里释放
        
        // 记录初始化信息 - 在锁释放后调用
        std::cout << "Logger::initialize: Calling info method" << std::endl;
        info("Logger initialized successfully (console only)");
        
        std::cout << "Logger::initialize: Completed successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Logger: " << e.what() << std::endl;
        return false;
    }
}

void Logger::setLogLevel(LogLevel level) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_level_ = level;
    
    std::string level_str = getLevelString(level);
    info("Log level set to: " + level_str);
}

bool Logger::setLogFile(const std::string& file_path) {
    std::cout << "setLogFile: Starting with path: " << file_path << std::endl;
    std::lock_guard<std::mutex> lock(mutex_);
    
    try {
        std::cout << "setLogFile: Acquired mutex lock" << std::endl;
        
        // 关闭现有文件
        if (log_file_ && log_file_->is_open()) {
            std::cout << "setLogFile: Closing existing log file" << std::endl;
            log_file_->close();
        }
        
        // 确保目录存在
        std::cout << "setLogFile: Creating directory path" << std::endl;
        std::filesystem::path path(file_path);
        std::cout << "setLogFile: Parent path: " << path.parent_path() << std::endl;
        std::filesystem::create_directories(path.parent_path());
        std::cout << "setLogFile: Directory created successfully" << std::endl;
        
        // 创建新的文件流
        std::cout << "setLogFile: Opening log file" << std::endl;
        log_file_ = std::make_unique<std::ofstream>(file_path, std::ios::app);
        if (!log_file_->is_open()) {
            std::cerr << "Failed to open log file: " << file_path << std::endl;
            file_output_ = false;
            return false;
        }
        
        std::cout << "setLogFile: Log file opened successfully" << std::endl;
        log_file_path_ = file_path;
        file_output_ = true;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error setting log file: " << e.what() << std::endl;
        file_output_ = false;
        return false;
    }
}

void Logger::setConsoleOutput(bool enable) {
    std::lock_guard<std::mutex> lock(mutex_);
    console_output_ = enable;
    
    if (enable) {
        info("Console output enabled");
    } else {
        info("Console output disabled");
    }
}

void Logger::debug(const std::string& message) {
    writeLog(LogLevel::DEBUG, message);
}

void Logger::info(const std::string& message) {
    writeLog(LogLevel::INFO, message);
}

void Logger::warning(const std::string& message) {
    writeLog(LogLevel::WARNING, message);
}

void Logger::error(const std::string& message) {
    writeLog(LogLevel::ERROR, message);
}

void Logger::fatal(const std::string& message) {
    writeLog(LogLevel::FATAL, message);
}

void Logger::flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (console_output_) {
        std::cout.flush();
        std::cerr.flush();
    }
    
    if (file_output_ && log_file_ && log_file_->is_open()) {
        log_file_->flush();
    }
}

void Logger::writeLog(LogLevel level, const std::string& message) {
    std::cout << "writeLog: Starting with message: " << message << std::endl;
    
    // 检查日志级别
    if (level < current_level_) {
        std::cout << "writeLog: Level check failed, returning" << std::endl;
        return;
    }
    
    std::cout << "writeLog: Attempting to acquire mutex" << std::endl;
    std::lock_guard<std::mutex> lock(mutex_);
    std::cout << "writeLog: Mutex acquired" << std::endl;
    
    // 生成日志条目
    std::cout << "writeLog: Getting timestamp" << std::endl;
    std::string timestamp = getCurrentTimestamp();
    std::cout << "writeLog: Getting level string" << std::endl;
    std::string level_str = getLevelString(level);
    
    std::cout << "writeLog: Building log entry" << std::endl;
    std::ostringstream log_entry;
    log_entry << "[" << timestamp << "] [" << level_str << "] " << message;
    
    std::string log_line = log_entry.str();
    std::cout << "writeLog: Log line built: " << log_line << std::endl;
    
    // 输出到控制台
    if (console_output_) {
        std::cout << "writeLog: Outputting to console" << std::endl;
        if (level >= LogLevel::ERROR) {
            std::cerr << log_line << std::endl;
        } else {
            std::cout << log_line << std::endl;
        }
        std::cout << "writeLog: Console output completed" << std::endl;
    }
    
    // 输出到文件
    if (file_output_ && log_file_ && log_file_->is_open()) {
        std::cout << "writeLog: Outputting to file" << std::endl;
        *log_file_ << log_line << std::endl;
        
        // 对于错误和致命错误，立即刷新
        if (level >= LogLevel::ERROR) {
            log_file_->flush();
        }
        std::cout << "writeLog: File output completed" << std::endl;
    }
    
    std::cout << "writeLog: Method completed" << std::endl;
}

std::string Logger::getCurrentTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    oss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    
    return oss.str();
}

std::string Logger::getLevelString(LogLevel level) const {
    switch (level) {
        case LogLevel::DEBUG:   return "DEBUG";
        case LogLevel::INFO:    return "INFO ";
        case LogLevel::WARNING: return "WARN ";
        case LogLevel::ERROR:   return "ERROR";
        case LogLevel::FATAL:   return "FATAL";
        default:                return "UNKNW";
    }
}

std::string Logger::getDefaultLogPath() const {
    // 获取日志目录
    std::string log_dir;
    
#ifdef _WIN32
    const char* appdata = std::getenv("APPDATA");
    if (appdata) {
        log_dir = std::string(appdata) + "/Duorou/logs";
    } else {
        log_dir = "./logs";
    }
#else
    const char* home = std::getenv("HOME");
    if (home) {
        log_dir = std::string(home) + "/.local/share/duorou/logs";
    } else {
        log_dir = "./logs";
    }
#endif
    
    // 生成带时间戳的日志文件名
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::ostringstream filename;
    filename << "duorou_" << std::put_time(std::localtime(&time_t), "%Y%m%d") << ".log";
    
    return log_dir + "/" + filename.str();
}

} // namespace core
} // namespace duorou