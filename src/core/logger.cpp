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
    try {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            // Enable console output by default; file output controlled externally
            console_output_ = true;
            initialized_ = true;
        }
        // Log initialization info (using logger itself)
        info("Logger initialized successfully (console only)");
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Logger: " << e.what() << std::endl;
        return false;
    }
}

void Logger::setLogLevel(LogLevel level) {
    std::string level_str;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        current_level_ = level;
        level_str = getLevelString(level);
    }
    info("Log level set to: " + level_str);
}

bool Logger::setLogFile(const std::string& file_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    try {
        // Close existing file
        if (log_file_ && log_file_->is_open()) {
            log_file_->close();
        }
        // Ensure directory exists
        std::filesystem::path path(file_path);
        std::filesystem::create_directories(path.parent_path());
        // Open new log file
        log_file_ = std::make_unique<std::ofstream>(file_path, std::ios::app);
        if (!log_file_ || !log_file_->is_open()) {
            std::cerr << "Failed to open log file: " << file_path << std::endl;
            file_output_ = false;
            return false;
        }
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
    writeLog(LogLevel::LOAD_ERROR, message);
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
    // Honor current level
    if (level < current_level_) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Build log line
    std::string timestamp = getCurrentTimestamp();
    std::string level_str = getLevelString(level);
    std::ostringstream log_entry;
    log_entry << "[" << timestamp << "] [" << level_str << "] " << message;
    std::string log_line = log_entry.str();
    
    // Console output
    if (console_output_) {
        if (level >= LogLevel::LOAD_ERROR) {
            std::cerr << log_line << std::endl;
        } else {
            std::cout << log_line << std::endl;
        }
    }
    // File output
    if (file_output_ && log_file_ && log_file_->is_open()) {
        *log_file_ << log_line << std::endl;
        if (level >= LogLevel::LOAD_ERROR) {
            log_file_->flush();
        }
    }
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
        case LogLevel::LOAD_ERROR:   return "ERROR";
        case LogLevel::FATAL:   return "FATAL";
        default:                return "UNKNW";
    }
}

std::string Logger::getDefaultLogPath() const {
    // Get log directory
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
    
    // Generate timestamped log filename
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::ostringstream filename;
    filename << "duorou_" << std::put_time(std::localtime(&time_t), "%Y%m%d") << ".log";
    
    return log_dir + "/" + filename.str();
}

} // namespace core
} // namespace duorou