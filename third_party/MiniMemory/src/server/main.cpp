#include "CommandHandler.hpp"
#include "ConfigParser.hpp"
#include "DataStore.hpp"
#include "Aof.hpp"
#include "TcpServer.hpp"
#include <atomic>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#include <filesystem>
#else
#include <filesystem>
#include <signal.h>
#endif

// 全局变量
TcpServer *g_server = nullptr;
std::atomic<bool> g_running(true);

#ifdef _WIN32
BOOL WINAPI console_ctrl_handler(DWORD ctrl_type) {
    switch (ctrl_type) {
        case CTRL_C_EVENT:
        case CTRL_BREAK_EVENT:
        case CTRL_CLOSE_EVENT:
        case CTRL_LOGOFF_EVENT:
        case CTRL_SHUTDOWN_EVENT:
            std::cout << "\nReceived shutdown signal, gracefully shutting down..." << std::endl;
            g_running = false;
            if (g_server) {
                g_server->stop();
            }
            return TRUE;
        default:
            return FALSE;
    }
}
#else
void signal_handler(int signum) {
    std::cout << "\nReceived signal " << signum << ", gracefully shutting down..." << std::endl;
    g_running = false;
    if (g_server) {
        g_server->stop();
    }
}
#endif

int main(int argc, char *argv[]) {
#ifdef _WIN32
    // 设置 Windows 控制台处理函数
    if (!SetConsoleCtrlHandler(console_ctrl_handler, TRUE)) {
        std::cerr << "Failed to set control handler" << std::endl;
        return 1;
    }
#else
    // 设置信号处理
    signal(SIGINT, signal_handler);  // Ctrl+C
    signal(SIGTERM, signal_handler); // kill 命令
#endif

    // 解析命令行参数
    std::string configPath = "conf/mcs.conf"; // 默认配置文件路径

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            configPath = argv[i + 1];
            ++i; // 跳过配置文件路径
        }
    }

    // 检查配置文件是否存在
#ifdef _WIN32
    DWORD fileAttributes = GetFileAttributesA(configPath.c_str());
    bool configExists = (fileAttributes != INVALID_FILE_ATTRIBUTES && 
                         !(fileAttributes & FILE_ATTRIBUTE_DIRECTORY));
#else
    bool configExists = std::filesystem::exists(configPath);
#endif

    if (!configExists) {
        std::cerr << "The configuration file does not exist: " << configPath << std::endl;
        std::cerr << "Use the default configuration..." << std::endl;
    }

  // 加载配置文件
  ConfigParser config(configPath);

  // 获取配置项
  std::string host = config.getString("bind", "127.0.0.1");
  int port = config.getInt("port", 6379);
  std::string password = config.getString("requirepass", "");

  // 获取内存限制
  std::string memoryStr = config.getString("maxmemory", "0");
  size_t maxMemory = 0;

  if (memoryStr.find("gb") != std::string::npos) {
    maxMemory = std::stoull(memoryStr.substr(0, memoryStr.find("gb"))) * 1024 *
                1024 * 1024;
  } else if (memoryStr.find("mb") != std::string::npos) {
    maxMemory =
        std::stoull(memoryStr.substr(0, memoryStr.find("mb"))) * 1024 * 1024;
  } else if (memoryStr.find("kb") != std::string::npos) {
    maxMemory = std::stoull(memoryStr.substr(0, memoryStr.find("kb"))) * 1024;
  } else {
    try {
      maxMemory = std::stoull(memoryStr);
    } catch (...) {
      std::cerr << "Invalid maxmemory configuration: " << memoryStr << std::endl;
    }
  }

  std::string maxMemoryPolicy =
      config.getString("maxmemory-policy", "noeviction");

  // 获取保存条件
  auto saveConditions = config.getSaveConditions();
  // 是否在每次变更后立即保存快照（写盘），默认关闭
  bool saveImmediate = config.getBool("save_immediate", false);

  // 创建数据存储
  DataStore store;
  CommandHandler handler(store);

  bool appendonly = config.getBool("appendonly", false);
  // 解析数据文件路径为与配置文件同目录的绝对路径，避免工作目录变化导致无法找到文件
  std::string aofFileConf = config.getString("appendfilename", "appendonly.aof");
  std::string aofPathResolvedStr = aofFileConf;
  std::string mcdbPathResolvedStr = "dump.mcdb";
#ifndef _WIN32
  {
    std::filesystem::path configDir = std::filesystem::path(configPath).parent_path();
    std::filesystem::path aofPathResolved = std::filesystem::path(aofFileConf);
    if (!aofPathResolved.is_absolute()) {
      aofPathResolved = configDir / aofPathResolved;
    }
    aofPathResolvedStr = aofPathResolved.string();
    mcdbPathResolvedStr = (configDir / std::filesystem::path("dump.mcdb")).string();
  }
#endif
  std::cout << "Resolved data paths: AOF=" << aofPathResolvedStr
            << ", MCDB=" << mcdbPathResolvedStr << std::endl;
  if (appendonly) {
    // 确保 AOF 目录存在，避免因目录缺失导致 AOF 未写入
    try {
      std::filesystem::path aofPath(aofPathResolvedStr);
      auto aofDir = aofPath.parent_path();
      if (!aofDir.empty() && !std::filesystem::exists(aofDir)) {
        std::filesystem::create_directories(aofDir);
        std::cout << "Created AOF directory: " << aofDir.string() << std::endl;
      }
    } catch (const std::exception &e) {
      std::cerr << "Failed to ensure AOF directory: " << e.what() << std::endl;
    }
    std::cout << "AOF enabled, file: " << aofPathResolvedStr << std::endl;
  }
  bool loaded = false;
  if (appendonly) {
    std::ifstream af(aofPathResolvedStr, std::ios::binary);
    if (af.good()) {
      std::cout << "Found AOF file, attempting to replay..." << std::endl;
      if (AofWriter::replay(aofPathResolvedStr, store)) {
        std::cout << "Successfully loaded AOF file" << std::endl;
        loaded = true;
      } else {
        std::cerr << "Failed to load AOF file" << std::endl;
      }
    }
  }
  if (!loaded) {
    std::ifstream testFile(mcdbPathResolvedStr, std::ios::binary);
    if (testFile.good()) {
      std::cout << "Found MCDB file, attempting to load..." << std::endl;
      if (store.loadMCDB(mcdbPathResolvedStr)) {
        std::cout << "Successfully loaded MCDB file: " << mcdbPathResolvedStr << std::endl;
        loaded = true;
      } else {
        std::cout << "Failed to load MCDB file, starting with empty database"
                  << std::endl;
      }
    } else {
      std::cout << "MCDB file does not exist, starting with empty database"
                << std::endl;
    }
  }

  std::unique_ptr<AofWriter> aof;
  if (appendonly) {
    aof.reset(new AofWriter(aofPathResolvedStr));
    store.setApplyCallback([&aof, &store, saveImmediate, mcdbPathResolvedStr](const std::vector<std::string>& args){
      if (aof) aof->append(args);
      if (saveImmediate) {
        // 可选：在每次变更后立即保存 MCDB（性能开销较大）
        store.saveMCDB(mcdbPathResolvedStr);
      }
    });
    if (saveImmediate) {
      std::cout << "Immediate MCDB save is ENABLED (save_immediate = yes)" << std::endl;
    }
  }

  // 创建服务器
  TcpServer server(configPath);
  g_server = &server; // 设置全局指针

  server.command_handler = [&](const std::vector<std::string> &cmd) {
    // 如果设置了密码，需要验证
    if (!password.empty() && cmd.size() > 0 && cmd[0] != "AUTH") {
      // 检查是否已认证
      // 这里需要添加认证状态的检查逻辑
      // 暂时简化处理
    }
    return handler.handle_command(cmd);
  };

  std::cout << "MiniCache server started on " << host << ":" << port
            << std::endl;
  std::cout << "Press Ctrl+C to gracefully exit" << std::endl;

  // 创建一个后台线程定期清理过期键
  std::thread cleanup_thread([&store]() {
    while (g_running) { // 使用全局变量
      // 每秒清理一次过期键
      std::this_thread::sleep_for(std::chrono::seconds(1));
      store.cleanExpiredKeys();
    }
  });

  // 设置自动保存（基于变更次数 + 时间条件）
  std::thread autosave_thread([&store, &saveConditions, mcdbPathResolvedStr]() {
    std::unordered_map<int, std::chrono::steady_clock::time_point>
        lastSaveTimes;

    for (const auto &condition : saveConditions) {
      lastSaveTimes[condition.first] = std::chrono::steady_clock::now();
    }

    // 动态调整检查间隔：若存在 seconds == 0 的条件，则缩短为 100ms
    int minSeconds = 1000000;
    for (const auto &condition : saveConditions) {
      if (condition.first < minSeconds) minSeconds = condition.first;
    }

    while (g_running) {
      auto now = std::chrono::steady_clock::now();
      // 从数据存储读取自上次检查以来的变更次数，并重置计数
      int changes = store.getAndResetChangeCount();

      // 检查每个保存条件
      for (const auto &condition : saveConditions) {
        int seconds = condition.first;
        int requiredChanges = condition.second;

        // 检查时间条件
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                           now - lastSaveTimes[seconds])
                           .count();

        // 如果时间已到并且有足够的更改，执行保存
        if (elapsed >= seconds && changes >= requiredChanges) {
          std::cout << "Auto-save triggered: " << seconds
                    << " seconds have passed with " << changes << " changes" << std::endl;

          if (store.saveMCDB(mcdbPathResolvedStr)) {
            std::cout << "Auto-save succeeded" << std::endl;
          } else {
            std::cerr << "Auto-save failed" << std::endl;
          }

          // 重置所有计时器，避免短时间内重复触发
          for (auto &pair : lastSaveTimes) {
            pair.second = now;
          }

          // 一旦保存，跳出循环
          break;
        }
      }

      // 若配置存在 seconds == 0 的条件，则缩短为 100ms；否则 1s
      if (minSeconds == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      } else {
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
    }
  });

  // 设置线程为分离状态，让它在后台运行
  cleanup_thread.detach();
  autosave_thread.detach();

  server.start();

  // 在退出前保存数据
  std::cout << "Saving data to MCDB file..." << std::endl;
  store.saveMCDB(mcdbPathResolvedStr);

  std::cout << "Server has been shut down" << std::endl;
  return 0;
}