#include "session_storage_adapter.h"
#include <algorithm>
#include <arpa/inet.h>
#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <nlohmann/json.hpp>
#include <sstream>
#include <sys/socket.h>
#include <unistd.h>

using json = nlohmann::json;

namespace duorou {
namespace gui {

// 静态常量定义
const std::string SessionStorageAdapter::SESSION_LIST_KEY = "session_list";
const std::string SessionStorageAdapter::SESSION_DATA_PREFIX = "session_data:";

SessionStorageAdapter::SessionStorageAdapter()
    : server_host_("localhost"), server_port_(6379), socket_fd_(-1),
      connected_(false) {}

SessionStorageAdapter::~SessionStorageAdapter() { disconnectFromServer(); }

bool SessionStorageAdapter::initialize(const std::string &server_host,
                                       int server_port) {
  server_host_ = server_host;
  server_port_ = server_port;

  return connectToServer();
}

bool SessionStorageAdapter::connectToServer() {
  if (connected_) {
    return true;
  }

  socket_fd_ = socket(AF_INET, SOCK_STREAM, 0);
  if (socket_fd_ < 0) {
    std::cerr << "Error creating socket" << std::endl;
    return false;
  }

  struct sockaddr_in server_addr;
  memset(&server_addr, 0, sizeof(server_addr));
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(server_port_);

  if (inet_pton(AF_INET, server_host_.c_str(), &server_addr.sin_addr) <= 0) {
    std::cerr << "Invalid address: " << server_host_ << std::endl;
    close(socket_fd_);
    socket_fd_ = -1;
    return false;
  }

  if (connect(socket_fd_, (struct sockaddr *)&server_addr,
              sizeof(server_addr)) < 0) {
    std::cerr << "Connection failed to " << server_host_ << ":" << server_port_
              << std::endl;
    close(socket_fd_);
    socket_fd_ = -1;
    return false;
  }

  connected_ = true;
  std::cout << "Connected to MiniMemory server at " << server_host_ << ":"
            << server_port_ << std::endl;
  return true;
}

void SessionStorageAdapter::disconnectFromServer() {
  if (socket_fd_ >= 0) {
    close(socket_fd_);
    socket_fd_ = -1;
  }
  connected_ = false;
}

bool SessionStorageAdapter::sendCommand(const std::string &command) {
  if (!connected_ || socket_fd_ < 0) {
    if (!connectToServer()) {
      return false;
    }
  }

  ssize_t bytes_sent = send(socket_fd_, command.c_str(), command.length(), 0);
  return bytes_sent == static_cast<ssize_t>(command.length());
}

std::string SessionStorageAdapter::receiveResponse() {
  if (!connected_ || socket_fd_ < 0) {
    return "";
  }

  char buffer[4096];
  ssize_t bytes_received = recv(socket_fd_, buffer, sizeof(buffer) - 1, 0);

  if (bytes_received <= 0) {
    disconnectFromServer();
    return "";
  }

  buffer[bytes_received] = '\0';
  return std::string(buffer);
}

std::string SessionStorageAdapter::buildSetCommand(const std::string &key,
                                                   const std::string &value) {
  return "*3\r\n$3\r\nSET\r\n$" + std::to_string(key.length()) + "\r\n" + key +
         "\r\n$" + std::to_string(value.length()) + "\r\n" + value + "\r\n";
}

std::string SessionStorageAdapter::buildGetCommand(const std::string &key) {
  return "*2\r\n$3\r\nGET\r\n$" + std::to_string(key.length()) + "\r\n" + key +
         "\r\n";
}

std::string SessionStorageAdapter::buildDelCommand(const std::string &key) {
  return "*2\r\n$3\r\nDEL\r\n$" + std::to_string(key.length()) + "\r\n" + key +
         "\r\n";
}

std::string SessionStorageAdapter::buildExistsCommand(const std::string &key) {
  return "*2\r\n$6\r\nEXISTS\r\n$" + std::to_string(key.length()) + "\r\n" +
         key + "\r\n";
}

bool SessionStorageAdapter::saveSession(const ChatSession &session) {
  try {
    // 序列化会话数据
    std::string json_data = serializeSession(session);

    // 保存会话数据
    std::string session_key = getSessionKey(session.get_id());
    std::string set_cmd = buildSetCommand(session_key, json_data);

    if (!sendCommand(set_cmd)) {
      std::cerr << "Failed to send SET command" << std::endl;
      return false;
    }

    std::string response = receiveResponse();
    if (response.find("+OK") != 0) {
      std::cerr << "SET command failed: " << response << std::endl;
      return false;
    }

    // 更新会话列表
    std::vector<std::string> session_ids = getAllSessionIds();
    if (std::find(session_ids.begin(), session_ids.end(), session.get_id()) ==
        session_ids.end()) {
      session_ids.push_back(session.get_id());

      // 将会话ID列表序列化为JSON
      json session_list_json = session_ids;
      std::string list_cmd =
          buildSetCommand(SESSION_LIST_KEY, session_list_json.dump());

      if (!sendCommand(list_cmd)) {
        std::cerr << "Failed to update session list" << std::endl;
        return false;
      }

      receiveResponse(); // 消费响应
    }

    return true;
  } catch (const std::exception &e) {
    std::cerr << "Error saving session: " << e.what() << std::endl;
    return false;
  }
}

std::unique_ptr<ChatSession>
SessionStorageAdapter::loadSession(const std::string &session_id) {
  try {
    std::string session_key = getSessionKey(session_id);
    std::string get_cmd = buildGetCommand(session_key);

    if (!sendCommand(get_cmd)) {
      std::cerr << "Failed to send GET command" << std::endl;
      return nullptr;
    }

    std::string response = receiveResponse();

    // 解析Redis响应格式
    if (response.empty() ||
        response[0] == '$' && response.find("-1") != std::string::npos) {
      return nullptr; // 键不存在
    }

    // 提取实际数据（跳过Redis协议头）
    size_t data_start = response.find("\r\n");
    if (data_start != std::string::npos) {
      data_start += 2;
      std::string json_data = response.substr(data_start);
      // 移除末尾的\r\n
      if (json_data.length() >= 2 &&
          json_data.substr(json_data.length() - 2) == "\r\n") {
        json_data = json_data.substr(0, json_data.length() - 2);
      }
      return deserializeSession(json_data);
    }

    return nullptr;
  } catch (const std::exception &e) {
    std::cerr << "Error loading session: " << e.what() << std::endl;
    return nullptr;
  }
}

bool SessionStorageAdapter::deleteSession(const std::string &session_id) {
  try {
    // 删除会话数据
    std::string session_key = getSessionKey(session_id);
    std::string del_cmd = buildDelCommand(session_key);

    if (!sendCommand(del_cmd)) {
      return false;
    }

    receiveResponse(); // 消费响应

    // 从会话列表中移除
    std::vector<std::string> session_ids = getAllSessionIds();
    auto it = std::find(session_ids.begin(), session_ids.end(), session_id);
    if (it != session_ids.end()) {
      session_ids.erase(it);

      // 更新会话列表
      json session_list_json = session_ids;
      std::string set_cmd =
          buildSetCommand(SESSION_LIST_KEY, session_list_json.dump());

      if (!sendCommand(set_cmd)) {
        return false;
      }

      receiveResponse(); // 消费响应
    }

    return true;
  } catch (const std::exception &e) {
    std::cerr << "Error deleting session: " << e.what() << std::endl;
    return false;
  }
}

std::vector<std::string> SessionStorageAdapter::getAllSessionIds() {
  try {
    std::string get_cmd = buildGetCommand(SESSION_LIST_KEY);

    if (!sendCommand(get_cmd)) {
      return {};
    }

    std::string response = receiveResponse();

    // 解析Redis响应
    if (response.empty() ||
        response[0] == '$' && response.find("-1") != std::string::npos) {
      return {}; // 键不存在，返回空列表
    }

    // 提取JSON数据
    size_t data_start = response.find("\r\n");
    if (data_start != std::string::npos) {
      data_start += 2;
      std::string json_data = response.substr(data_start);
      // 移除末尾的\r\n
      if (json_data.length() >= 2 &&
          json_data.substr(json_data.length() - 2) == "\r\n") {
        json_data = json_data.substr(0, json_data.length() - 2);
      }

      json session_list_json = json::parse(json_data);
      return session_list_json.get<std::vector<std::string>>();
    }

    return {};
  } catch (const std::exception &e) {
    std::cerr << "Error getting session IDs: " << e.what() << std::endl;
    return {};
  }
}

bool SessionStorageAdapter::sessionExists(const std::string &session_id) {
  try {
    std::string session_key = getSessionKey(session_id);
    std::string exists_cmd = buildExistsCommand(session_key);

    if (!sendCommand(exists_cmd)) {
      return false;
    }

    std::string response = receiveResponse();
    return response.find(":1") != std::string::npos;
  } catch (const std::exception &e) {
    std::cerr << "Error checking session existence: " << e.what() << std::endl;
    return false;
  }
}

bool SessionStorageAdapter::saveToFile() {
  // 网络模式下，数据已经持久化到服务器
  return true;
}

bool SessionStorageAdapter::loadFromFile() {
  // 网络模式下，数据从服务器加载
  return true;
}

bool SessionStorageAdapter::clearAllSessions() {
  try {
    std::vector<std::string> session_ids = getAllSessionIds();

    for (const auto &session_id : session_ids) {
      deleteSession(session_id);
    }

    // 清空会话列表
    std::string set_cmd = buildSetCommand(SESSION_LIST_KEY, "[]");
    if (!sendCommand(set_cmd)) {
      return false;
    }

    receiveResponse(); // 消费响应
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Error clearing all sessions: " << e.what() << std::endl;
    return false;
  }
}

size_t SessionStorageAdapter::getSessionCount() {
  return getAllSessionIds().size();
}

std::string
SessionStorageAdapter::serializeSession(const ChatSession &session) {
  json session_json;
  session_json["id"] = session.get_id();
  session_json["title"] = session.get_title();
  session_json["custom_name"] = session.get_custom_name();

  // 转换时间点为时间戳
  auto created_time = session.get_created_time();
  auto last_updated = session.get_last_updated();
  session_json["created_at"] = std::chrono::duration_cast<std::chrono::seconds>(
                                   created_time.time_since_epoch())
                                   .count();
  session_json["last_updated"] =
      std::chrono::duration_cast<std::chrono::seconds>(
          last_updated.time_since_epoch())
          .count();

  json messages_json = json::array();
  for (const auto &message : session.get_messages()) {
    json message_json;
    message_json["content"] = message.content;
    message_json["is_user"] = message.is_user;
    auto msg_timestamp = std::chrono::duration_cast<std::chrono::seconds>(
                             message.timestamp.time_since_epoch())
                             .count();
    message_json["timestamp"] = msg_timestamp;
    messages_json.push_back(message_json);
  }
  session_json["messages"] = messages_json;

  return session_json.dump();
}

std::unique_ptr<ChatSession>
SessionStorageAdapter::deserializeSession(const std::string &json_data) {
  try {
    json session_json = json::parse(json_data);

    // 从JSON中提取基本信息
    std::string id = session_json["id"].get<std::string>();
    std::string title = session_json["title"].get<std::string>();
    
    // 提取自定义名称（向后兼容，如果不存在则为空字符串）
    std::string custom_name = "";
    if (session_json.contains("custom_name")) {
      custom_name = session_json["custom_name"].get<std::string>();
    }

    // 转换时间戳为时间点
    auto created_timestamp = session_json["created_at"].get<int64_t>();
    auto last_updated_timestamp = session_json["last_updated"].get<int64_t>();

    auto created_time =
        std::chrono::system_clock::from_time_t(created_timestamp);
    auto last_updated =
        std::chrono::system_clock::from_time_t(last_updated_timestamp);

    // 使用包含自定义名称的反序列化构造函数创建会话
    auto session =
        std::make_unique<ChatSession>(id, title, custom_name, created_time, last_updated);

    // 反序列化消息列表
    if (session_json.contains("messages") &&
        session_json["messages"].is_array()) {
      for (const auto &msg_json : session_json["messages"]) {
        std::string content = msg_json["content"].get<std::string>();
        bool is_user = msg_json["is_user"].get<bool>();
        session->add_message(content, is_user);
      }
    }

    return session;
  } catch (const std::exception &e) {
    std::cerr << "Error deserializing session: " << e.what() << std::endl;
    return nullptr;
  }
}

std::string
SessionStorageAdapter::getSessionKey(const std::string &session_id) {
  return SESSION_DATA_PREFIX + session_id;
}

} // namespace gui
} // namespace duorou