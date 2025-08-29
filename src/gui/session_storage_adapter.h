#ifndef SESSION_STORAGE_ADAPTER_H
#define SESSION_STORAGE_ADAPTER_H

#include "chat_session.h"
#include <memory>
#include <string>
#include <vector>

namespace duorou {
namespace gui {

/**
 * 会话存储适配器类
 * 封装MiniMemory的DataStore接口，专门用于聊天会话的持久化存储
 */
class SessionStorageAdapter {
public:
  SessionStorageAdapter();
  ~SessionStorageAdapter();

  // 初始化存储连接
  bool initialize(const std::string &server_host = "localhost",
                  int server_port = 6379);

  // 保存单个会话
  bool saveSession(const duorou::gui::ChatSession &session);

  // 加载单个会话
  std::unique_ptr<duorou::gui::ChatSession>
  loadSession(const std::string &session_id);

  // 删除会话
  bool deleteSession(const std::string &session_id);

  // 获取所有会话ID列表
  std::vector<std::string> getAllSessionIds();

  // 检查会话是否存在
  bool sessionExists(const std::string &session_id);

  // 保存所有会话数据到磁盘
  bool saveToFile();

  // 从磁盘加载所有会话数据
  bool loadFromFile();

  // 清空所有会话数据
  bool clearAllSessions();

  // 获取会话数量
  size_t getSessionCount();

private:
  std::string server_host_;
  int server_port_;
  int socket_fd_;
  bool connected_;

  // 序列化会话为JSON字符串
  std::string serializeSession(const duorou::gui::ChatSession &session);

  // 从JSON字符串反序列化会话
  std::unique_ptr<duorou::gui::ChatSession>
  deserializeSession(const std::string &json_data);

  // 生成会话在存储中的键名
  std::string getSessionKey(const std::string &session_id);

  // 会话列表键名
  static const std::string SESSION_LIST_KEY;

  // 会话数据键前缀
  static const std::string SESSION_DATA_PREFIX;

  // 网络通信方法
  bool connectToServer();
  void disconnectFromServer();
  bool sendCommand(const std::string &command);
  std::string receiveResponse();

  // Redis协议命令构建
  std::string buildSetCommand(const std::string &key, const std::string &value);
  std::string buildGetCommand(const std::string &key);
  std::string buildDelCommand(const std::string &key);
  std::string buildExistsCommand(const std::string &key);
};

} // namespace gui
} // namespace duorou

#endif // SESSION_STORAGE_ADAPTER_H