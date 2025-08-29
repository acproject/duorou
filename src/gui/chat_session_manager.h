#ifndef DUOROU_GUI_CHAT_SESSION_MANAGER_H
#define DUOROU_GUI_CHAT_SESSION_MANAGER_H

#include "chat_session.h"
#include "session_storage_adapter.h"
#include <functional>
#include <memory>
#include <vector>

namespace duorou {
namespace gui {

/**
 * 聊天会话管理器 - 管理所有聊天会话的创建、切换和持久化
 */
class ChatSessionManager {
public:
  // 会话变更回调函数类型
  using SessionChangeCallback =
      std::function<void(const std::string &session_id)>;
  using SessionListChangeCallback = std::function<void()>;

  ChatSessionManager();
  ~ChatSessionManager() = default;

  // 禁用拷贝构造和赋值
  ChatSessionManager(const ChatSessionManager &) = delete;
  ChatSessionManager &operator=(const ChatSessionManager &) = delete;

  /**
   * 创建新的聊天会话
   * @param title 会话标题（可选）
   * @return 新会话的ID
   */
  std::string create_new_session(const std::string &title = "New Chat");

  /**
   * 切换到指定会话
   * @param session_id 会话ID
   * @return 成功返回true
   */
  bool switch_to_session(const std::string &session_id);

  /**
   * 删除指定会话
   * @param session_id 会话ID
   * @return 成功返回true
   */
  bool delete_session(const std::string &session_id);

  /**
   * 获取当前活动会话
   * @return 当前会话指针，如果没有则返回nullptr
   */
  ChatSession *get_current_session();

  /**
   * 获取指定会话
   * @param session_id 会话ID
   * @return 会话指针，如果不存在则返回nullptr
   */
  ChatSession *get_session(const std::string &session_id);

  /**
   * 获取所有会话列表
   * @return 会话列表
   */
  std::vector<ChatSession *> get_all_sessions();

  /**
   * 获取当前会话ID
   * @return 当前会话ID
   */
  const std::string &get_current_session_id() const {
    return current_session_id_;
  }

  /**
   * 添加消息到当前会话
   * @param message 消息内容
   * @param is_user 是否为用户消息
   * @return 成功返回true
   */
  bool add_message_to_current_session(const std::string &message, bool is_user);

  /**
   * 清空当前会话
   * @return 成功返回true
   */
  bool clear_current_session();

  /**
   * 设置会话变更回调
   * @param callback 回调函数
   */
  void set_session_change_callback(SessionChangeCallback callback) {
    session_change_callback_ = callback;
  }

  /**
   * 设置会话列表变更回调
   * @param callback 回调函数
   */
  void set_session_list_change_callback(SessionListChangeCallback callback) {
    session_list_change_callback_ = callback;
  }

  /**
   * 保存所有会话到存储
   * @return 成功返回true
   */
  bool save_sessions();

  /**
   * 从存储加载会话
   * @return 成功返回true
   */
  bool load_sessions();

  /**
   * 设置会话标题
   * @param session_id 会话ID
   * @param title 新标题
   * @return 成功返回true
   */
  bool set_session_title(const std::string &session_id,
                          const std::string &title);

  /**
   * 设置会话自定义名称
   * @param session_id 会话ID
   * @param custom_name 自定义名称
   * @return 成功返回true
   */
  bool set_session_custom_name(const std::string &session_id,
                                const std::string &custom_name);

  /**
   * 重命名会话（设置自定义名称）
   * @param session_id 会话ID
   * @param new_name 新的自定义名称
   * @return 成功返回true
   */
  bool rename_session(const std::string &session_id,
                      const std::string &new_name);

  /**
   * 获取会话数量
   * @return 会话数量
   */
  size_t get_session_count() const { return sessions_.size(); }

private:
  std::vector<std::unique_ptr<ChatSession>> sessions_;     // 所有会话
  std::string current_session_id_;                         // 当前活动会话ID
  SessionChangeCallback session_change_callback_;          // 会话变更回调
  SessionListChangeCallback session_list_change_callback_; // 会话列表变更回调
  std::unique_ptr<SessionStorageAdapter> storage_adapter_; // 存储适配器

  /**
   * 查找会话索引
   * @param session_id 会话ID
   * @return 会话索引，如果不存在返回-1
   */
  int find_session_index(const std::string &session_id);

  /**
   * 通知会话变更
   */
  void notify_session_change();

  /**
   * 通知会话列表变更
   */
  void notify_session_list_change();
};

} // namespace gui
} // namespace duorou

#endif // DUOROU_GUI_CHAT_SESSION_MANAGER_H