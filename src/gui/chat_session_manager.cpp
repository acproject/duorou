#include "chat_session_manager.h"
#include "../core/logger.h"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <thread>

namespace duorou {
namespace gui {

ChatSessionManager::ChatSessionManager()
    : storage_adapter_(std::make_unique<SessionStorageAdapter>()) {
  // 初始化存储适配器
  storage_adapter_->initialize("127.0.0.1", 6379);

  // 尝试从存储加载会话
  if (!load_sessions()) {
    // 如果加载失败或没有会话，创建默认会话
    create_new_session("Welcome Chat");
  }
}

std::string ChatSessionManager::create_new_session(const std::string &title) {
  auto session = std::make_unique<ChatSession>(title);
  std::string session_id = session->get_id();

  // 保存到存储
  storage_adapter_->saveSession(*session);

  sessions_.push_back(std::move(session));

  // 切换到新会话
  current_session_id_ = session_id;

  // 保存到磁盘
  storage_adapter_->saveToFile();

  notify_session_list_change();
  notify_session_change();

  return session_id;
}

bool ChatSessionManager::switch_to_session(const std::string &session_id) {
  int index = find_session_index(session_id);
  if (index == -1) {
    return false;
  }

  current_session_id_ = session_id;
  notify_session_change();
  return true;
}

bool ChatSessionManager::delete_session(const std::string &session_id) {
  int index = find_session_index(session_id);
  if (index == -1) {
    return false;
  }

  // 从存储中删除
  storage_adapter_->deleteSession(session_id);

  // 如果删除的是当前会话，需要切换到其他会话
  bool was_current = (session_id == current_session_id_);

  sessions_.erase(sessions_.begin() + index);

  if (was_current) {
    if (!sessions_.empty()) {
      // 切换到第一个会话
      current_session_id_ = sessions_[0]->get_id();
      notify_session_change();
    } else {
      // 如果没有会话了，创建一个新的
      create_new_session("New Chat");
    }
  }

  // 保存到磁盘
  storage_adapter_->saveToFile();

  notify_session_list_change();
  return true;
}

ChatSession *ChatSessionManager::get_current_session() {
  return get_session(current_session_id_);
}

ChatSession *ChatSessionManager::get_session(const std::string &session_id) {
  int index = find_session_index(session_id);
  if (index == -1) {
    return nullptr;
  }
  return sessions_[index].get();
}

std::vector<ChatSession *> ChatSessionManager::get_all_sessions() {
  std::vector<ChatSession *> result;
  for (auto &session : sessions_) {
    result.push_back(session.get());
  }

  // 按最后更新时间排序（最新的在前）
  std::sort(result.begin(), result.end(), [](ChatSession *a, ChatSession *b) {
    return a->get_last_updated() > b->get_last_updated();
  });

  return result;
}

bool ChatSessionManager::add_message_to_current_session(
    const std::string &message, bool is_user) {
  ChatSession *current = get_current_session();
  if (!current) {
    return false;
  }

  current->add_message(message, is_user);

  // 立即通知 UI（更新会话标题等）；持久化改为后台线程，避免阻塞主线程
  notify_session_list_change();

  // 将持久化移至后台线程，保护存储操作避免并发竞争
  std::string sid = current->get_id();
  std::thread([this, sid]() {
    try {
      std::lock_guard<std::mutex> lock(storage_mutex_);
      ChatSession *session = get_session(sid);
      if (!session) return;
      storage_adapter_->saveSession(*session);
      storage_adapter_->saveToFile();
    } catch (const std::exception &e) {
      std::cerr << "Async save session error: " << e.what() << std::endl;
    }
  }).detach();

  return true;
}

bool ChatSessionManager::clear_current_session() {
  ChatSession *current = get_current_session();
  if (!current) {
    return false;
  }

  current->clear_messages();
  return true;
}

bool ChatSessionManager::set_session_title(const std::string &session_id,
                                           const std::string &new_title) {
  auto it =
      std::find_if(sessions_.begin(), sessions_.end(),
                   [&session_id](const std::unique_ptr<ChatSession> &session) {
                     return session->get_id() == session_id;
                   });

  if (it != sessions_.end()) {
    (*it)->set_title(new_title);
    // 保存更新的会话
    storage_adapter_->saveSession(**it);
    storage_adapter_->saveToFile();
    notify_session_list_change();
    return true;
  }
  return false;
}

bool ChatSessionManager::set_session_custom_name(const std::string &session_id,
                                                 const std::string &custom_name) {
  auto it =
      std::find_if(sessions_.begin(), sessions_.end(),
                   [&session_id](const std::unique_ptr<ChatSession> &session) {
                     return session->get_id() == session_id;
                   });

  if (it != sessions_.end()) {
    (*it)->set_custom_name(custom_name);
    // 保存更新的会话
    storage_adapter_->saveSession(**it);
    storage_adapter_->saveToFile();
    notify_session_list_change();
    return true;
  }
  return false;
}

bool ChatSessionManager::rename_session(const std::string &session_id,
                                       const std::string &new_name) {
  return set_session_custom_name(session_id, new_name);
}

bool ChatSessionManager::save_sessions() {
  try {
    // 保存所有会话到存储
    for (const auto &session : sessions_) {
      storage_adapter_->saveSession(*session);
    }

    // 保存到磁盘
    return storage_adapter_->saveToFile();
  } catch (const std::exception &e) {
    std::cerr << "Error saving sessions: " << e.what() << std::endl;
    return false;
  }
}

bool ChatSessionManager::load_sessions() {
  try {
    // 从存储加载数据
    if (!storage_adapter_->loadFromFile()) {
      return false;
    }

    // 获取所有会话ID
    std::vector<std::string> session_ids = storage_adapter_->getAllSessionIds();

    // 清空现有会话
    sessions_.clear();
    current_session_id_.clear();

    // 加载每个会话
    for (const auto &session_id : session_ids) {
      auto session = storage_adapter_->loadSession(session_id);
      if (session) {
        sessions_.push_back(std::move(session));
      }
    }

    // 如果没有会话，返回false让调用者创建默认会话
    if (sessions_.empty()) {
      return false;
    }

    // 设置第一个会话为当前会话
    current_session_id_ = sessions_[0]->get_id();

    notify_session_list_change();
    notify_session_change();

    return true;
  } catch (const std::exception &e) {
    std::cerr << "Error loading sessions: " << e.what() << std::endl;
    return false;
  }
}

int ChatSessionManager::find_session_index(const std::string &session_id) {
  for (size_t i = 0; i < sessions_.size(); ++i) {
    if (sessions_[i]->get_id() == session_id) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

void ChatSessionManager::notify_session_change() {
  if (session_change_callback_) {
    session_change_callback_(current_session_id_);
  }
}

void ChatSessionManager::notify_session_list_change() {
  if (session_list_change_callback_) {
    session_list_change_callback_();
  }
}

} // namespace gui
} // namespace duorou