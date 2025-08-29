#include "chat_session.h"
#include <iomanip>
#include <random>
#include <sstream>

namespace duorou {
namespace gui {

ChatSession::ChatSession()
    : id_(generate_id()), title_("New Chat"), custom_name_(""),
      created_time_(std::chrono::system_clock::now()),
      last_updated_(created_time_) {}

ChatSession::ChatSession(const std::string &title)
    : id_(generate_id()), title_(title), custom_name_(""),
      created_time_(std::chrono::system_clock::now()),
      last_updated_(created_time_) {}

ChatSession::ChatSession(
    const std::string &id, const std::string &title,
    const std::chrono::system_clock::time_point &created_time,
    const std::chrono::system_clock::time_point &last_updated)
    : id_(id), title_(title), custom_name_(""), created_time_(created_time),
      last_updated_(last_updated) {}

ChatSession::ChatSession(
    const std::string &id, const std::string &title,
    const std::string &custom_name,
    const std::chrono::system_clock::time_point &created_time,
    const std::chrono::system_clock::time_point &last_updated)
    : id_(id), title_(title), custom_name_(custom_name), created_time_(created_time),
      last_updated_(last_updated) {}

void ChatSession::add_message(const std::string &message, bool is_user) {
  messages_.emplace_back(message, is_user);
  update_timestamp();

  // 如果是第一条用户消息且标题还是默认的，则根据消息内容生成标题
  if (is_user && title_ == "New Chat" && !message.empty()) {
    std::string new_title = message;
    // 限制标题长度
    if (new_title.length() > 30) {
      new_title = new_title.substr(0, 27) + "...";
    }
    title_ = new_title;
  }
}

void ChatSession::clear_messages() {
  messages_.clear();
  update_timestamp();
}

std::string ChatSession::generate_id() {
  // 使用时间戳和随机数生成唯一ID
  auto now = std::chrono::system_clock::now();
  auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                       now.time_since_epoch())
                       .count();

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(1000, 9999);

  std::stringstream ss;
  ss << "chat_" << timestamp << "_" << dis(gen);
  return ss.str();
}

void ChatSession::update_timestamp() {
  last_updated_ = std::chrono::system_clock::now();
}

} // namespace gui
} // namespace duorou