#ifndef DUOROU_GUI_CHAT_SESSION_H
#define DUOROU_GUI_CHAT_SESSION_H

#include <string>
#include <vector>
#include <memory>
#include <chrono>

namespace duorou {
namespace gui {

/**
 * 聊天消息结构
 */
struct ChatMessage {
    std::string content;     // 消息内容
    bool is_user;           // 是否为用户消息
    std::chrono::system_clock::time_point timestamp; // 时间戳
    
    ChatMessage(const std::string& msg, bool user) 
        : content(msg), is_user(user), timestamp(std::chrono::system_clock::now()) {}
};

/**
 * 聊天会话类 - 管理单个聊天会话的消息历史
 */
class ChatSession {
public:
    ChatSession();
    explicit ChatSession(const std::string& title);
    ~ChatSession() = default;

    // 禁用拷贝构造和赋值
    ChatSession(const ChatSession&) = delete;
    ChatSession& operator=(const ChatSession&) = delete;

    /**
     * 添加消息到会话
     * @param message 消息内容
     * @param is_user 是否为用户消息
     */
    void add_message(const std::string& message, bool is_user);

    /**
     * 获取所有消息
     * @return 消息列表
     */
    const std::vector<ChatMessage>& get_messages() const { return messages_; }

    /**
     * 获取会话标题
     * @return 会话标题
     */
    const std::string& get_title() const { return title_; }

    /**
     * 设置会话标题
     * @param title 新标题
     */
    void set_title(const std::string& title) { title_ = title; }

    /**
     * 获取会话ID
     * @return 会话ID
     */
    const std::string& get_id() const { return id_; }

    /**
     * 获取创建时间
     * @return 创建时间
     */
    const std::chrono::system_clock::time_point& get_created_time() const { return created_time_; }

    /**
     * 获取最后更新时间
     * @return 最后更新时间
     */
    const std::chrono::system_clock::time_point& get_last_updated() const { return last_updated_; }

    /**
     * 清空会话消息
     */
    void clear_messages();

    /**
     * 检查会话是否为空
     * @return 如果没有消息返回true
     */
    bool is_empty() const { return messages_.empty(); }

private:
    std::string id_;                                    // 会话唯一ID
    std::string title_;                                 // 会话标题
    std::vector<ChatMessage> messages_;                 // 消息列表
    std::chrono::system_clock::time_point created_time_; // 创建时间
    std::chrono::system_clock::time_point last_updated_; // 最后更新时间

    /**
     * 生成唯一ID
     * @return 唯一ID字符串
     */
    std::string generate_id();

    /**
     * 更新最后修改时间
     */
    void update_timestamp();
};

} // namespace gui
} // namespace duorou

#endif // DUOROU_GUI_CHAT_SESSION_H