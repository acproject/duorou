#include "chat_session_manager.h"
#include "../core/logger.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

namespace duorou {
namespace gui {

ChatSessionManager::ChatSessionManager() {
    // 创建默认会话
    create_new_session("Welcome Chat");
}

std::string ChatSessionManager::create_new_session(const std::string& title) {
    auto session = std::make_unique<ChatSession>(title);
    std::string session_id = session->get_id();
    
    sessions_.push_back(std::move(session));
    
    // 切换到新会话
    current_session_id_ = session_id;
    
    notify_session_list_change();
    notify_session_change();
    
    return session_id;
}

bool ChatSessionManager::switch_to_session(const std::string& session_id) {
    int index = find_session_index(session_id);
    if (index == -1) {
        return false;
    }
    
    current_session_id_ = session_id;
    notify_session_change();
    return true;
}

bool ChatSessionManager::delete_session(const std::string& session_id) {
    int index = find_session_index(session_id);
    if (index == -1) {
        return false;
    }
    
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
    
    notify_session_list_change();
    return true;
}

ChatSession* ChatSessionManager::get_current_session() {
    return get_session(current_session_id_);
}

ChatSession* ChatSessionManager::get_session(const std::string& session_id) {
    int index = find_session_index(session_id);
    if (index == -1) {
        return nullptr;
    }
    return sessions_[index].get();
}

std::vector<ChatSession*> ChatSessionManager::get_all_sessions() {
    std::vector<ChatSession*> result;
    for (auto& session : sessions_) {
        result.push_back(session.get());
    }
    
    // 按最后更新时间排序（最新的在前）
    std::sort(result.begin(), result.end(), [](ChatSession* a, ChatSession* b) {
        return a->get_last_updated() > b->get_last_updated();
    });
    
    return result;
}

bool ChatSessionManager::add_message_to_current_session(const std::string& message, bool is_user) {
    ChatSession* current = get_current_session();
    if (!current) {
        return false;
    }
    
    current->add_message(message, is_user);
    notify_session_list_change(); // 可能会更新会话标题
    return true;
}

bool ChatSessionManager::clear_current_session() {
    ChatSession* current = get_current_session();
    if (!current) {
        return false;
    }
    
    current->clear_messages();
    return true;
}

bool ChatSessionManager::save_sessions_to_file(const std::string& file_path) {
    try {
        std::ofstream file(file_path);
        if (!file.is_open()) {
            std::cerr << "Failed to open file for writing: " << file_path << std::endl;
            return false;
        }
        
        // 简单的文本格式保存
        file << "DUOROU_SESSIONS_V1\n";
        file << sessions_.size() << "\n";
        
        for (const auto& session : sessions_) {
            file << "SESSION_START\n";
            file << session->get_id() << "\n";
            file << session->get_title() << "\n";
            
            // 转换时间戳
            auto created_time = std::chrono::duration_cast<std::chrono::seconds>(
                session->get_created_time().time_since_epoch()).count();
            auto last_updated = std::chrono::duration_cast<std::chrono::seconds>(
                session->get_last_updated().time_since_epoch()).count();
            
            file << created_time << "\n";
            file << last_updated << "\n";
            
            // 保存消息
            file << session->get_messages().size() << "\n";
            for (const auto& message : session->get_messages()) {
                file << "MESSAGE_START\n";
                file << (message.is_user ? "1" : "0") << "\n";
                
                auto msg_timestamp = std::chrono::duration_cast<std::chrono::seconds>(
                    message.timestamp.time_since_epoch()).count();
                file << msg_timestamp << "\n";
                
                // 保存消息内容，替换换行符
                std::string content = message.content;
                std::replace(content.begin(), content.end(), '\n', '\x01'); // 临时替换换行符
                file << content << "\n";
                file << "MESSAGE_END\n";
            }
            file << "SESSION_END\n";
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving sessions: " << e.what() << std::endl;
        return false;
    }
}

bool ChatSessionManager::load_sessions_from_file(const std::string& file_path) {
    try {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            // 文件不存在不算错误，使用默认会话
            return true;
        }
        
        std::string line;
        if (!std::getline(file, line) || line != "DUOROU_SESSIONS_V1") {
            std::cerr << "Invalid session file format" << std::endl;
            return false;
        }
        
        if (!std::getline(file, line)) {
            return false;
        }
        
        size_t session_count = std::stoul(line);
        
        // 清空现有会话
        sessions_.clear();
        current_session_id_.clear();
        
        for (size_t i = 0; i < session_count; ++i) {
            if (!std::getline(file, line) || line != "SESSION_START") {
                break;
            }
            
            std::string session_id, title;
            if (!std::getline(file, session_id) || !std::getline(file, title)) {
                break;
            }
            
            auto session = std::make_unique<ChatSession>(title);
            
            // 跳过时间戳（暂时不恢复）
            std::getline(file, line); // created_time
            std::getline(file, line); // last_updated
            
            // 读取消息数量
            if (!std::getline(file, line)) {
                break;
            }
            size_t message_count = std::stoul(line);
            
            // 读取消息
            for (size_t j = 0; j < message_count; ++j) {
                if (!std::getline(file, line) || line != "MESSAGE_START") {
                    break;
                }
                
                std::string is_user_str, timestamp_str, content;
                if (!std::getline(file, is_user_str) || 
                    !std::getline(file, timestamp_str) ||
                    !std::getline(file, content)) {
                    break;
                }
                
                bool is_user = (is_user_str == "1");
                
                // 恢复换行符
                std::replace(content.begin(), content.end(), '\x01', '\n');
                
                session->add_message(content, is_user);
                
                std::getline(file, line); // MESSAGE_END
            }
            
            sessions_.push_back(std::move(session));
            std::getline(file, line); // SESSION_END
        }
        
        // 如果没有会话，创建默认会话
        if (sessions_.empty()) {
            create_new_session("Welcome Chat");
        } else {
            // 设置第一个会话为当前会话
            current_session_id_ = sessions_[0]->get_id();
        }
        
        notify_session_list_change();
        notify_session_change();
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading sessions: " << e.what() << std::endl;
        return false;
    }
}

int ChatSessionManager::find_session_index(const std::string& session_id) {
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