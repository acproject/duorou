#include "chat_view.h"
#include "../core/logger.h"

#include <iostream>

namespace duorou {
namespace gui {

ChatView::ChatView()
    : main_widget_(nullptr)
    , chat_scrolled_(nullptr)
    , chat_box_(nullptr)
    , input_box_(nullptr)
    , input_entry_(nullptr)
    , send_button_(nullptr)
    , clear_button_(nullptr)
{
}

ChatView::~ChatView() {
    // GTK4会自动清理子组件
}

bool ChatView::initialize() {
    // 创建主容器
    main_widget_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    if (!main_widget_) {
        std::cerr << "Failed to create chat view main container" << std::endl;
        return false;
    }

    gtk_widget_set_margin_start(main_widget_, 10);
    gtk_widget_set_margin_end(main_widget_, 10);
    gtk_widget_set_margin_top(main_widget_, 10);
    gtk_widget_set_margin_bottom(main_widget_, 10);

    // 创建聊天显示区域
    create_chat_area();
    
    // 创建输入区域
    create_input_area();
    
    // 连接信号
    connect_signals();

    std::cout << "Chat view initialized successfully" << std::endl;
    return true;
}

void ChatView::send_message(const std::string& message) {
    if (message.empty()) {
        return;
    }

    // 添加用户消息到聊天显示
    add_message(message, true);
    
    // TODO: 这里应该调用AI模型处理消息
    // 暂时添加一个模拟回复
    add_message("This is a placeholder response. AI integration will be implemented later.", false);
}

void ChatView::add_message(const std::string& message, bool is_user) {
    if (!chat_box_) {
        return;
    }

    // 创建消息容器
    GtkWidget* message_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    gtk_widget_set_margin_start(message_box, 10);
    gtk_widget_set_margin_end(message_box, 10);
    gtk_widget_set_margin_top(message_box, 5);
    gtk_widget_set_margin_bottom(message_box, 5);
    
    // 创建消息标签
    std::string formatted_message = is_user ? "[User]: " + message : "[Assistant]: " + message;
    GtkWidget* message_label = gtk_label_new(formatted_message.c_str());
    gtk_label_set_wrap(GTK_LABEL(message_label), TRUE);
    gtk_label_set_wrap_mode(GTK_LABEL(message_label), PANGO_WRAP_WORD_CHAR);
    gtk_widget_set_halign(message_label, is_user ? GTK_ALIGN_END : GTK_ALIGN_START);
    
    // 添加样式类
    if (is_user) {
        gtk_widget_add_css_class(message_box, "user-message");
    } else {
        gtk_widget_add_css_class(message_box, "assistant-message");
    }
    
    gtk_box_append(GTK_BOX(message_box), message_label);
    gtk_box_append(GTK_BOX(chat_box_), message_box);
    
    // 滚动到底部
    scroll_to_bottom();
}

void ChatView::clear_chat() {
    if (chat_box_) {
        // 移除所有子组件
        GtkWidget* child = gtk_widget_get_first_child(chat_box_);
        while (child) {
            GtkWidget* next = gtk_widget_get_next_sibling(child);
            gtk_box_remove(GTK_BOX(chat_box_), child);
            child = next;
        }
    }
}

void ChatView::create_chat_area() {
    // 创建聊天消息容器
    chat_box_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_widget_set_valign(chat_box_, GTK_ALIGN_START);
    
    // 创建滚动窗口
    chat_scrolled_ = gtk_scrolled_window_new();
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(chat_scrolled_), 
                                   GTK_POLICY_AUTOMATIC, GTK_POLICY_AUTOMATIC);
    gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(chat_scrolled_), chat_box_);
    
    // 设置滚动窗口大小
    gtk_widget_set_vexpand(chat_scrolled_, TRUE);
    gtk_widget_set_hexpand(chat_scrolled_, TRUE);
    
    // 添加到主容器
    gtk_box_append(GTK_BOX(main_widget_), chat_scrolled_);
}

void ChatView::create_input_area() {
    // 创建输入容器
    input_box_ = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    gtk_widget_set_margin_start(input_box_, 10);
    gtk_widget_set_margin_end(input_box_, 10);
    gtk_widget_set_margin_bottom(input_box_, 10);
    
    // 创建消息输入框
    input_entry_ = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(input_entry_), "Type your message here...");
    gtk_widget_set_hexpand(input_entry_, TRUE);
    
    // 创建发送按钮
    send_button_ = gtk_button_new_with_label("Send");
    gtk_widget_set_size_request(send_button_, 80, -1);
    
    // 创建清空按钮
    clear_button_ = gtk_button_new_with_label("Clear");
    gtk_widget_set_size_request(clear_button_, 80, -1);
    
    // 添加到输入容器
    gtk_box_append(GTK_BOX(input_box_), input_entry_);
    gtk_box_append(GTK_BOX(input_box_), send_button_);
    gtk_box_append(GTK_BOX(input_box_), clear_button_);
    
    // 添加到主容器
    gtk_box_append(GTK_BOX(main_widget_), input_box_);
}

void ChatView::connect_signals() {
    // 连接发送按钮信号
    g_signal_connect(send_button_, "clicked", G_CALLBACK(on_send_button_clicked), this);
    
    // 连接清空按钮信号
    g_signal_connect(clear_button_, "clicked", G_CALLBACK(on_clear_button_clicked), this);
    
    // 连接回车键发送消息
    g_signal_connect(input_entry_, "activate", G_CALLBACK(on_input_entry_activate), this);
}

void ChatView::scroll_to_bottom() {
    if (chat_scrolled_) {
        GtkAdjustment* vadj = gtk_scrolled_window_get_vadjustment(GTK_SCROLLED_WINDOW(chat_scrolled_));
        if (vadj) {
            gtk_adjustment_set_value(vadj, gtk_adjustment_get_upper(vadj));
        }
    }
}

// 静态回调函数实现
void ChatView::on_send_button_clicked(GtkWidget* widget, gpointer user_data) {
    ChatView* chat_view = static_cast<ChatView*>(user_data);
    
    // 获取输入文本
    const char* text = gtk_entry_buffer_get_text(gtk_entry_get_buffer(GTK_ENTRY(chat_view->input_entry_)));
    if (text && strlen(text) > 0) {
        std::string message(text);
        chat_view->send_message(message);
        
        // 清空输入框
        gtk_entry_buffer_set_text(gtk_entry_get_buffer(GTK_ENTRY(chat_view->input_entry_)), "", 0);
    }
}

void ChatView::on_clear_button_clicked(GtkWidget* widget, gpointer user_data) {
    ChatView* chat_view = static_cast<ChatView*>(user_data);
    chat_view->clear_chat();
}

void ChatView::on_input_entry_activate(GtkWidget* widget, gpointer user_data) {
    ChatView* chat_view = static_cast<ChatView*>(user_data);
    
    // 获取输入文本
    const char* text = gtk_entry_buffer_get_text(gtk_entry_get_buffer(GTK_ENTRY(widget)));
    if (text && strlen(text) > 0) {
        std::string message(text);
        chat_view->send_message(message);
        
        // 清空输入框
        gtk_entry_buffer_set_text(gtk_entry_get_buffer(GTK_ENTRY(widget)), "", 0);
    }
}

} // namespace gui
} // namespace duorou