#include "chat_view.h"
#include "../core/logger.h"
#include "../media/video_capture.h"
#include "../media/audio_capture.h"

#include <iostream>
#include <thread>
#include <chrono>

namespace duorou {
namespace gui {

ChatView::ChatView()
    : main_widget_(nullptr), chat_scrolled_(nullptr), chat_box_(nullptr),
      input_box_(nullptr), input_entry_(nullptr), send_button_(nullptr),
      upload_image_button_(nullptr),
    upload_file_button_(nullptr),
    video_record_button_(nullptr),
    selected_image_path_(""),
    selected_file_path_(""),
      model_selector_(nullptr), input_container_(nullptr), welcome_cleared_(false),
      video_capture_(nullptr),
    audio_capture_(nullptr),
    video_display_window_(std::make_unique<VideoDisplayWindow>()),
    is_recording_(false) {}

ChatView::~ChatView() {
  // 确保停止录制并清理媒体资源
  if (is_recording_) {
    stop_recording();
  }
  
  // 显式清理音频和视频捕获对象
  if (audio_capture_) {
    audio_capture_->stop_capture();
    audio_capture_.reset();
  }
  
  if (video_capture_) {
    video_capture_->stop_capture();
    video_capture_.reset();
  }
  
  // 清理视频显示窗口
  if (video_display_window_) {
    video_display_window_->hide();
    video_display_window_.reset();
  }
  
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

void ChatView::send_message(const std::string &message) {
  if (message.empty()) {
    return;
  }

  // 添加用户消息到聊天显示
  add_message(message, true);

  // TODO: 这里应该调用AI模型处理消息
  // 暂时添加一个模拟回复
  add_message("This is a placeholder response. AI integration will be "
              "implemented later.",
              false);
}

void ChatView::add_message(const std::string &message, bool is_user) {
  if (!chat_box_) {
    return;
  }

  // 创建消息容器 - 使用水平布局实现对齐
  GtkWidget *message_container = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
  gtk_widget_set_margin_start(message_container, 10);
  gtk_widget_set_margin_end(message_container, 10);
  gtk_widget_set_margin_top(message_container, 4);
  gtk_widget_set_margin_bottom(message_container, 4);

  // 创建消息标签 - 不添加前缀，直接显示消息内容
  GtkWidget *message_label = gtk_label_new(message.c_str());
  gtk_label_set_wrap(GTK_LABEL(message_label), TRUE);
  gtk_label_set_wrap_mode(GTK_LABEL(message_label), PANGO_WRAP_WORD_CHAR);
  gtk_label_set_max_width_chars(GTK_LABEL(message_label), 50); // 限制最大字符数
  gtk_label_set_xalign(GTK_LABEL(message_label), 0.0); // 左对齐文本
  
  // 创建气泡框架容器
  GtkWidget *bubble_frame = gtk_frame_new(NULL);
  gtk_frame_set_child(GTK_FRAME(bubble_frame), message_label);
  
  // 创建气泡容器
  GtkWidget *bubble_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
  
  if (is_user) {
    // 用户消息：右对齐，直接设置背景色
    gtk_widget_add_css_class(bubble_frame, "user-bubble");
    // 直接设置背景色和样式
    GtkCssProvider *provider = gtk_css_provider_new();
    gtk_css_provider_load_from_string(provider, 
      "frame { background: #48bb78; color: white; border-radius: 18px; padding: 12px 16px; margin: 4px; border: none; }");
    gtk_style_context_add_provider(gtk_widget_get_style_context(bubble_frame),
                                   GTK_STYLE_PROVIDER(provider),
                                   GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
    g_object_unref(provider);
    gtk_widget_set_halign(bubble_box, GTK_ALIGN_END);
    
    // 添加左侧空白以实现右对齐效果
    GtkWidget *spacer = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    gtk_widget_set_hexpand(spacer, TRUE);
    gtk_box_append(GTK_BOX(message_container), spacer);
    
    gtk_box_append(GTK_BOX(bubble_box), bubble_frame);
    gtk_box_append(GTK_BOX(message_container), bubble_box);
  } else {
    // AI助手消息：左对齐，直接设置背景色
    gtk_widget_add_css_class(bubble_frame, "assistant-bubble");
    // 直接设置背景色和样式
    GtkCssProvider *provider = gtk_css_provider_new();
    gtk_css_provider_load_from_string(provider, 
      "frame { background: #bee3f8; color: #2d3748; border: 1px solid #90cdf4; border-radius: 18px; padding: 12px 16px; margin: 4px; }");
    gtk_style_context_add_provider(gtk_widget_get_style_context(bubble_frame),
                                   GTK_STYLE_PROVIDER(provider),
                                   GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
    g_object_unref(provider);
    gtk_widget_set_halign(bubble_box, GTK_ALIGN_START);
    
    gtk_box_append(GTK_BOX(bubble_box), bubble_frame);
    gtk_box_append(GTK_BOX(message_container), bubble_box);
    
    // 添加右侧空白以实现左对齐效果
    GtkWidget *spacer = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    gtk_widget_set_hexpand(spacer, TRUE);
    gtk_box_append(GTK_BOX(message_container), spacer);
  }

  gtk_box_append(GTK_BOX(chat_box_), message_container);

  // 滚动到底部
  scroll_to_bottom();
}

void ChatView::clear_chat() {
  if (chat_box_) {
    // 移除所有子组件
    GtkWidget *child = gtk_widget_get_first_child(chat_box_);
    while (child) {
      GtkWidget *next = gtk_widget_get_next_sibling(child);
      gtk_box_remove(GTK_BOX(chat_box_), child);
      child = next;
    }
  }
}

void ChatView::create_chat_area() {
  // 创建聊天消息容器
  chat_box_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
  gtk_widget_set_valign(chat_box_, GTK_ALIGN_START);

  // 创建欢迎界面
  create_welcome_screen();

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
  // 创建主输入容器
  input_box_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
  gtk_widget_set_margin_start(input_box_, 20);
  gtk_widget_set_margin_end(input_box_, 20);
  gtk_widget_set_margin_bottom(input_box_, 20);
  gtk_widget_set_margin_top(input_box_, 10);

  // 创建模型选择器容器
  GtkWidget *model_container = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
  gtk_widget_set_halign(model_container, GTK_ALIGN_CENTER);

  // 创建模型选择器
  model_selector_ = gtk_drop_down_new_from_strings(
      (const char *[]){"gpt-3.5-turbo", "gpt-4", "claude-3", "llama2", NULL});
  gtk_widget_add_css_class(model_selector_, "model-selector");

  // 创建模型标签
  GtkWidget *model_label = gtk_label_new("Model:");
  gtk_widget_add_css_class(model_label, "model-label");

  gtk_box_append(GTK_BOX(model_container), model_label);
  gtk_box_append(GTK_BOX(model_container), model_selector_);

  // 创建输入框容器
  input_container_ = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
  gtk_widget_add_css_class(input_container_, "input-container");
  gtk_widget_set_hexpand(input_container_, TRUE);

  // 创建消息输入框
  input_entry_ = gtk_entry_new();
  gtk_entry_set_placeholder_text(GTK_ENTRY(input_entry_), "Send a message...");
  gtk_widget_set_hexpand(input_entry_, TRUE);
  gtk_widget_add_css_class(input_entry_, "message-input");
  
  // 设置输入法相关属性以避免Pango错误
  gtk_entry_set_input_purpose(GTK_ENTRY(input_entry_), GTK_INPUT_PURPOSE_FREE_FORM);
  gtk_entry_set_input_hints(GTK_ENTRY(input_entry_), GTK_INPUT_HINT_NONE);
  
  // 禁用一些可能导致Pango错误的功能
  gtk_entry_set_has_frame(GTK_ENTRY(input_entry_), TRUE);
  gtk_entry_set_activates_default(GTK_ENTRY(input_entry_), FALSE);
  
  // 设置最大长度以避免缓冲区溢出
  gtk_entry_set_max_length(GTK_ENTRY(input_entry_), 1000);
  
  // 设置覆写模式为FALSE，避免光标位置问题
  gtk_entry_set_overwrite_mode(GTK_ENTRY(input_entry_), FALSE);
  
  // 启用输入法支持
  gtk_widget_set_can_focus(input_entry_, TRUE);
  gtk_widget_set_focusable(input_entry_, TRUE);

  // 创建上传图片按钮
  upload_image_button_ = gtk_button_new_with_label("🖼️");
  gtk_widget_add_css_class(upload_image_button_, "upload-button");
  gtk_widget_set_size_request(upload_image_button_, 40, 40);
  gtk_widget_set_tooltip_text(upload_image_button_, "Upload Image");

  // 创建上传文件按钮
  upload_file_button_ = gtk_button_new_with_label("📎");
  gtk_widget_add_css_class(upload_file_button_, "upload-button");
  gtk_widget_set_size_request(upload_file_button_, 40, 40);
  gtk_widget_set_tooltip_text(upload_file_button_, "Upload File (MD, DOC, Excel, PPT, PDF)");

  // 创建录制视频按钮图标 - 使用相对路径
  std::string icon_path_base = "src/gui/";
  video_off_image_ = gtk_picture_new_for_filename((icon_path_base + "video-off.png").c_str());
  video_on_image_ = gtk_picture_new_for_filename((icon_path_base + "video-on.png").c_str());
  
  // 检查图标是否加载成功
  if (!video_off_image_ || !video_on_image_) {
    std::cout << "警告: 无法加载录制按钮图标，使用文本替代" << std::endl;
    // 如果图标加载失败，创建文本标签作为替代
    if (!video_off_image_) {
      video_off_image_ = gtk_label_new("⏹");
    }
    if (!video_on_image_) {
      video_on_image_ = gtk_label_new("⏺");
    }
  }
  
  // 设置图标大小
  gtk_widget_set_size_request(video_off_image_, 24, 24);
  gtk_widget_set_size_request(video_on_image_, 24, 24);
  
  // 创建录制视频按钮
  video_record_button_ = gtk_button_new();
  gtk_button_set_child(GTK_BUTTON(video_record_button_), video_off_image_); // 默认显示关闭状态
  gtk_widget_add_css_class(video_record_button_, "upload-button");
  gtk_widget_set_size_request(video_record_button_, 40, 40);
  gtk_widget_set_tooltip_text(video_record_button_, "开始录制视频/桌面捕获");

  // 创建发送按钮
  send_button_ = gtk_button_new_with_label("↑");
  gtk_widget_add_css_class(send_button_, "send-button");
  gtk_widget_set_size_request(send_button_, 40, 40);

  // 添加到输入容器
  gtk_box_append(GTK_BOX(input_container_), upload_image_button_);
  gtk_box_append(GTK_BOX(input_container_), upload_file_button_);
  gtk_box_append(GTK_BOX(input_container_), input_entry_);
  gtk_box_append(GTK_BOX(input_container_), video_record_button_);
  gtk_box_append(GTK_BOX(input_container_), send_button_);

  // 添加到主输入容器
  gtk_box_append(GTK_BOX(input_box_), model_container);
  gtk_box_append(GTK_BOX(input_box_), input_container_);

  // 添加到主容器
  gtk_box_append(GTK_BOX(main_widget_), input_box_);
}

void ChatView::create_welcome_screen() {
  // 创建欢迎界面容器
  GtkWidget *welcome_container = gtk_box_new(GTK_ORIENTATION_VERTICAL, 20);
  gtk_widget_set_halign(welcome_container, GTK_ALIGN_CENTER);
  gtk_widget_set_valign(welcome_container, GTK_ALIGN_CENTER);
  gtk_widget_set_vexpand(welcome_container, TRUE);
  gtk_widget_set_hexpand(welcome_container, TRUE);

  // 创建应用图标 (使用duorou01.png图片)
  // 使用绝对路径确保能找到图片文件
  const char* icon_path = "/Users/acproject/workspace/cpp_projects/duorou/src/gui/duorou01.png";
  GtkWidget *icon_picture = gtk_picture_new_for_filename(icon_path);
  
  // 如果绝对路径失败，尝试相对路径
  if (!gtk_picture_get_file(GTK_PICTURE(icon_picture))) {
    g_object_unref(icon_picture);
    icon_picture = gtk_picture_new_for_filename("src/gui/duorou01.png");
  }
  
  gtk_picture_set_content_fit(GTK_PICTURE(icon_picture),
                              GTK_CONTENT_FIT_CONTAIN);
  gtk_widget_set_size_request(icon_picture, 16, 16);
  gtk_widget_add_css_class(icon_picture, "welcome-icon");

  // 创建欢迎文本
  GtkWidget *welcome_title = gtk_label_new("Welcome to Duorou");
  gtk_widget_add_css_class(welcome_title, "welcome-title");

  GtkWidget *welcome_subtitle = gtk_label_new("Your AI Desktop Assistant");
  gtk_widget_add_css_class(welcome_subtitle, "welcome-subtitle");

  GtkWidget *welcome_hint =
      gtk_label_new("Start a conversation by typing a message below");
  gtk_widget_add_css_class(welcome_hint, "welcome-hint");

  // 添加到容器
  gtk_box_append(GTK_BOX(welcome_container), icon_picture);
  gtk_box_append(GTK_BOX(welcome_container), welcome_title);
  gtk_box_append(GTK_BOX(welcome_container), welcome_subtitle);
  gtk_box_append(GTK_BOX(welcome_container), welcome_hint);

  // 添加到聊天容器
  gtk_box_append(GTK_BOX(chat_box_), welcome_container);
}

void ChatView::connect_signals() {
  // 连接发送按钮信号
  g_signal_connect(send_button_, "clicked", G_CALLBACK(on_send_button_clicked),
                   this);

  // 连接上传图片按钮信号
  g_signal_connect(upload_image_button_, "clicked", G_CALLBACK(on_upload_image_button_clicked),
                   this);

  // 连接上传文件按钮信号
  g_signal_connect(upload_file_button_, "clicked", G_CALLBACK(on_upload_file_button_clicked),
                   this);

  // 连接录制视频按钮信号
  g_signal_connect(video_record_button_, "clicked", G_CALLBACK(on_video_record_button_clicked),
                   this);

  // 连接回车键发送消息
  g_signal_connect(input_entry_, "activate",
                   G_CALLBACK(on_input_entry_activate), this);
}

void ChatView::scroll_to_bottom() {
  if (chat_scrolled_) {
    GtkAdjustment *vadj = gtk_scrolled_window_get_vadjustment(
        GTK_SCROLLED_WINDOW(chat_scrolled_));
    if (vadj) {
      gtk_adjustment_set_value(vadj, gtk_adjustment_get_upper(vadj));
    }
  }
}

// 静态回调函数实现
void ChatView::on_send_button_clicked(GtkWidget *widget, gpointer user_data) {
  ChatView *chat_view = static_cast<ChatView *>(user_data);

  if (!chat_view->input_entry_) {
    return;
  }

  // 使用gtk_editable_get_text直接获取文本，避免buffer操作导致的Pango错误
  const char *text_ptr = gtk_editable_get_text(GTK_EDITABLE(chat_view->input_entry_));
  std::string message_text = text_ptr ? std::string(text_ptr) : "";
  
  // 检查是否有文本消息或选择的文件
  bool has_text = !message_text.empty();
  bool has_image = !chat_view->selected_image_path_.empty();
  bool has_file = !chat_view->selected_file_path_.empty();
  
  if (has_text || has_image || has_file) {
    // 使用gtk_editable_set_text清空输入框
    gtk_editable_set_text(GTK_EDITABLE(chat_view->input_entry_), "");
    
    // 只在第一次发送消息时清除欢迎界面
    if (!chat_view->welcome_cleared_) {
      chat_view->clear_chat();
      chat_view->welcome_cleared_ = true;
    }
    
    // 构建完整消息
    std::string full_message = message_text;
    
    // 添加图片信息
    if (has_image) {
      if (!full_message.empty()) full_message += "\n";
      full_message += "📷 图片: " + std::string(g_path_get_basename(chat_view->selected_image_path_.c_str()));
    }
    
    // 添加文档信息
    if (has_file) {
      if (!full_message.empty()) full_message += "\n";
      full_message += "📎 文档: " + std::string(g_path_get_basename(chat_view->selected_file_path_.c_str()));
    }
    
    // 发送消息
    chat_view->send_message(full_message);
    
    // 清空选择的文件路径并重置按钮提示
    if (has_image) {
      chat_view->selected_image_path_.clear();
      gtk_widget_set_tooltip_text(chat_view->upload_image_button_, "上传图片");
    }
    if (has_file) {
      chat_view->selected_file_path_.clear();
      gtk_widget_set_tooltip_text(chat_view->upload_file_button_, "上传文档");
    }
  }
}

void ChatView::on_input_entry_activate(GtkWidget *widget, gpointer user_data) {
  ChatView *chat_view = static_cast<ChatView *>(user_data);

  if (!widget) {
    return;
  }

  // 使用gtk_editable_get_text直接获取文本，避免buffer操作导致的Pango错误
  const char *text_ptr = gtk_editable_get_text(GTK_EDITABLE(widget));
  std::string message_text = text_ptr ? std::string(text_ptr) : "";
  
  // 检查是否有文本消息或选择的文件
  bool has_text = !message_text.empty();
  bool has_image = !chat_view->selected_image_path_.empty();
  bool has_file = !chat_view->selected_file_path_.empty();
  
  if (has_text || has_image || has_file) {
    // 使用gtk_editable_set_text清空输入框
    gtk_editable_set_text(GTK_EDITABLE(widget), "");
    
    // 只在第一次发送消息时清除欢迎界面
    if (!chat_view->welcome_cleared_) {
      chat_view->clear_chat();
      chat_view->welcome_cleared_ = true;
    }
    
    // 构建完整消息
    std::string full_message = message_text;
    
    // 添加图片信息
    if (has_image) {
      if (!full_message.empty()) full_message += "\n";
      full_message += "📷 图片: " + std::string(g_path_get_basename(chat_view->selected_image_path_.c_str()));
    }
    
    // 添加文档信息
    if (has_file) {
      if (!full_message.empty()) full_message += "\n";
      full_message += "📎 文档: " + std::string(g_path_get_basename(chat_view->selected_file_path_.c_str()));
    }
    
    // 发送消息
    chat_view->send_message(full_message);
    
    // 清空选择的文件路径并重置按钮提示
    if (has_image) {
      chat_view->selected_image_path_.clear();
      gtk_widget_set_tooltip_text(chat_view->upload_image_button_, "上传图片");
    }
    if (has_file) {
      chat_view->selected_file_path_.clear();
      gtk_widget_set_tooltip_text(chat_view->upload_file_button_, "上传文档");
    }
  }
}

void ChatView::on_upload_image_button_clicked(GtkWidget *widget, gpointer user_data) {
  ChatView *chat_view = static_cast<ChatView *>(user_data);
  
  // 创建文件选择对话框
  GtkWidget *dialog = gtk_file_chooser_dialog_new(
    "Select Image",
    GTK_WINDOW(gtk_widget_get_root(widget)),
    GTK_FILE_CHOOSER_ACTION_OPEN,
    "_Cancel", GTK_RESPONSE_CANCEL,
    "_Open", GTK_RESPONSE_ACCEPT,
    NULL);
  
  // 设置图片文件过滤器
  GtkFileFilter *filter = gtk_file_filter_new();
  gtk_file_filter_set_name(filter, "Image files");
  gtk_file_filter_add_mime_type(filter, "image/png");
  gtk_file_filter_add_mime_type(filter, "image/jpeg");
  gtk_file_filter_add_mime_type(filter, "image/gif");
  gtk_file_filter_add_mime_type(filter, "image/bmp");
  gtk_file_filter_add_mime_type(filter, "image/webp");
  gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dialog), filter);
  
  // 显示对话框
  gtk_widget_show(dialog);
  
  // 存储chat_view指针到dialog的数据中
  g_object_set_data(G_OBJECT(dialog), "chat_view", chat_view);
  
  // 连接响应信号
  g_signal_connect(dialog, "response", G_CALLBACK(on_image_dialog_response), NULL);
}

void ChatView::on_upload_file_button_clicked(GtkWidget *widget, gpointer user_data) {
  ChatView *chat_view = static_cast<ChatView *>(user_data);
  
  // 创建文件选择对话框
  GtkWidget *dialog = gtk_file_chooser_dialog_new(
    "Select Document",
    GTK_WINDOW(gtk_widget_get_root(widget)),
    GTK_FILE_CHOOSER_ACTION_OPEN,
    "_Cancel", GTK_RESPONSE_CANCEL,
    "_Open", GTK_RESPONSE_ACCEPT,
    NULL);
  
  // 设置文档文件过滤器
  GtkFileFilter *filter = gtk_file_filter_new();
  gtk_file_filter_set_name(filter, "Document files");
  gtk_file_filter_add_pattern(filter, "*.md");
  gtk_file_filter_add_pattern(filter, "*.doc");
  gtk_file_filter_add_pattern(filter, "*.docx");
  gtk_file_filter_add_pattern(filter, "*.xls");
  gtk_file_filter_add_pattern(filter, "*.xlsx");
  gtk_file_filter_add_pattern(filter, "*.ppt");
  gtk_file_filter_add_pattern(filter, "*.pptx");
  gtk_file_filter_add_pattern(filter, "*.pdf");
  gtk_file_filter_add_pattern(filter, "*.txt");
  gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dialog), filter);
  
  // 显示对话框
  gtk_widget_show(dialog);
  
  // 存储chat_view指针到dialog的数据中
  g_object_set_data(G_OBJECT(dialog), "chat_view", chat_view);
  
  // 连接响应信号
  g_signal_connect(dialog, "response", G_CALLBACK(on_file_dialog_response), NULL);
}

void ChatView::on_image_dialog_response(GtkDialog *dialog, gint response_id, gpointer user_data) {
  ChatView *chat_view = static_cast<ChatView *>(g_object_get_data(G_OBJECT(dialog), "chat_view"));
  
  if (response_id == GTK_RESPONSE_ACCEPT) {
    GtkFileChooser *chooser = GTK_FILE_CHOOSER(dialog);
    GFile *file = gtk_file_chooser_get_file(chooser);
    
    if (file) {
      char *filename = g_file_get_path(file);
      if (filename) {
        // 存储选择的图片路径，不直接发送
        chat_view->selected_image_path_ = std::string(filename);
        
        // 更新上传按钮的提示文本或样式来表示已选择文件
        gtk_widget_set_tooltip_text(chat_view->upload_image_button_, 
                                   ("已选择图片: " + std::string(g_path_get_basename(filename))).c_str());
        
        g_free(filename);
      }
      g_object_unref(file);
    }
  }
  
  gtk_window_destroy(GTK_WINDOW(dialog));
}

void ChatView::on_file_dialog_response(GtkDialog *dialog, gint response_id, gpointer user_data) {
  ChatView *chat_view = static_cast<ChatView *>(g_object_get_data(G_OBJECT(dialog), "chat_view"));
  
  if (response_id == GTK_RESPONSE_ACCEPT) {
    GtkFileChooser *chooser = GTK_FILE_CHOOSER(dialog);
    GFile *file = gtk_file_chooser_get_file(chooser);
    
    if (file) {
      char *filename = g_file_get_path(file);
      if (filename) {
        // 存储选择的文档路径，不直接发送
        chat_view->selected_file_path_ = std::string(filename);
        
        // 更新上传按钮的提示文本或样式来表示已选择文件
        gtk_widget_set_tooltip_text(chat_view->upload_file_button_, 
                                   ("已选择文档: " + std::string(g_path_get_basename(filename))).c_str());
        
        g_free(filename);
      }
      g_object_unref(file);
    }
  }
  
  gtk_window_destroy(GTK_WINDOW(dialog));
}

void ChatView::on_video_record_button_clicked(GtkWidget *widget, gpointer user_data) {
   ChatView *chat_view = static_cast<ChatView *>(user_data);
   
   // Toggle功能：如果正在录制则停止，否则开始桌面捕获
   if (chat_view->is_recording_) {
     chat_view->stop_recording();
   } else {
     chat_view->start_desktop_capture();
   }
  }
  
  void ChatView::start_desktop_capture() {
    std::cout << "Starting desktop capture..." << std::endl;
    
    if (is_recording_) {
      // 停止当前录制
      stop_recording();
      return;
    }
    
    // 确保之前的资源已经清理
    if (video_capture_) {
      video_capture_->stop_capture();
      video_capture_.reset();
    }
    if (audio_capture_) {
      audio_capture_->stop_capture();
      audio_capture_.reset();
    }
    
    // 初始化视频捕获
    video_capture_ = std::make_unique<media::VideoCapture>();
    
    // 初始化音频捕获
    audio_capture_ = std::make_unique<media::AudioCapture>();
    
    // 设置视频帧回调
    video_capture_->set_frame_callback([this](const media::VideoFrame& frame) {
      // 处理视频帧数据
      std::cout << "收到视频帧: " << frame.width << "x" << frame.height 
                << ", 时间戳: " << frame.timestamp << std::endl;
      
      // 更新视频显示窗口（在主线程中执行）
      if (video_display_window_) {
        // 复制帧数据到堆内存，避免生命周期问题
        auto frame_copy = std::make_shared<media::VideoFrame>(frame);
        
        g_idle_add([](gpointer user_data) -> gboolean {
          auto* data = static_cast<std::pair<ChatView*, std::shared_ptr<media::VideoFrame>>*>(user_data);
          ChatView* chat_view = data->first;
          auto frame_ptr = data->second;
          
          // 检查ChatView对象是否仍然有效
          if (chat_view && chat_view->video_display_window_) {
            try {
              chat_view->video_display_window_->update_frame(*frame_ptr);
              
              if (!chat_view->video_display_window_->is_visible()) {
                std::cout << "显示视频窗口..." << std::endl;
                chat_view->video_display_window_->show();
              }
            } catch (const std::exception& e) {
              std::cout << "更新视频帧时出错: " << e.what() << std::endl;
            }
          }
          
          delete data;
          return G_SOURCE_REMOVE;
        }, new std::pair<ChatView*, std::shared_ptr<media::VideoFrame>>(this, frame_copy));
      }
    });
    
    // 设置音频帧回调
    audio_capture_->set_frame_callback([this](const media::AudioFrame& frame) {
      // 处理音频帧数据
      std::cout << "收到音频帧: " << frame.frame_count << " 采样, "
                << frame.sample_rate << "Hz, 时间戳: " << frame.timestamp << std::endl;
    });
    
    // 初始化桌面捕获
    if (video_capture_->initialize(media::VideoSource::DESKTOP_CAPTURE)) {
      if (video_capture_->start_capture()) {
        // 初始化麦克风音频捕获
        if (audio_capture_->initialize(media::AudioSource::MICROPHONE)) {
          if (audio_capture_->start_capture()) {
            is_recording_ = true;
            
            // 安全地切换按钮图标
            if (video_record_button_ && video_on_image_) {
              gtk_button_set_child(GTK_BUTTON(video_record_button_), video_on_image_);
              gtk_widget_set_tooltip_text(video_record_button_, "停止录制");
            }
            
            // 显示成功信息
            GtkWidget *dialog = gtk_message_dialog_new(
              GTK_WINDOW(gtk_widget_get_root(main_widget_)),
              GTK_DIALOG_MODAL,
              GTK_MESSAGE_INFO,
              GTK_BUTTONS_OK,
              "桌面录制已开始\n\n正在捕获桌面视频和麦克风音频。");
            
            // 确保对话框在最上层
            gtk_window_set_modal(GTK_WINDOW(dialog), TRUE);
            gtk_window_present(GTK_WINDOW(dialog));
            gtk_widget_show(dialog);
            g_signal_connect(dialog, "response", G_CALLBACK(gtk_window_destroy), NULL);
          } else {
            std::cout << "音频捕获启动失败" << std::endl;
          }
        } else {
          std::cout << "音频捕获初始化失败" << std::endl;
        }
      } else {
        std::cout << "视频捕获启动失败" << std::endl;
      }
    } else {
      std::cout << "视频捕获初始化失败" << std::endl;
      
      // 显示错误信息
      GtkWidget *dialog = gtk_message_dialog_new(
        GTK_WINDOW(gtk_widget_get_root(main_widget_)),
        GTK_DIALOG_MODAL,
        GTK_MESSAGE_ERROR,
        GTK_BUTTONS_OK,
        "桌面捕获初始化失败\n\n请检查系统权限设置。");
      
      // 确保对话框在最上层
      gtk_window_set_modal(GTK_WINDOW(dialog), TRUE);
      gtk_window_present(GTK_WINDOW(dialog));
      gtk_widget_show(dialog);
      g_signal_connect(dialog, "response", G_CALLBACK(gtk_window_destroy), NULL);
    }
  }
  
  void ChatView::start_camera_capture() {
    std::cout << "Starting camera capture..." << std::endl;
    
    if (is_recording_) {
      // 停止当前录制
      stop_recording();
      return;
    }
    
    // 检查摄像头是否可用
    if (!media::VideoCapture::is_camera_available()) {
      // 显示摄像头不可用信息，并提供回退到桌面捕获的选项
      GtkWidget *dialog = gtk_message_dialog_new(
        GTK_WINDOW(gtk_widget_get_root(main_widget_)),
        GTK_DIALOG_MODAL,
        GTK_MESSAGE_WARNING,
        GTK_BUTTONS_NONE,
        "未检测到可用的摄像头设备\n\n是否使用桌面捕获作为替代？");
      
      gtk_dialog_add_button(GTK_DIALOG(dialog), "使用桌面捕获", GTK_RESPONSE_YES);
      gtk_dialog_add_button(GTK_DIALOG(dialog), "取消", GTK_RESPONSE_NO);
      
      // 确保对话框在最上层
      gtk_window_set_modal(GTK_WINDOW(dialog), TRUE);
      gtk_window_present(GTK_WINDOW(dialog));
      gtk_widget_show(dialog);
      g_signal_connect(dialog, "response", G_CALLBACK(+[](GtkDialog *dialog, gint response_id, gpointer user_data) {
        ChatView *chat_view = static_cast<ChatView *>(user_data);
        
        if (response_id == GTK_RESPONSE_YES) {
          chat_view->start_desktop_capture();
        }
        
        gtk_window_destroy(GTK_WINDOW(dialog));
      }), this);
      return;
    }
    
    // 确保之前的资源已经清理
    if (video_capture_) {
      video_capture_->stop_capture();
      video_capture_.reset();
    }
    if (audio_capture_) {
      audio_capture_->stop_capture();
      audio_capture_.reset();
    }
    
    // 初始化视频捕获
    video_capture_ = std::make_unique<media::VideoCapture>();
    
    // 初始化音频捕获
    audio_capture_ = std::make_unique<media::AudioCapture>();
    
    // 设置视频帧回调
    video_capture_->set_frame_callback([this](const media::VideoFrame& frame) {
      // 处理视频帧数据
      std::cout << "收到摄像头视频帧: " << frame.width << "x" << frame.height 
                << ", 时间戳: " << frame.timestamp << std::endl;
      
      // 更新视频显示窗口（在主线程中执行）
      if (video_display_window_) {
        // 复制帧数据到堆内存，避免生命周期问题
        auto frame_copy = std::make_shared<media::VideoFrame>(frame);
        
        g_idle_add([](gpointer user_data) -> gboolean {
          auto* data = static_cast<std::pair<ChatView*, std::shared_ptr<media::VideoFrame>>*>(user_data);
          ChatView* chat_view = data->first;
          auto frame_ptr = data->second;
          
          if (chat_view->video_display_window_) {
            std::cout << "更新摄像头视频显示窗口，帧大小: " << frame_ptr->width << "x" << frame_ptr->height << std::endl;
            chat_view->video_display_window_->update_frame(*frame_ptr);
            if (!chat_view->video_display_window_->is_visible()) {
              std::cout << "显示摄像头视频窗口..." << std::endl;
              chat_view->video_display_window_->show();
            } else {
              std::cout << "摄像头视频窗口已经可见" << std::endl;
            }
          } else {
            std::cout << "video_display_window_ 为空" << std::endl;
          }
          
          delete data;
          return G_SOURCE_REMOVE;
        }, new std::pair<ChatView*, std::shared_ptr<media::VideoFrame>>(this, frame_copy));
      }
    });
    
    // 设置音频帧回调
    audio_capture_->set_frame_callback([this](const media::AudioFrame& frame) {
      // 处理音频帧数据
      std::cout << "收到音频帧: " << frame.frame_count << " 采样, "
                << frame.sample_rate << "Hz, 时间戳: " << frame.timestamp << std::endl;
    });
    
    // 初始化摄像头捕获
    if (video_capture_->initialize(media::VideoSource::CAMERA, 0)) {
      if (video_capture_->start_capture()) {
        // 初始化麦克风音频捕获
        if (audio_capture_->initialize(media::AudioSource::MICROPHONE)) {
          if (audio_capture_->start_capture()) {
            is_recording_ = true;
            
            // 安全地切换按钮图标
            if (video_record_button_ && video_on_image_) {
              gtk_button_set_child(GTK_BUTTON(video_record_button_), video_on_image_);
              gtk_widget_set_tooltip_text(video_record_button_, "停止录制");
            }
            
            // 显示成功信息
            GtkWidget *dialog = gtk_message_dialog_new(
              GTK_WINDOW(gtk_widget_get_root(main_widget_)),
              GTK_DIALOG_MODAL,
              GTK_MESSAGE_INFO,
              GTK_BUTTONS_OK,
              "摄像头录制已开始\n\n正在捕获摄像头视频和麦克风音频。");
            
            // 确保对话框在最上层
            gtk_window_set_modal(GTK_WINDOW(dialog), TRUE);
            gtk_window_present(GTK_WINDOW(dialog));
            gtk_widget_show(dialog);
            g_signal_connect(dialog, "response", G_CALLBACK(gtk_window_destroy), NULL);
          } else {
            std::cout << "音频捕获启动失败" << std::endl;
          }
        } else {
          std::cout << "音频捕获初始化失败" << std::endl;
        }
      } else {
        std::cout << "摄像头捕获启动失败" << std::endl;
      }
    } else {
      std::cout << "摄像头捕获初始化失败" << std::endl;
      
      // 显示错误信息
      GtkWidget *dialog = gtk_message_dialog_new(
        GTK_WINDOW(gtk_widget_get_root(main_widget_)),
        GTK_DIALOG_MODAL,
        GTK_MESSAGE_ERROR,
        GTK_BUTTONS_OK,
        "摄像头捕获初始化失败\n\n请检查摄像头权限设置。");
      
      // 确保对话框在最上层
      gtk_window_set_modal(GTK_WINDOW(dialog), TRUE);
      gtk_window_present(GTK_WINDOW(dialog));
      gtk_widget_show(dialog);
      g_signal_connect(dialog, "response", G_CALLBACK(gtk_window_destroy), NULL);
    }
  }
  
  void ChatView::stop_recording() {
    std::cout << "Stopping recording..." << std::endl;
    
    if (!is_recording_) {
      return;
    }
    
    // 停止视频捕获
    if (video_capture_) {
      video_capture_->stop_capture();
      // 等待一小段时间，确保所有待处理的回调完成
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      video_capture_.reset(); // 重置视频捕获对象
    }
    
    // 停止音频捕获
    if (audio_capture_) {
      audio_capture_->stop_capture();
      audio_capture_.reset(); // 重置音频捕获对象
    }
    
    is_recording_ = false;
    
    // 安全地切换按钮图标
    if (video_record_button_ && video_off_image_) {
      // 直接设置新的子组件，GTK4会自动处理旧组件的移除
      gtk_button_set_child(GTK_BUTTON(video_record_button_), video_off_image_);
      gtk_widget_set_tooltip_text(video_record_button_, "开始录制视频/桌面捕获");
    }
    
    // 隐藏视频显示窗口
    if (video_display_window_) {
      video_display_window_->hide();
    }
    
    // 显示停止信息
    GtkWidget *dialog = gtk_message_dialog_new(
      GTK_WINDOW(gtk_widget_get_root(main_widget_)),
      GTK_DIALOG_MODAL,
      GTK_MESSAGE_INFO,
      GTK_BUTTONS_OK,
      "录制已停止\n\n视频和音频捕获已结束。");
    
    gtk_widget_show(dialog);
    g_signal_connect(dialog, "response", G_CALLBACK(gtk_window_destroy), NULL);
  }
  
  } // namespace gui
  } // namespace duorou