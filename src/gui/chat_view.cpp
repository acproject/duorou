#include "chat_view.h"
#include "../core/logger.h"

#include <iostream>

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
      model_selector_(nullptr), input_container_(nullptr), welcome_cleared_(false) {}

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

  // 创建录制视频按钮
  video_record_button_ = gtk_button_new_with_label("🎥");
  gtk_widget_add_css_class(video_record_button_, "upload-button");
  gtk_widget_set_size_request(video_record_button_, 40, 40);
  gtk_widget_set_tooltip_text(video_record_button_, "录制视频/桌面捕获");

  // 创建发送按钮
  send_button_ = gtk_button_new_with_label("↑");
  gtk_widget_add_css_class(send_button_, "send-button");
  gtk_widget_set_size_request(send_button_, 40, 40);

  // 添加到输入容器
  gtk_box_append(GTK_BOX(input_container_), upload_image_button_);
  gtk_box_append(GTK_BOX(input_container_), upload_file_button_);
  gtk_box_append(GTK_BOX(input_container_), video_record_button_);
  gtk_box_append(GTK_BOX(input_container_), input_entry_);
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
   
   // 创建选择对话框
   GtkWidget *dialog = gtk_dialog_new_with_buttons(
     "选择视频源",
     GTK_WINDOW(gtk_widget_get_root(widget)),
     GTK_DIALOG_MODAL,
     "取消", GTK_RESPONSE_CANCEL,
     "确定", GTK_RESPONSE_OK,
     NULL);
   
   // 创建内容区域
   GtkWidget *content_area = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
   GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
   gtk_widget_set_margin_start(vbox, 20);
   gtk_widget_set_margin_end(vbox, 20);
   gtk_widget_set_margin_top(vbox, 20);
   gtk_widget_set_margin_bottom(vbox, 20);
   
   // 创建单选按钮组
   GtkWidget *desktop_radio = gtk_check_button_new_with_label("🖥️ 桌面捕获 (默认)");
   GtkWidget *camera_radio = gtk_check_button_new_with_label("📹 摄像头");
   
   // 设置单选按钮组
   gtk_check_button_set_group(GTK_CHECK_BUTTON(camera_radio), GTK_CHECK_BUTTON(desktop_radio));
   gtk_check_button_set_active(GTK_CHECK_BUTTON(desktop_radio), TRUE);
   
   // 添加到容器
   gtk_box_append(GTK_BOX(vbox), desktop_radio);
   gtk_box_append(GTK_BOX(vbox), camera_radio);
   gtk_box_append(GTK_BOX(content_area), vbox);
   
   // 显示对话框
   gtk_widget_show(dialog);
   
   // 连接响应信号
   g_signal_connect(dialog, "response", G_CALLBACK(+[](GtkDialog *dialog, gint response_id, gpointer user_data) {
     ChatView *chat_view = static_cast<ChatView *>(user_data);
     
     if (response_id == GTK_RESPONSE_OK) {
       // 获取内容区域
       GtkWidget *content_area = gtk_dialog_get_content_area(dialog);
       GtkWidget *vbox = gtk_widget_get_first_child(content_area);
       GtkWidget *desktop_radio = gtk_widget_get_first_child(vbox);
       GtkWidget *camera_radio = gtk_widget_get_next_sibling(desktop_radio);
       
       bool use_desktop = gtk_check_button_get_active(GTK_CHECK_BUTTON(desktop_radio));
       
       if (use_desktop) {
         chat_view->start_desktop_capture();
       } else {
         chat_view->start_camera_capture();
       }
     }
     
     gtk_window_destroy(GTK_WINDOW(dialog));
   }), chat_view);
  }
  
  void ChatView::start_desktop_capture() {
    std::cout << "Starting desktop capture..." << std::endl;
    
    // 显示信息对话框
    GtkWidget *dialog = gtk_message_dialog_new(
      GTK_WINDOW(gtk_widget_get_root(main_widget_)),
      GTK_DIALOG_MODAL,
      GTK_MESSAGE_INFO,
      GTK_BUTTONS_OK,
      "桌面捕获功能已启动\n\n这是一个演示功能，实际的桌面捕获需要集成屏幕录制库。");
    
    gtk_widget_show(dialog);
    g_signal_connect(dialog, "response", G_CALLBACK(gtk_window_destroy), NULL);
    
    // TODO: 集成实际的桌面捕获功能
    // 可以使用 FFmpeg, GStreamer 或系统API
  }
  
  void ChatView::start_camera_capture() {
    std::cout << "Starting camera capture..." << std::endl;
    
    // 检查摄像头可用性（简化版本）
    bool camera_available = false;
    
    // TODO: 实际检查摄像头设备
    // 在macOS上可以使用AVFoundation框架
    // 在Linux上可以检查/dev/video*设备
    
    if (!camera_available) {
      // 显示警告对话框并回退到桌面捕获
      GtkWidget *dialog = gtk_message_dialog_new(
        GTK_WINDOW(gtk_widget_get_root(main_widget_)),
        GTK_DIALOG_MODAL,
        GTK_MESSAGE_WARNING,
        GTK_BUTTONS_OK,
        "未检测到可用的摄像头设备\n\n将自动切换到桌面捕获模式。");
      
      gtk_widget_show(dialog);
      g_signal_connect(dialog, "response", G_CALLBACK(+[](GtkDialog *dialog, gint response_id, gpointer user_data) {
        ChatView *chat_view = static_cast<ChatView *>(user_data);
        gtk_window_destroy(GTK_WINDOW(dialog));
        // 回退到桌面捕获
        chat_view->start_desktop_capture();
      }), this);
    } else {
      // 显示摄像头启动信息
      GtkWidget *dialog = gtk_message_dialog_new(
        GTK_WINDOW(gtk_widget_get_root(main_widget_)),
        GTK_DIALOG_MODAL,
        GTK_MESSAGE_INFO,
        GTK_BUTTONS_OK,
        "摄像头捕获功能已启动\n\n这是一个演示功能，实际的摄像头捕获需要集成相机库。");
      
      gtk_widget_show(dialog);
      g_signal_connect(dialog, "response", G_CALLBACK(gtk_window_destroy), NULL);
      
      // TODO: 集成实际的摄像头捕获功能
      // 可以使用 OpenCV, GStreamer 或系统API
    }
  }
  
  } // namespace gui
  } // namespace duorou