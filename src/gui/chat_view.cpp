#include "chat_view.h"
#include "../core/logger.h"
#include "../core/model_manager.h"
#include "../core/text_generator.h"
#include "../extensions/ollama/ollama_model_manager.h"
#include "../media/audio_capture.h"
#include "../media/video_capture.h"
#include "chat_session_manager.h"
#ifdef __APPLE__
#include "../media/macos_screen_capture.h"
#endif

#include <chrono>
#include <iostream>
#include <thread>

namespace duorou {
namespace gui {

ChatView::ChatView()
    : main_widget_(nullptr), chat_scrolled_(nullptr), chat_box_(nullptr),
      input_box_(nullptr), input_entry_(nullptr), send_button_(nullptr),
      upload_image_button_(nullptr), upload_file_button_(nullptr),
      video_record_button_(nullptr), selected_image_path_(""),
      selected_file_path_(""), model_selector_(nullptr),
      input_container_(nullptr), welcome_cleared_(false),
      video_capture_(nullptr), audio_capture_(nullptr),
      enhanced_video_window_(std::make_unique<EnhancedVideoCaptureWindow>()),
      video_source_dialog_(std::make_unique<VideoSourceDialog>()),
      is_recording_(false), updating_button_state_(false),
      session_manager_(nullptr), model_manager_(nullptr),
      cached_video_frame_(nullptr),
      last_video_update_(std::chrono::steady_clock::now()),
      last_audio_update_(std::chrono::steady_clock::now()) {
  // 初始化增强视频窗口
  if (enhanced_video_window_) {
    enhanced_video_window_->initialize();
  }

  // 设置视频窗口关闭回调
  enhanced_video_window_->set_close_callback([this]() {
    stop_recording();
    // 确保按钮被重新启用
    if (video_record_button_) {
      gtk_widget_set_sensitive(video_record_button_, TRUE);
    }
  });

  // 设置窗口选择回调（桌面捕获模式）
  enhanced_video_window_->set_window_selection_callback(
      [this](const EnhancedVideoCaptureWindow::WindowInfo &window_info) {
        std::cout << "窗口选择: " << window_info.title
                  << " (ID: " << window_info.window_id << ")" << std::endl;
        if (video_capture_) {
          video_capture_->set_capture_window_id(window_info.window_id);
          std::cout << "已设置捕获窗口ID: " << window_info.window_id
                    << std::endl;

          // 如果正在录制，使用动态更新窗口功能
          if (video_capture_->is_capturing()) {
            std::cout << "动态更新屏幕捕获窗口..." << std::endl;
#ifdef __APPLE__
            duorou::media::update_macos_screen_capture_window(
                window_info.window_id);
#endif
          }
        }
      });

  // 设置设备选择回调（摄像头模式）
  enhanced_video_window_->set_device_selection_callback(
      [this](const EnhancedVideoCaptureWindow::DeviceInfo &device_info) {
        std::cout << "设备选择: " << device_info.name
                  << " (索引: " << device_info.device_index << ")" << std::endl;
        if (video_capture_) {
          // 记录当前是否在录制状态
          bool was_capturing = video_capture_->is_capturing();

          // 如果正在录制，先停止当前捕获
          if (was_capturing) {
            std::cout << "停止当前摄像头捕获以应用新设备选择..." << std::endl;
            video_capture_->stop_capture();
          }

          video_capture_->set_camera_device_index(device_info.device_index);
          std::cout << "已设置捕获设备索引: " << device_info.device_index
                    << std::endl;

          // 如果选择的是有效设备（索引>=0），总是尝试启动摄像头
          if (device_info.device_index >= 0) {
            std::cout << "重新初始化并启动摄像头捕获..." << std::endl;

            // 重新创建视频捕获对象以确保完全重置
            video_capture_.reset();
            video_capture_ = std::make_unique<media::VideoCapture>();

            // 设置视频帧回调
            video_capture_->set_frame_callback(
                [this](const media::VideoFrame &frame) {
                  static int camera_frame_count = 0;
                  camera_frame_count++;

                  if (camera_frame_count <= 5 || camera_frame_count % 30 == 0) {
                    std::cout << "收到摄像头视频帧 #" << camera_frame_count
                              << ": " << frame.width << "x" << frame.height
                              << std::endl;
                  }

                  auto now = std::chrono::steady_clock::now();
                  auto time_since_last_update =
                      std::chrono::duration_cast<std::chrono::milliseconds>(
                          now - last_video_update_)
                          .count();

                  if (time_since_last_update >= VIDEO_UPDATE_INTERVAL_MS) {
                    last_video_update_ = now;

                    if (enhanced_video_window_) {
                      media::VideoFrame *frame_copy =
                          new media::VideoFrame(frame);

                      g_idle_add(
                          [](gpointer user_data) -> gboolean {
                            auto *data = static_cast<
                                std::pair<ChatView *, media::VideoFrame *> *>(
                                user_data);
                            ChatView *chat_view = data->first;
                            media::VideoFrame *frame_ptr = data->second;

                            if (chat_view->enhanced_video_window_) {
                              try {
                                chat_view->enhanced_video_window_->update_frame(
                                    *frame_ptr);

                                if (!chat_view->enhanced_video_window_
                                         ->is_visible()) {
                                  std::cout << "显示摄像头视频窗口..."
                                            << std::endl;
                                  chat_view->enhanced_video_window_->show(
                                      EnhancedVideoCaptureWindow::CaptureMode::
                                          CAMERA);
                                }
                              } catch (const std::exception &e) {
                                std::cout
                                    << "更新摄像头视频帧时出错: " << e.what()
                                    << std::endl;
                              }
                            }

                            delete frame_ptr;
                            delete data;
                            return G_SOURCE_REMOVE;
                          },
                          new std::pair<ChatView *, media::VideoFrame *>(
                              this, frame_copy));
                    }
                  }
                });

            // 重新初始化以应用新的设备索引
            if (video_capture_->initialize(duorou::media::VideoSource::CAMERA,
                                           device_info.device_index) &&
                video_capture_->start_capture()) {
              is_recording_ = true;
              std::cout << "摄像头捕获已启动，新设备选择已生效" << std::endl;
            } else {
              std::cout << "启动摄像头捕获失败" << std::endl;
            }
          } else if (device_info.device_index == -1) {
            is_recording_ = false;
            std::cout << "摄像头已禁用，停止捕获" << std::endl;
          }
        }
      });
}

ChatView::~ChatView() {
  std::cout << "ChatView析构开始..." << std::endl;

  // 1. 首先停止所有录制活动，避免在析构过程中触发回调
  if (is_recording_) {
    std::cout << "析构时检测到录制正在进行，强制停止..." << std::endl;
    is_recording_ = false;

    // 立即停止视频和音频捕获，不等待回调
    if (video_capture_) {
      try {
        video_capture_->stop_capture();
      } catch (const std::exception &e) {
        std::cout << "析构时停止视频捕获异常: " << e.what() << std::endl;
      }
    }

    if (audio_capture_) {
      try {
        audio_capture_->stop_capture();
      } catch (const std::exception &e) {
        std::cout << "析构时停止音频捕获异常: " << e.what() << std::endl;
      }
    }
  }

  // 2. 清除视频窗口的关闭回调，避免在析构时触发
  if (enhanced_video_window_) {
    try {
      enhanced_video_window_->set_close_callback(nullptr);
      enhanced_video_window_->hide();
    } catch (const std::exception &e) {
      std::cout << "析构时处理视频窗口异常: " << e.what() << std::endl;
    }
  }

  // 3. 重置所有状态，确保资源正确清理
  try {
    reset_state();
  } catch (const std::exception &e) {
    std::cout << "析构时重置状态异常: " << e.what() << std::endl;
  }

  // 4. 清理视频显示窗口
  if (enhanced_video_window_) {
    try {
      enhanced_video_window_.reset();
    } catch (const std::exception &e) {
      std::cout << "析构时清理视频窗口异常: " << e.what() << std::endl;
    }
  }

  std::cout << "ChatView析构完成" << std::endl;
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

  // 初始化视频源选择对话框
  if (video_source_dialog_ && !video_source_dialog_->initialize()) {
    std::cerr << "Failed to initialize video source dialog" << std::endl;
    return false;
  }

  std::cout << "Chat view initialized successfully" << std::endl;
  return true;
}

void ChatView::send_message(const std::string &message) {
  if (message.empty()) {
    return;
  }

  // 添加用户消息到聊天显示
  add_message(message, true);

  // 保存用户消息到当前会话
  if (session_manager_) {
    session_manager_->add_message_to_current_session(message, true);
  }

  // 显示AI正在思考的指示器
  add_message("AI正在思考中...", false);

  // 禁用发送按钮，防止重复发送
  if (send_button_) {
    gtk_widget_set_sensitive(send_button_, FALSE);
  }
  if (input_entry_) {
    gtk_widget_set_sensitive(input_entry_, FALSE);
  }

  // 在后台线程中调用AI模型处理消息
  std::thread([this, message]() {
    std::string ai_response = generate_ai_response(message);

    // 创建数据结构来传递给主线程
    struct CallbackData {
      ChatView *chat_view;
      std::string *response;
    };

    CallbackData *data = new CallbackData{this, new std::string(ai_response)};

    // 使用 g_idle_add 在主线程中更新UI
    g_idle_add(
        [](gpointer user_data) -> gboolean {
          CallbackData *data = static_cast<CallbackData *>(user_data);
          ChatView *chat_view = data->chat_view;
          std::string *response = data->response;

          if (chat_view && response) {
            // 移除"AI正在思考中..."消息
            chat_view->remove_last_message();

            // 添加AI回复
            chat_view->add_message(*response, false);

            // 保存AI回复到当前会话
            if (chat_view->session_manager_) {
              chat_view->session_manager_->add_message_to_current_session(
                  *response, false);
            }

            // 重新启用发送按钮
            if (chat_view->send_button_) {
              gtk_widget_set_sensitive(chat_view->send_button_, TRUE);
            }
            if (chat_view->input_entry_) {
              gtk_widget_set_sensitive(chat_view->input_entry_, TRUE);
            }
          }

          // 清理内存
          delete response;
          delete data;
          return G_SOURCE_REMOVE;
        },
        data);
  }).detach();
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
  gtk_label_set_xalign(GTK_LABEL(message_label), 0.0);         // 左对齐文本

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
    gtk_css_provider_load_from_string(
        provider, "frame { background: #48bb78; color: white; border-radius: "
                  "18px; padding: 12px 16px; margin: 4px; border: none; }");
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
    gtk_css_provider_load_from_string(
        provider,
        "frame { background: #bee3f8; color: #2d3748; border: 1px solid "
        "#90cdf4; border-radius: 18px; padding: 12px 16px; margin: 4px; }");
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

void ChatView::remove_last_message() {
  if (chat_box_) {
    // 获取最后一个子组件
    GtkWidget *last_child = gtk_widget_get_last_child(chat_box_);
    if (last_child) {
      gtk_box_remove(GTK_BOX(chat_box_), last_child);
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

  // 创建模型选择器（初始为空，稍后通过update_model_selector填充）
  model_selector_ = gtk_drop_down_new_from_strings(
      (const char *[]){"No models available", NULL});
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
  gtk_entry_set_input_purpose(GTK_ENTRY(input_entry_),
                              GTK_INPUT_PURPOSE_FREE_FORM);
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
  upload_image_button_ = gtk_button_new_with_label("图");
  gtk_widget_add_css_class(upload_image_button_, "upload-button");
  gtk_widget_set_size_request(upload_image_button_, 40, 40);
  gtk_widget_set_tooltip_text(upload_image_button_, "Upload Image");

  // 创建上传文件按钮
  upload_file_button_ = gtk_button_new_with_label("文");
  gtk_widget_add_css_class(upload_file_button_, "upload-button");
  gtk_widget_set_size_request(upload_file_button_, 40, 40);
  gtk_widget_set_tooltip_text(upload_file_button_,
                              "Upload File (MD, DOC, Excel, PPT, PDF)");

  // 创建录制视频按钮图标 - 使用相对路径
  std::string icon_path_base = "src/gui/";
  // video_off_image_ =
  // gtk_picture_new_for_filename((icon_path_base +
  // "video-off.png").c_str());
  video_off_image_ =
      gtk_picture_new_for_filename((icon_path_base + "video-on.png").c_str());

  // 检查图标是否加载成功
  if (!video_off_image_ || !video_off_image_) {
    std::cout << "警告: 无法加载录制按钮图标，使用文本替代" << std::endl;
    // 如果图标加载失败，创建文本标签作为替代
    if (!video_off_image_) {
      video_off_image_ = gtk_label_new("停");
    }
    if (!video_off_image_) {
      video_off_image_ = gtk_label_new("录");
    }
  }

  // 设置图标大小
  gtk_widget_set_size_request(video_off_image_, 24, 24);
  gtk_widget_set_size_request(video_off_image_, 24, 24);

  // 创建录制视频按钮 (使用GtkToggleButton)
  video_record_button_ = gtk_toggle_button_new();
  gtk_button_set_child(GTK_BUTTON(video_record_button_),
                       video_off_image_); // 默认显示关闭状态
  gtk_widget_add_css_class(video_record_button_, "upload-button");
  gtk_widget_set_size_request(video_record_button_, 40, 40);
  gtk_widget_set_tooltip_text(video_record_button_, "开始录制视频/桌面捕获");

  // 设置toggle状态变化的回调
  g_signal_connect(video_record_button_, "toggled",
                   G_CALLBACK(on_video_record_button_toggled), this);

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
  const char *icon_path =
      "/Users/acproject/workspace/cpp_projects/duorou/src/gui/duorou01.png";
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
  g_signal_connect(upload_image_button_, "clicked",
                   G_CALLBACK(on_upload_image_button_clicked), this);

  // 连接上传文件按钮信号
  g_signal_connect(upload_file_button_, "clicked",
                   G_CALLBACK(on_upload_file_button_clicked), this);

  // 连接录制视频按钮信号
  g_signal_connect(video_record_button_, "clicked",
                   G_CALLBACK(on_video_record_button_clicked), this);

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
  const char *text_ptr =
      gtk_editable_get_text(GTK_EDITABLE(chat_view->input_entry_));
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
      if (!full_message.empty())
        full_message += "\n";
      full_message += "图片: " + std::string(g_path_get_basename(
                                     chat_view->selected_image_path_.c_str()));
    }

    // 添加文档信息
    if (has_file) {
      if (!full_message.empty())
        full_message += "\n";
      full_message += "文档: " + std::string(g_path_get_basename(
                                     chat_view->selected_file_path_.c_str()));
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
      if (!full_message.empty())
        full_message += "\n";
      full_message += "图片: " + std::string(g_path_get_basename(
                                     chat_view->selected_image_path_.c_str()));
    }

    // 添加文档信息
    if (has_file) {
      if (!full_message.empty())
        full_message += "\n";
      full_message += "文档: " + std::string(g_path_get_basename(
                                     chat_view->selected_file_path_.c_str()));
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

void ChatView::on_upload_image_button_clicked(GtkWidget *widget,
                                              gpointer user_data) {
  ChatView *chat_view = static_cast<ChatView *>(user_data);

  // 创建文件选择对话框
  GtkWidget *dialog = gtk_file_chooser_dialog_new(
      "Select Image", GTK_WINDOW(gtk_widget_get_root(widget)),
      GTK_FILE_CHOOSER_ACTION_OPEN, "_Cancel", GTK_RESPONSE_CANCEL, "_Open",
      GTK_RESPONSE_ACCEPT, NULL);

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
  g_signal_connect(dialog, "response", G_CALLBACK(on_image_dialog_response),
                   NULL);
}

void ChatView::on_upload_file_button_clicked(GtkWidget *widget,
                                             gpointer user_data) {
  ChatView *chat_view = static_cast<ChatView *>(user_data);

  // 创建文件选择对话框
  GtkWidget *dialog = gtk_file_chooser_dialog_new(
      "Select Document", GTK_WINDOW(gtk_widget_get_root(widget)),
      GTK_FILE_CHOOSER_ACTION_OPEN, "_Cancel", GTK_RESPONSE_CANCEL, "_Open",
      GTK_RESPONSE_ACCEPT, NULL);

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
  g_signal_connect(dialog, "response", G_CALLBACK(on_file_dialog_response),
                   NULL);
}

void ChatView::on_image_dialog_response(GtkDialog *dialog, gint response_id,
                                        gpointer user_data) {
  ChatView *chat_view =
      static_cast<ChatView *>(g_object_get_data(G_OBJECT(dialog), "chat_view"));

  if (response_id == GTK_RESPONSE_ACCEPT) {
    GtkFileChooser *chooser = GTK_FILE_CHOOSER(dialog);
    GFile *file = gtk_file_chooser_get_file(chooser);

    if (file) {
      char *filename = g_file_get_path(file);
      if (filename) {
        // 存储选择的图片路径，不直接发送
        chat_view->selected_image_path_ = std::string(filename);

        // 更新上传按钮的提示文本或样式来表示已选择文件
        gtk_widget_set_tooltip_text(
            chat_view->upload_image_button_,
            ("已选择图片: " + std::string(g_path_get_basename(filename)))
                .c_str());

        g_free(filename);
      }
      g_object_unref(file);
    }
  }

  gtk_window_destroy(GTK_WINDOW(dialog));
}

void ChatView::on_file_dialog_response(GtkDialog *dialog, gint response_id,
                                       gpointer user_data) {
  ChatView *chat_view =
      static_cast<ChatView *>(g_object_get_data(G_OBJECT(dialog), "chat_view"));

  if (response_id == GTK_RESPONSE_ACCEPT) {
    GtkFileChooser *chooser = GTK_FILE_CHOOSER(dialog);
    GFile *file = gtk_file_chooser_get_file(chooser);

    if (file) {
      char *filename = g_file_get_path(file);
      if (filename) {
        // 存储选择的文档路径，不直接发送
        chat_view->selected_file_path_ = std::string(filename);

        // 更新上传按钮的提示文本或样式来表示已选择文件
        gtk_widget_set_tooltip_text(
            chat_view->upload_file_button_,
            ("已选择文档: " + std::string(g_path_get_basename(filename)))
                .c_str());

        g_free(filename);
      }
      g_object_unref(file);
    }
  }

  gtk_window_destroy(GTK_WINDOW(dialog));
}

void ChatView::on_video_record_button_clicked(GtkWidget *widget,
                                              gpointer user_data) {
  ChatView *chat_view = static_cast<ChatView *>(user_data);

  // 如果按钮在录制过程中被禁用，只允许停止录制
  if (!gtk_widget_get_sensitive(widget) && !chat_view->is_recording_) {
    return;
  }

  // Toggle功能：如果正在录制则停止，否则显示选择对话框
  if (chat_view->is_recording_) {
    chat_view->stop_recording();
  } else {
    chat_view->show_video_source_dialog();
  }
}

void ChatView::on_video_record_button_toggled(GtkToggleButton *toggle_button,
                                              gpointer user_data) {
  ChatView *chat_view = static_cast<ChatView *>(user_data);

  // 防止在程序关闭时处理信号
  if (!chat_view || !chat_view->video_record_button_) {
    return;
  }

  // 防止递归调用
  if (chat_view->updating_button_state_) {
    return;
  }

  // 禁用按钮1秒，防止快速重复点击
  gtk_widget_set_sensitive(GTK_WIDGET(toggle_button), FALSE);

  // 1秒后重新启用按钮
  g_timeout_add(
      1000,
      [](gpointer data) -> gboolean {
        GtkWidget *button = GTK_WIDGET(data);
        if (button && GTK_IS_WIDGET(button)) {
          gtk_widget_set_sensitive(button, TRUE);
        }
        return G_SOURCE_REMOVE;
      },
      toggle_button);

  gboolean is_active = gtk_toggle_button_get_active(toggle_button);
  std::cout << "视频录制按钮状态变化: "
            << (is_active ? "激活(开启)" : "非激活(关闭)") << std::endl;

  if (is_active) {
    // 按钮被激活，但不直接开始录制，而是显示选择对话框
    if (!chat_view->is_recording_) {
      // 先重置按钮状态，避免在用户取消时按钮保持激活状态
      chat_view->updating_button_state_ = true;
      gtk_toggle_button_set_active(toggle_button, FALSE);
      chat_view->updating_button_state_ = false;

      // 显示视频源选择对话框
      chat_view->show_video_source_dialog();
    }
  } else {
    // Toggle按钮非激活 = 关闭状态 = 显示video-off图标
    if (chat_view->video_off_image_ &&
        GTK_IS_WIDGET(chat_view->video_off_image_)) {
      gtk_widget_set_visible(chat_view->video_off_image_, TRUE);
      gtk_button_set_child(GTK_BUTTON(toggle_button),
                           chat_view->video_off_image_);
      // 移除录制状态的CSS类
      gtk_widget_remove_css_class(GTK_WIDGET(toggle_button), "recording");
      // 确保基础样式类存在
      if (!gtk_widget_has_css_class(GTK_WIDGET(toggle_button),
                                    "upload-button")) {
        gtk_widget_add_css_class(GTK_WIDGET(toggle_button), "upload-button");
      }
      gtk_widget_set_tooltip_text(GTK_WIDGET(toggle_button),
                                  "开始录制视频/桌面捕获");
    }
    std::cout << "图标已切换为video-off（关闭状态）" << std::endl;

    if (chat_view->is_recording_) {
      chat_view->stop_recording();
    }
  }
}

void ChatView::start_desktop_capture() {
  std::cout << "Starting desktop capture..." << std::endl;

  if (is_recording_) {
    // 停止当前录制
    stop_recording();
    return;
  }

  // 防止重复初始化，设置标志位
  static bool initializing = false;
  if (initializing) {
    std::cout << "桌面捕获正在初始化中，请稍候..." << std::endl;
    return;
  }
  initializing = true;

  // 确保之前的资源已经清理
  if (video_capture_) {
    std::cout << "正在停止之前的视频捕获..." << std::endl;
    video_capture_->stop_capture();
    // 等待一段时间确保资源完全释放
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    video_capture_.reset();
    std::cout << "之前的视频捕获已停止" << std::endl;
  }
  if (audio_capture_) {
    std::cout << "正在停止之前的音频捕获..." << std::endl;
    audio_capture_->stop_capture();
    audio_capture_.reset();
    std::cout << "之前的音频捕获已停止" << std::endl;
  }

  // 在macOS上，确保ScreenCaptureKit资源完全清理
#ifdef __APPLE__
  std::cout << "正在清理macOS屏幕捕获资源..." << std::endl;
  media::cleanup_macos_screen_capture();
  // 额外等待时间确保macOS资源完全释放
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  std::cout << "macOS屏幕捕获资源清理完成" << std::endl;
#endif

  // 初始化视频捕获
  video_capture_ = std::make_unique<media::VideoCapture>();

  // 初始化音频捕获
  audio_capture_ = std::make_unique<media::AudioCapture>();

  // 设置视频帧回调 - 使用缓存机制减少闪烁
  video_capture_->set_frame_callback([this](const media::VideoFrame &frame) {
    // 静态计数器，只在开始时输出几帧信息
    static int frame_count = 0;
    frame_count++;

    if (frame_count <= 5 || frame_count % 30 == 0) { // 只输出前5帧和每30帧一次
      std::cout << "收到视频帧 #" << frame_count << ": " << frame.width << "x"
                << frame.height << std::endl;
    }

    // 检查是否需要更新视频帧（基于时间间隔）
    auto now = std::chrono::steady_clock::now();
    auto time_since_last_update =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_video_update_)
            .count();

    if (time_since_last_update >= VIDEO_UPDATE_INTERVAL_MS) {
      last_video_update_ = now;

      // 直接更新视频显示窗口，避免复杂的内存分配
      if (enhanced_video_window_) {
        // 创建帧的副本用于异步更新
        media::VideoFrame *frame_copy = new media::VideoFrame(frame);

        g_idle_add(
            [](gpointer user_data) -> gboolean {
              auto *data =
                  static_cast<std::pair<ChatView *, media::VideoFrame *> *>(
                      user_data);
              ChatView *chat_view = data->first;
              media::VideoFrame *frame_ptr = data->second;

              // 检查ChatView对象是否仍然有效
              if (chat_view && chat_view->enhanced_video_window_) {
                try {
                  chat_view->enhanced_video_window_->update_frame(*frame_ptr);

                  // 只在第一次显示时输出日志
                  if (!chat_view->enhanced_video_window_->is_visible()) {
                    std::cout << "显示视频窗口..." << std::endl;
                    chat_view->enhanced_video_window_->show(
                        EnhancedVideoCaptureWindow::CaptureMode::DESKTOP);
                  }
                } catch (const std::exception &e) {
                  std::cout << "更新视频帧时出错: " << e.what() << std::endl;
                }
              }

              delete frame_ptr;
              delete data;
              return G_SOURCE_REMOVE;
            },
            new std::pair<ChatView *, media::VideoFrame *>(this, frame_copy));
      }
    }
  });

  // 设置音频帧回调 - 使用缓存机制减少处理频率
  audio_capture_->set_frame_callback([this](const media::AudioFrame &frame) {
    // 静态计数器，只在开始时输出几帧信息
    static int audio_frame_count = 0;
    audio_frame_count++;

    if (audio_frame_count <= 3 ||
        audio_frame_count % 100 == 0) { // 只输出前3帧和每100帧一次
      std::cout << "收到音频帧 #" << audio_frame_count << ": "
                << frame.frame_count << " 采样, " << frame.sample_rate << "Hz"
                << std::endl;
    }

    // 检查是否需要处理音频帧（基于时间间隔）
    auto now = std::chrono::steady_clock::now();
    auto time_since_last_update =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_audio_update_)
            .count();

    if (time_since_last_update >= AUDIO_UPDATE_INTERVAL_MS) {
      // 缓存音频帧（保留最近的几帧）
      cached_audio_frames_.push_back(frame);

      // 限制缓存大小，只保留最近的10帧
      if (cached_audio_frames_.size() > 10) {
        cached_audio_frames_.erase(cached_audio_frames_.begin());
      }

      last_audio_update_ = now;
    }
  });

  // 初始化桌面捕获
  if (video_capture_->initialize(media::VideoSource::DESKTOP_CAPTURE)) {
    if (video_capture_->start_capture()) {
      // 初始化麦克风音频捕获
      if (audio_capture_->initialize(media::AudioSource::MICROPHONE)) {
        if (audio_capture_->start_capture()) {
          is_recording_ = true;

          // 只更新按钮状态，图标由toggle回调处理
          if (video_record_button_) {
            // 只在按钮未激活时才设置为激活状态，避免递归
            if (!gtk_toggle_button_get_active(
                    GTK_TOGGLE_BUTTON(video_record_button_))) {
              gtk_toggle_button_set_active(
                  GTK_TOGGLE_BUTTON(video_record_button_), TRUE);
            }
            std::cout << "按钮状态已切换为激活状态" << std::endl;
          }

          std::cout << "桌面录制已开始 - 正在捕获桌面视频和麦克风音频"
                    << std::endl;

          // 重置初始化标志
          initializing = false;
        } else {
          std::cout << "音频捕获启动失败" << std::endl;
          // 重置按钮状态，使用标志防止递归调用
          if (video_record_button_) {
            updating_button_state_ = true;
            gtk_toggle_button_set_active(
                GTK_TOGGLE_BUTTON(video_record_button_), FALSE);
            // 直接更新图标为关闭状态
            if (video_off_image_) {
              gtk_button_set_child(GTK_BUTTON(video_record_button_),
                                   video_off_image_);
              gtk_widget_set_tooltip_text(GTK_WIDGET(video_record_button_),
                                          "开始录制视频/桌面捕获");
            }
            updating_button_state_ = false;
          }
          initializing = false;
        }
      } else {
        std::cout << "音频捕获初始化失败" << std::endl;
        // 重置按钮状态，使用标志防止递归调用
        if (video_record_button_) {
          updating_button_state_ = true;
          gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(video_record_button_),
                                       FALSE);
          // 直接更新图标为关闭状态
          if (video_off_image_) {
            gtk_button_set_child(GTK_BUTTON(video_record_button_),
                                 video_off_image_);
            gtk_widget_set_tooltip_text(GTK_WIDGET(video_record_button_),
                                        "开始录制视频/桌面捕获");
          }
          updating_button_state_ = false;
        }
        initializing = false;
      }
    } else {
      std::cout << "视频捕获启动失败" << std::endl;
      // 重置按钮状态，使用标志防止递归调用
      if (video_record_button_) {
        updating_button_state_ = true;
        gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(video_record_button_),
                                     FALSE);
        // 直接更新图标为关闭状态
        if (video_off_image_) {
          gtk_button_set_child(GTK_BUTTON(video_record_button_),
                               video_off_image_);
          gtk_widget_set_tooltip_text(GTK_WIDGET(video_record_button_),
                                      "开始录制视频/桌面捕获");
        }
        updating_button_state_ = false;
      }
      initializing = false;
    }
  } else {
    std::cout << "视频捕获初始化失败" << std::endl;

    // 重置按钮状态，图标由toggle回调处理
    if (video_record_button_) {
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(video_record_button_),
                                   FALSE);
    }

    // 显示错误信息
    GtkWidget *dialog = gtk_message_dialog_new(
        GTK_WINDOW(gtk_widget_get_root(main_widget_)), GTK_DIALOG_MODAL,
        GTK_MESSAGE_ERROR, GTK_BUTTONS_OK,
        "桌面捕获初始化失败\n\n请检查系统权限设置。");

    // 确保对话框在最上层
    gtk_window_set_modal(GTK_WINDOW(dialog), TRUE);
    gtk_window_present(GTK_WINDOW(dialog));
    gtk_widget_show(dialog);
    g_signal_connect(dialog, "response", G_CALLBACK(gtk_window_destroy), NULL);

    // 重置初始化标志
    initializing = false;
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
        GTK_WINDOW(gtk_widget_get_root(main_widget_)), GTK_DIALOG_MODAL,
        GTK_MESSAGE_WARNING, GTK_BUTTONS_NONE,
        "未检测到可用的摄像头设备\n\n是否使用桌面捕获作为替代？");

    gtk_dialog_add_button(GTK_DIALOG(dialog), "使用桌面捕获", GTK_RESPONSE_YES);
    gtk_dialog_add_button(GTK_DIALOG(dialog), "取消", GTK_RESPONSE_NO);

    // 确保对话框在最上层
    gtk_window_set_modal(GTK_WINDOW(dialog), TRUE);
    gtk_window_present(GTK_WINDOW(dialog));
    gtk_widget_show(dialog);
    g_signal_connect(dialog, "response",
                     G_CALLBACK(+[](GtkDialog *dialog, gint response_id,
                                    gpointer user_data) {
                       ChatView *chat_view = static_cast<ChatView *>(user_data);

                       if (response_id == GTK_RESPONSE_YES) {
                         chat_view->start_desktop_capture();
                       }

                       gtk_window_destroy(GTK_WINDOW(dialog));
                     }),
                     this);
    return;
  }

  // 确保之前的资源已经清理
  if (video_capture_) {
    std::cout << "正在停止之前的视频捕获..." << std::endl;
    video_capture_->stop_capture();
    // 等待一段时间确保资源完全释放
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    video_capture_.reset();
    std::cout << "之前的视频捕获已停止" << std::endl;
  }
  if (audio_capture_) {
    std::cout << "正在停止之前的音频捕获..." << std::endl;
    audio_capture_->stop_capture();
    audio_capture_.reset();
    std::cout << "之前的音频捕获已停止" << std::endl;
  }

  // 在macOS上，确保ScreenCaptureKit资源完全清理
#ifdef __APPLE__
  std::cout << "正在清理macOS屏幕捕获资源..." << std::endl;
  media::cleanup_macos_screen_capture();
  // 额外等待时间确保macOS资源完全释放
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  std::cout << "macOS屏幕捕获资源清理完成" << std::endl;
#endif

  // 初始化视频捕获
  video_capture_ = std::make_unique<media::VideoCapture>();

  // 初始化音频捕获
  audio_capture_ = std::make_unique<media::AudioCapture>();

  // 设置视频帧回调 - 使用缓存机制减少闪烁
  video_capture_->set_frame_callback([this](const media::VideoFrame &frame) {
    // 静态计数器和标志，减少日志输出
    static int camera_frame_count = 0;
    static bool window_shown_logged = false;
    camera_frame_count++;

    if (camera_frame_count <= 5 ||
        camera_frame_count % 30 == 0) { // 只输出前5帧和每30帧一次
      std::cout << "收到摄像头视频帧 #" << camera_frame_count << ": "
                << frame.width << "x" << frame.height << std::endl;
    }

    // 检查是否需要更新视频帧（基于时间间隔）
    auto now = std::chrono::steady_clock::now();
    auto time_since_last_update =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_video_update_)
            .count();

    if (time_since_last_update >= VIDEO_UPDATE_INTERVAL_MS) {
      last_video_update_ = now;

      // 直接更新视频显示窗口，避免复杂的内存分配
      if (enhanced_video_window_) {
        // 创建帧的副本用于异步更新
        media::VideoFrame *frame_copy = new media::VideoFrame(frame);

        g_idle_add(
            [](gpointer user_data) -> gboolean {
              auto *data =
                  static_cast<std::pair<ChatView *, media::VideoFrame *> *>(
                      user_data);
              ChatView *chat_view = data->first;
              media::VideoFrame *frame_ptr = data->second;

              if (chat_view->enhanced_video_window_) {
                try {
                  chat_view->enhanced_video_window_->update_frame(*frame_ptr);

                  // 只在第一次显示时输出日志
                  if (!chat_view->enhanced_video_window_->is_visible()) {
                    std::cout << "显示摄像头视频窗口..." << std::endl;
                    chat_view->enhanced_video_window_->show(
                        EnhancedVideoCaptureWindow::CaptureMode::CAMERA);
                  }
                } catch (const std::exception &e) {
                  std::cout << "更新摄像头视频帧时出错: " << e.what()
                            << std::endl;
                }
              }

              delete frame_ptr;
              delete data;
              return G_SOURCE_REMOVE;
            },
            new std::pair<ChatView *, media::VideoFrame *>(this, frame_copy));
      }
    }
  });

  // 设置音频帧回调 - 使用缓存机制减少处理频率
  audio_capture_->set_frame_callback([this](const media::AudioFrame &frame) {
    // 静态计数器，只在开始时输出几帧信息
    static int camera_audio_frame_count = 0;
    camera_audio_frame_count++;

    if (camera_audio_frame_count <= 3 ||
        camera_audio_frame_count % 100 == 0) { // 只输出前3帧和每100帧一次
      std::cout << "收到摄像头音频帧 #" << camera_audio_frame_count << ": "
                << frame.frame_count << " 采样, " << frame.sample_rate << "Hz"
                << std::endl;
    }

    // 检查是否需要处理音频帧（基于时间间隔）
    auto now = std::chrono::steady_clock::now();
    auto time_since_last_update =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_audio_update_)
            .count();

    if (time_since_last_update >= AUDIO_UPDATE_INTERVAL_MS) {
      // 缓存音频帧（保留最近的几帧）
      cached_audio_frames_.push_back(frame);

      // 限制缓存大小，只保留最近的10帧
      if (cached_audio_frames_.size() > 10) {
        cached_audio_frames_.erase(cached_audio_frames_.begin());
      }

      last_audio_update_ = now;
    }
  });

  // 初始化摄像头捕获
  if (video_capture_->initialize(media::VideoSource::CAMERA, 0)) {
    if (video_capture_->start_capture()) {
      // 初始化麦克风音频捕获
      if (audio_capture_->initialize(media::AudioSource::MICROPHONE)) {
        if (audio_capture_->start_capture()) {
          is_recording_ = true;

          // 只更新按钮状态，图标由toggle回调处理
          if (video_record_button_) {
            // 只在按钮未激活时才设置为激活状态，避免递归
            if (!gtk_toggle_button_get_active(
                    GTK_TOGGLE_BUTTON(video_record_button_))) {
              gtk_toggle_button_set_active(
                  GTK_TOGGLE_BUTTON(video_record_button_), TRUE);
            }
          }

          std::cout << "摄像头录制已开始 - 正在捕获摄像头视频和麦克风音频"
                    << std::endl;
        } else {
          std::cout << "音频捕获启动失败" << std::endl;
          // 重置按钮状态，使用标志防止递归调用
          if (video_record_button_) {
            updating_button_state_ = true;
            gtk_toggle_button_set_active(
                GTK_TOGGLE_BUTTON(video_record_button_), FALSE);
            // 直接更新图标为关闭状态
            if (video_off_image_) {
              gtk_button_set_child(GTK_BUTTON(video_record_button_),
                                   video_off_image_);
              gtk_widget_set_tooltip_text(GTK_WIDGET(video_record_button_),
                                          "开始录制视频/桌面捕获");
            }
            updating_button_state_ = false;
          }
        }
      } else {
        std::cout << "音频捕获初始化失败" << std::endl;
        // 重置按钮状态，使用标志防止递归调用
        if (video_record_button_) {
          updating_button_state_ = true;
          gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(video_record_button_),
                                       FALSE);
          // 直接更新图标为关闭状态
          if (video_off_image_) {
            gtk_button_set_child(GTK_BUTTON(video_record_button_),
                                 video_off_image_);
            gtk_widget_set_tooltip_text(GTK_WIDGET(video_record_button_),
                                        "开始录制视频/桌面捕获");
          }
          updating_button_state_ = false;
        }
      }
    } else {
      std::cout << "摄像头捕获启动失败" << std::endl;
      // 重置按钮状态，使用标志防止递归调用
      if (video_record_button_) {
        updating_button_state_ = true;
        gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(video_record_button_),
                                     FALSE);
        // 直接更新图标为关闭状态
        if (video_off_image_) {
          gtk_button_set_child(GTK_BUTTON(video_record_button_),
                               video_off_image_);
          gtk_widget_set_tooltip_text(GTK_WIDGET(video_record_button_),
                                      "开始录制视频/桌面捕获");
        }
        updating_button_state_ = false;
      }
    }
  } else {
    std::cout << "摄像头捕获初始化失败" << std::endl;

    // 重置按钮状态，使用标志防止递归调用
    if (video_record_button_) {
      updating_button_state_ = true;
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(video_record_button_),
                                   FALSE);
      // 直接更新图标为关闭状态
      if (video_off_image_) {
        gtk_button_set_child(GTK_BUTTON(video_record_button_),
                             video_off_image_);
        gtk_widget_set_tooltip_text(GTK_WIDGET(video_record_button_),
                                    "开始录制视频/桌面捕获");
      }
      updating_button_state_ = false;
    }

    // 显示错误信息
    GtkWidget *dialog = gtk_message_dialog_new(
        GTK_WINDOW(gtk_widget_get_root(main_widget_)), GTK_DIALOG_MODAL,
        GTK_MESSAGE_ERROR, GTK_BUTTONS_OK,
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
    std::cout << "Recording is not active, skipping stop operation"
              << std::endl;
    return;
  }

  // 防止重复调用
  static bool stopping = false;
  if (stopping) {
    std::cout << "Stop recording already in progress, skipping" << std::endl;
    return;
  }
  stopping = true;

  // 先设置录制状态为false
  is_recording_ = false;

  // 停止视频捕获
  if (video_capture_) {
    try {
      std::cout << "正在停止视频捕获..." << std::endl;
      video_capture_->stop_capture();
      // 等待一小段时间，确保所有待处理的回调完成
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      video_capture_.reset(); // 重置视频捕获对象
      std::cout << "视频捕获已停止" << std::endl;
    } catch (const std::exception &e) {
      std::cout << "Error stopping video capture: " << e.what() << std::endl;
    }
  }

  // 停止音频捕获
  if (audio_capture_) {
    try {
      std::cout << "正在停止音频捕获..." << std::endl;
      audio_capture_->stop_capture();
      audio_capture_.reset(); // 重置音频捕获对象
      std::cout << "音频捕获已停止" << std::endl;
    } catch (const std::exception &e) {
      std::cout << "Error stopping audio capture: " << e.what() << std::endl;
    }
  }

  // 在macOS上，确保ScreenCaptureKit资源完全清理
#ifdef __APPLE__
  try {
    std::cout << "正在清理macOS屏幕捕获资源..." << std::endl;
    media::cleanup_macos_screen_capture();
    // 额外等待时间确保macOS资源完全释放
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    std::cout << "macOS屏幕捕获资源清理完成" << std::endl;
  } catch (const std::exception &e) {
    std::cout << "Error cleaning up macOS screen capture: " << e.what()
              << std::endl;
  }
#endif

  // 更新按钮状态和图标
  if (video_record_button_) {
    // 设置标志防止递归调用
    updating_button_state_ = true;

    // 强制设置按钮为非激活状态
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(video_record_button_),
                                 FALSE);

    // 确保图标对象有效并重新设置
    if (video_off_image_ && GTK_IS_WIDGET(video_off_image_)) {
      // 确保图标可见
      gtk_widget_set_visible(video_off_image_, TRUE);
      gtk_button_set_child(GTK_BUTTON(video_record_button_), video_off_image_);

      // 重新应用CSS类确保样式正确
      gtk_widget_remove_css_class(video_record_button_, "recording");
      if (!gtk_widget_has_css_class(video_record_button_, "upload-button")) {
        gtk_widget_add_css_class(video_record_button_, "upload-button");
      }

      gtk_widget_set_tooltip_text(GTK_WIDGET(video_record_button_),
                                  "开始录制视频/桌面捕获");
    } else {
      // 如果图标对象无效，重新创建
      std::cout << "警告: video_off_image_无效，重新创建图标" << std::endl;
      std::string icon_path_base = "src/gui/";
      video_off_image_ = gtk_picture_new_for_filename(
          (icon_path_base + "video-off.png").c_str());
      if (!video_off_image_) {
        video_off_image_ = gtk_label_new("停");
      }
      gtk_widget_set_size_request(video_off_image_, 24, 24);
      gtk_widget_set_visible(video_off_image_, TRUE);
      gtk_button_set_child(GTK_BUTTON(video_record_button_), video_off_image_);

      // 重新应用CSS类
      gtk_widget_remove_css_class(video_record_button_, "recording");
      if (!gtk_widget_has_css_class(video_record_button_, "upload-button")) {
        gtk_widget_add_css_class(video_record_button_, "upload-button");
      }

      gtk_widget_set_tooltip_text(GTK_WIDGET(video_record_button_),
                                  "开始录制视频/桌面捕获");
    }

    // 重新启用按钮
    gtk_widget_set_sensitive(video_record_button_, TRUE);
    updating_button_state_ = false;
    std::cout
        << "按钮状态已切换为非激活状态，图标已更新为video-off，按钮已重新启用"
        << std::endl;
  }

  // 隐藏视频显示窗口
  if (enhanced_video_window_) {
    try {
      enhanced_video_window_->hide();
    } catch (const std::exception &e) {
      std::cout << "Error hiding video window: " << e.what() << std::endl;
    }
  }

  std::cout << "录制已停止 - 视频和音频捕获已结束" << std::endl;

  // 重置停止标志
  stopping = false;

  // 验证状态同步
  verify_button_state();
}

void ChatView::verify_button_state() {
  if (!video_record_button_) {
    return;
  }

  gboolean button_active =
      gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(video_record_button_));

  // 检查按钮状态与实际录制状态是否一致
  if (button_active != is_recording_) {
    std::cout << "状态不一致检测到: 按钮状态="
              << (button_active ? "激活(开启)" : "非激活(关闭)")
              << ", 录制状态=" << (is_recording_ ? "录制中" : "已停止")
              << std::endl;

    // 同步按钮状态到实际录制状态
    updating_button_state_ = true;
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(video_record_button_),
                                 is_recording_);

    // 根据toggle状态更新图标：激活=video-on，非激活=video-off
    if (is_recording_) {
      // 录制中，按钮应该是激活状态，显示video-on图标
      if (video_off_image_) {
        gtk_button_set_child(GTK_BUTTON(video_record_button_),
                             video_off_image_);
        gtk_widget_set_tooltip_text(GTK_WIDGET(video_record_button_),
                                    "停止录制视频/桌面捕获");
      }
      std::cout << "同步：设置为激活状态，显示video-on图标" << std::endl;
    } else {
      // 未录制，按钮应该是非激活状态，显示video-off图标
      if (video_off_image_) {
        gtk_button_set_child(GTK_BUTTON(video_record_button_),
                             video_off_image_);
        gtk_widget_set_tooltip_text(GTK_WIDGET(video_record_button_),
                                    "开始录制视频/桌面捕获");
      }
      std::cout << "同步：设置为非激活状态，显示video-off图标" << std::endl;
    }

    updating_button_state_ = false;
  }
}

void ChatView::show_video_source_dialog() {
  std::cout << "show_video_source_dialog() called" << std::endl;

  if (!video_source_dialog_) {
    std::cerr << "Video source dialog not initialized" << std::endl;
    return;
  }

  std::cout << "Showing video source dialog..." << std::endl;

  // 显示对话框，传递回调函数
  video_source_dialog_->show(main_widget_,
                             [this](VideoSourceDialog::VideoSource source) {
                               on_video_source_selected(source);
                             });
}

void ChatView::on_video_source_selected(VideoSourceDialog::VideoSource source) {
  switch (source) {
  case VideoSourceDialog::VideoSource::DESKTOP_CAPTURE:
    std::cout << "用户选择：桌面录制" << std::endl;
    // 激活按钮并更新图标
    if (video_record_button_) {
      updating_button_state_ = true;
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(video_record_button_),
                                   TRUE);
      if (video_off_image_ && GTK_IS_WIDGET(video_off_image_)) {
        gtk_widget_set_visible(video_off_image_, TRUE);
        gtk_button_set_child(GTK_BUTTON(video_record_button_),
                             video_off_image_);
        // 添加录制状态的CSS类
        gtk_widget_add_css_class(video_record_button_, "recording");
        gtk_widget_set_tooltip_text(GTK_WIDGET(video_record_button_),
                                    "停止录制");
      }
      // 禁用按钮，防止在录制过程中被点击
      gtk_widget_set_sensitive(video_record_button_, FALSE);
      updating_button_state_ = false;
    }
    start_desktop_capture();
    break;
  case VideoSourceDialog::VideoSource::CAMERA:
    std::cout << "用户选择：摄像头" << std::endl;
    // 激活按钮并更新图标
    if (video_record_button_) {
      updating_button_state_ = true;
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(video_record_button_),
                                   TRUE);
      if (video_off_image_ && GTK_IS_WIDGET(video_off_image_)) {
        gtk_widget_set_visible(video_off_image_, TRUE);
        gtk_button_set_child(GTK_BUTTON(video_record_button_),
                             video_off_image_);
        // 添加录制状态的CSS类
        gtk_widget_add_css_class(video_record_button_, "recording");
        gtk_widget_set_tooltip_text(GTK_WIDGET(video_record_button_),
                                    "停止录制");
      }
      // 禁用按钮，防止在录制过程中被点击
      gtk_widget_set_sensitive(video_record_button_, FALSE);
      updating_button_state_ = false;
    }
    start_camera_capture();
    break;
  case VideoSourceDialog::VideoSource::CANCEL:
    std::cout << "用户取消选择" << std::endl;
    // 重置按钮状态
    if (video_record_button_) {
      updating_button_state_ = true;
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(video_record_button_),
                                   FALSE);
      if (video_off_image_) {
        gtk_button_set_child(GTK_BUTTON(video_record_button_),
                             video_off_image_);
        gtk_widget_set_tooltip_text(GTK_WIDGET(video_record_button_),
                                    "开始录制视频/桌面捕获");
      }
      updating_button_state_ = false;
    }
    break;
  }
}

void ChatView::reset_state() {
  std::cout << "开始重置ChatView状态..." << std::endl;

  // 直接停止录制活动，不调用stop_recording避免重复清理
  if (is_recording_) {
    is_recording_ = false;

    // 直接停止视频和音频捕获，不调用macOS清理函数
    if (video_capture_) {
      try {
        video_capture_->stop_capture();
      } catch (const std::exception &e) {
        std::cout << "重置状态时停止视频捕获异常: " << e.what() << std::endl;
      }
    }

    if (audio_capture_) {
      try {
        audio_capture_->stop_capture();
      } catch (const std::exception &e) {
        std::cout << "重置状态时停止音频捕获异常: " << e.what() << std::endl;
      }
    }
  }

  // 重置录制状态
  is_recording_ = false;
  updating_button_state_ = false;

  // 清理视频捕获
  if (video_capture_) {
    video_capture_.reset();
  }

  // 清理音频捕获
  if (audio_capture_) {
    audio_capture_.reset();
  }

  // 重置按钮状态
  if (video_record_button_) {
    updating_button_state_ = true;
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(video_record_button_),
                                 FALSE);
    gtk_widget_set_sensitive(video_record_button_, TRUE);

    // 重置按钮图标为关闭状态
    if (video_off_image_ && GTK_IS_WIDGET(video_off_image_)) {
      gtk_widget_set_visible(video_off_image_, TRUE);
      gtk_button_set_child(GTK_BUTTON(video_record_button_), video_off_image_);
      gtk_widget_remove_css_class(video_record_button_, "recording");
      gtk_widget_set_tooltip_text(GTK_WIDGET(video_record_button_), "开始录制");
    }
    updating_button_state_ = false;
  }

  // 隐藏视频显示窗口
  if (enhanced_video_window_) {
    try {
      enhanced_video_window_->hide();
    } catch (const std::exception &e) {
      std::cout << "隐藏视频窗口时出错: " << e.what() << std::endl;
    }
  }

  // 清理缓存的帧数据
  cached_video_frame_.reset();
  cached_audio_frames_.clear();

  // 重置时间戳
  last_video_update_ = std::chrono::steady_clock::now();
  last_audio_update_ = std::chrono::steady_clock::now();

  std::cout << "ChatView状态重置完成" << std::endl;
}

void ChatView::set_session_manager(ChatSessionManager *session_manager) {
  session_manager_ = session_manager;
}

void ChatView::load_session_messages(const std::string &session_id) {
  if (!session_manager_) {
    std::cerr << "Session manager not set" << std::endl;
    return;
  }

  // 清空当前聊天历史
  clear_chat();

  // 获取指定会话的消息
  auto session = session_manager_->get_session(session_id);
  if (!session) {
    std::cout << "Session not found: " << session_id << std::endl;
    return;
  }

  // 加载会话中的所有消息
  for (const auto &message : session->get_messages()) {
    add_message(message.content, message.is_user);
  }

  std::cout << "Loaded " << session->get_messages().size()
            << " messages for session: " << session_id << std::endl;
}

void ChatView::set_model_manager(core::ModelManager *model_manager) {
  model_manager_ = model_manager;
  // 设置模型管理器后立即更新模型选择器
  update_model_selector();
}

void ChatView::update_model_selector() {
  if (!model_manager_ || !model_selector_) {
    return;
  }

  // 获取可用模型列表
  auto available_models = model_manager_->getAllModels();

  if (available_models.empty()) {
    // 如果没有可用模型，显示提示信息
    const char *no_models[] = {"No models available", NULL};
    GtkStringList *string_list = gtk_string_list_new(no_models);
    gtk_drop_down_set_model(GTK_DROP_DOWN(model_selector_),
                            G_LIST_MODEL(string_list));
    return;
  }

  // 创建模型名称数组
  std::vector<const char *> model_names;
  for (const auto &model : available_models) {
    model_names.push_back(model.name.c_str());
  }
  model_names.push_back(NULL); // 以NULL结尾

  // 更新下拉菜单
  GtkStringList *string_list = gtk_string_list_new(model_names.data());
  gtk_drop_down_set_model(GTK_DROP_DOWN(model_selector_),
                          G_LIST_MODEL(string_list));

  // 默认选择第一个模型
  if (!available_models.empty()) {
    gtk_drop_down_set_selected(GTK_DROP_DOWN(model_selector_), 0);
  }

  std::cout << "Updated model selector with " << available_models.size()
            << " models" << std::endl;
}

std::string ChatView::generate_ai_response(const std::string &message) {
  std::cout << "[DEBUG] ChatView::generate_ai_response() called with message: "
            << message.substr(0, 50) << "..." << std::endl;

  if (!model_manager_) {
    std::cout << "[DEBUG] ChatView: Model manager not available" << std::endl;
    return "Error: Model manager not available.";
  }
  std::cout << "[DEBUG] ChatView: Model manager is available" << std::endl;

  // 获取当前选中的模型
  if (!model_selector_) {
    std::cout << "[DEBUG] ChatView: Model selector not available" << std::endl;
    return "Error: Model selector not available.";
  }
  std::cout << "[DEBUG] ChatView: Model selector is available" << std::endl;

  // 获取选中的模型索引
  guint selected_index =
      gtk_drop_down_get_selected(GTK_DROP_DOWN(model_selector_));
  std::cout << "[DEBUG] ChatView: Selected model index: " << selected_index
            << std::endl;

  // 获取可用模型列表
  auto available_models = model_manager_->getAllModels();
  std::cout << "[DEBUG] ChatView: Available models count: "
            << available_models.size() << std::endl;

  if (available_models.empty()) {
    std::cout << "[DEBUG] ChatView: No models available" << std::endl;
    return "Error: No models available for text generation.";
  }

  if (selected_index >= available_models.size()) {
    std::cout << "[DEBUG] ChatView: Invalid model selection - index "
              << selected_index << " >= " << available_models.size()
              << std::endl;
    return "Error: Invalid model selection.";
  }

  // 获取选中的模型
  const auto &selected_model = available_models[selected_index];
  std::string model_id = selected_model.name;
  std::cout << "[DEBUG] ChatView: Selected model ID: " << model_id << std::endl;

  try {
    // 首先尝试加载模型
    std::cout << "[DEBUG] ChatView: Attempting to load model: " << model_id
              << std::endl;
    bool model_loaded = model_manager_->loadModel(model_id);
    if (!model_loaded) {
      std::cout << "[DEBUG] ChatView: Failed to load model: " << model_id
                << std::endl;
      return "Error: Failed to load model: " + model_id;
    }
    std::cout << "[DEBUG] ChatView: Model loaded successfully: " << model_id
              << std::endl;

    // 获取文本生成器
    std::cout << "[DEBUG] ChatView: Getting text generator for model: "
              << model_id << std::endl;
    core::TextGenerator *text_generator =
        model_manager_->getTextGenerator(model_id);
    if (!text_generator) {
      std::cout << "[DEBUG] ChatView: Failed to get text generator for model: "
                << model_id << std::endl;
      return "Error: Failed to get text generator for model: " + model_id;
    }

    // 检查文本生成器是否可用
    if (!text_generator->canGenerate()) {
      std::cout
          << "[DEBUG] ChatView: Text generator is not ready for generation"
          << std::endl;
      return "Error: Text generator is not ready for generation";
    }

    // 设置生成参数
    core::GenerationParams params;
    params.max_tokens = 512;   // 最大生成512个token
    params.temperature = 0.7f; // 适中的随机性
    params.top_p = 0.9f;
    params.top_k = 40;
    params.repeat_penalty = 1.1f;

    // 生成回复
    std::cout << "[DEBUG] ChatView: Starting text generation..." << std::endl;
    core::GenerationResult result = text_generator->generate(message, params);

    if (result.finished && !result.text.empty()) {
      std::cout << "[DEBUG] ChatView: Text generation completed successfully"
                << std::endl;
      return result.text;
    } else {
      std::cout
          << "[DEBUG] ChatView: Text generation failed or returned empty result"
          << std::endl;
      return "Error: Text generation failed or returned empty result. Stop "
             "reason: " +
             result.stop_reason;
    }
  } catch (const std::exception &e) {
    std::cout << "[DEBUG] ChatView: Exception caught: " << e.what()
              << std::endl;
    return "Error generating response: " + std::string(e.what());
  }
}

} // namespace gui
} // namespace duorou