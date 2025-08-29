#ifndef DUOROU_GUI_CHAT_VIEW_H
#define DUOROU_GUI_CHAT_VIEW_H

#include "../media/audio_capture.h"
#include "../media/video_frame.h"
#include "enhanced_video_capture_window.h"
#include "video_source_dialog.h"
#include <chrono>
#include <gtk/gtk.h>
#include <memory>
#include <string>
#include <vector>

// 前向声明
namespace duorou {
namespace media {
class VideoCapture;
class AudioCapture;
} // namespace media
} // namespace duorou

namespace duorou {
namespace gui {

class ChatSessionManager;

/**
 * 聊天视图类 - 处理文本生成模型的交互界面
 */
class ChatView {
public:
  ChatView();
  ~ChatView();

  // 禁用拷贝构造和赋值
  ChatView(const ChatView &) = delete;
  ChatView &operator=(const ChatView &) = delete;

  /**
   * 初始化聊天视图
   * @return 成功返回true，失败返回false
   */
  bool initialize();

  /**
   * 获取主要的GTK组件
   * @return GTK组件指针
   */
  GtkWidget *get_widget() const { return main_widget_; }

  /**
   * 发送消息
   * @param message 用户输入的消息
   */
  void send_message(const std::string &message);

  /**
   * 添加消息到聊天历史
   * @param message 消息内容
   * @param is_user 是否为用户消息
   */
  void add_message(const std::string &message, bool is_user);

  /**
   * 清空聊天历史
   */
  void clear_chat();

  /**
   * 设置当前会话管理器
   * @param session_manager 会话管理器指针
   */
  void set_session_manager(ChatSessionManager* session_manager);

  /**
   * 加载并显示指定会话的消息
   * @param session_id 会话ID
   */
  void load_session_messages(const std::string& session_id);

private:
  GtkWidget *main_widget_;         // 主容器
  GtkWidget *chat_scrolled_;       // 滚动窗口
  GtkWidget *chat_box_;            // 聊天消息容器
  GtkWidget *input_box_;           // 输入区域容器
  GtkWidget *input_entry_;         // 输入框
  GtkWidget *send_button_;         // 发送按钮
  GtkWidget *upload_image_button_; // 上传图片按钮
  GtkWidget *upload_file_button_;  // 上传文件按钮
  GtkWidget *video_record_button_; // 录制视频按钮 (GtkToggleButton)

  // 存储选择的文件路径
  std::string selected_image_path_;
  std::string selected_file_path_;
  GtkWidget *model_selector_;  // 模型选择器
  GtkWidget *input_container_; // 输入框容器

  bool welcome_cleared_; // 标记是否已清除欢迎界面

  // 媒体捕获相关
  std::unique_ptr<media::VideoCapture> video_capture_;
  std::unique_ptr<media::AudioCapture> audio_capture_;
  std::unique_ptr<EnhancedVideoCaptureWindow> enhanced_video_window_;
  std::unique_ptr<VideoSourceDialog> video_source_dialog_;
  bool is_recording_; // 录制状态标记

  // 录制按钮图标
  GtkWidget *video_off_image_; // 录制关闭图标
  GtkWidget *video_on_image_;  // 录制开启图标

  // 防止递归调用的标志
  bool updating_button_state_; // 标记是否正在更新按钮状态

  // 会话管理器
  ChatSessionManager* session_manager_; // 会话管理器指针

  // 视频帧缓存相关
  std::shared_ptr<duorou::media::VideoFrame>
      cached_video_frame_;                                  // 缓存的视频帧
  std::chrono::steady_clock::time_point last_video_update_; // 上次视频更新时间
  static constexpr int VIDEO_UPDATE_INTERVAL_MS =
      66; // 视频更新间隔(约15fps，减少闪烁)

  // 音频帧缓存相关
  std::vector<duorou::media::AudioFrame> cached_audio_frames_; // 缓存的音频帧
  std::chrono::steady_clock::time_point last_audio_update_; // 上次音频更新时间
  static constexpr int AUDIO_UPDATE_INTERVAL_MS = 100;      // 音频更新间隔

  /**
   * 创建聊天显示区域
   */
  void create_chat_area();

  /**
   * 创建欢迎界面
   */
  void create_welcome_screen();

  /**
   * 创建输入区域
   */
  void create_input_area();

  /**
   * 连接信号
   */
  void connect_signals();

  /**
   * 滚动到底部
   */
  void scroll_to_bottom();

  // 静态回调函数
  static void on_send_button_clicked(GtkWidget *widget, gpointer user_data);
  static void on_upload_image_button_clicked(GtkWidget *widget,
                                             gpointer user_data);
  static void on_upload_file_button_clicked(GtkWidget *widget,
                                            gpointer user_data);
  static void on_video_record_button_clicked(GtkWidget *widget,
                                             gpointer user_data);
  static void on_video_record_button_toggled(GtkToggleButton *toggle_button,
                                             gpointer user_data);
  static void on_input_entry_activate(GtkWidget *widget, gpointer user_data);
  static void on_image_dialog_response(GtkDialog *dialog, gint response_id,
                                       gpointer user_data);
  static void on_file_dialog_response(GtkDialog *dialog, gint response_id,
                                      gpointer user_data);

  // 视频捕获方法
  void show_video_source_dialog();
  void start_desktop_capture();
  void start_camera_capture();
  void stop_recording();

  // 视频源选择回调
  void on_video_source_selected(VideoSourceDialog::VideoSource source);

  // 状态管理方法
  void verify_button_state();

  /**
   * 重置所有状态，防止段错误
   * 清理录制状态、按钮状态等
   */
  void reset_state();

private:
};

} // namespace gui
} // namespace duorou

#endif // DUOROU_GUI_CHAT_VIEW_H