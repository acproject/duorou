#ifndef DUOROU_GUI_CHAT_VIEW_H
#define DUOROU_GUI_CHAT_VIEW_H

#include "../media/audio_capture.h"
#include "../media/video_frame.h"
#include "enhanced_video_capture_window.h"
#include "video_source_dialog.h"
#include "../core/model_manager.h"
#include <chrono>
#include <gtk/gtk.h>
#include <memory>
#include <string>
#include <vector>

namespace duorou {
  namespace media {
    class VideoCapture;
    class AudioCapture;
  } // namespace media
  namespace core {
    class ModelManager;
    class ConfigManager; // 新增：前向声明 ConfigManager
  } // namespace core
} // namespace duorou

namespace duorou {
  namespace gui {

class ChatSessionManager;

/**
 * Chat view class - handles text generation model interaction interface
 */
class ChatView {
public:
  ChatView();
  ~ChatView();

  // Disable copy constructor and assignment
  ChatView(const ChatView &) = delete;
  ChatView &operator=(const ChatView &) = delete;

  /**
   * Initialize chat view
   * @return true on success, false on failure
   */
  bool initialize();

  /**
   * Get main GTK widget
   * @return GTK widget pointer
   */
  GtkWidget *get_widget() const { return main_widget_; }

  /**
   * Send message
   * @param message user input message
   */
  void send_message(const std::string &message);

  /**
   * Add message to chat history
   * @param message message content
   * @param is_user whether it's a user message
   */
  void add_message(const std::string &message, bool is_user);

  /**
   * Remove last message
   */
  void remove_last_message();

  /**
   * Clear chat history
   */
  void clear_chat();

  /**
   * Set current session manager
   * @param session_manager session manager pointer
   */
  void set_session_manager(ChatSessionManager *session_manager);

  /**
   * Set model manager
   * @param model_manager model manager pointer
   */
  void set_model_manager(core::ModelManager *model_manager);

  /**
   * Set config manager
   * @param config_manager config manager pointer
   */
  void set_config_manager(core::ConfigManager *config_manager);

  /**
   * Load and display messages for specified session
   * @param session_id session ID
   */
  void load_session_messages(const std::string &session_id);

  /**
   * Update model selector
   */
  void update_model_selector();

private:
  /**
   * Generate AI response
   * @param message user input message
   * @return AI generated response
   */
  std::string generate_ai_response(const std::string &message);

  /**
   * Stream AI response chunk by chunk
   */
  void stream_ai_response(const std::string &message);

  /**
   * Create an assistant message bubble and return its label for streaming
   */
  GtkWidget *add_assistant_placeholder(const std::string &text);

  /**
   * Append streamed text into the current assistant bubble
   */
  void append_stream_text(const std::string &delta, bool finished);
  GtkWidget *main_widget_;         // Main container
  GtkWidget *chat_scrolled_;       // Scrolled window
  GtkWidget *chat_box_;            // Chat message container
  GtkWidget *input_box_;           // Input area container
  GtkWidget *input_entry_;         // Input entry
  GtkWidget *send_button_;         // Send button
  GtkWidget *upload_image_button_; // Upload image button
  GtkWidget *upload_file_button_;  // Upload file button
  GtkWidget *video_record_button_; // Video record button (GtkToggleButton)

  // Store selected file paths
  std::string selected_image_path_;
  std::string selected_file_path_;
  GtkWidget *model_selector_;  // Model selector
  GtkWidget *input_container_; // Input container

  bool welcome_cleared_; // Flag indicating whether welcome screen has been cleared

  // Media capture related
  std::unique_ptr<media::VideoCapture> video_capture_;
  std::unique_ptr<media::AudioCapture> audio_capture_;
  std::unique_ptr<EnhancedVideoCaptureWindow> enhanced_video_window_;
  std::unique_ptr<VideoSourceDialog> video_source_dialog_;
  bool is_recording_; // Recording status flag

  // Recording button icons
  GtkWidget *video_off_image_; // Recording off icon
  GtkWidget *video_on_image_;  // Recording on icon

  // Flag to prevent recursive calls
  bool updating_button_state_; // Flag indicating whether button state is being updated

  // Session manager pointer
  ChatSessionManager *session_manager_; // Session manager pointer
  core::ModelManager *model_manager_;   // Model manager pointer
  core::ConfigManager *config_manager_; // 新增：配置管理器指针

  // Streaming related
  GtkWidget *streaming_label_ = nullptr; // Label of current streaming assistant bubble
  bool is_streaming_ = false;            // Streaming flag
  std::string streaming_buffer_;         // Accumulated streamed text

  // Video frame cache related
  std::shared_ptr<duorou::media::VideoFrame>
      cached_video_frame_;                                  // Cached video frame
  std::chrono::steady_clock::time_point last_video_update_; // Last video update time
  static constexpr int VIDEO_UPDATE_INTERVAL_MS =
      66; // Video update interval (about 15fps, reduce flicker)

  // Audio frame cache related
  std::vector<duorou::media::AudioFrame> cached_audio_frames_; // Cached audio frames
  std::chrono::steady_clock::time_point last_audio_update_; // Last audio update time
  static constexpr int AUDIO_UPDATE_INTERVAL_MS = 100;      // Audio update interval

  /**
   * Create chat display area
   */
  void create_chat_area();

  /**
   * Create welcome screen
   */
  void create_welcome_screen();

  /**
   * Create input area
   */
  void create_input_area();

  /**
   * Connect signals
   */
  void connect_signals();

  /**
   * Scroll to bottom
   */
  void scroll_to_bottom();

  // Static callback functions
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

  // Video capture methods
  void show_video_source_dialog();
  void start_desktop_capture();
  void start_camera_capture();
  void stop_recording();

  // Video source selection callback
  void on_video_source_selected(VideoSourceDialog::VideoSource source);

  // State management methods
  void verify_button_state();

  /**
   * Reset all states to prevent segmentation faults
   * Clean up recording state, button state, etc.
   */
  void reset_state();

private:
};

} // namespace gui
} // namespace duorou

#endif // DUOROU_GUI_CHAT_VIEW_H