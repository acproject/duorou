#ifndef DUOROU_GUI_CHAT_VIEW_H
#define DUOROU_GUI_CHAT_VIEW_H

#ifndef __cplusplus
typedef struct duorou_gui_chat_view ChatView;
#else

#include "../media/audio_capture.h"
#include "../media/video_frame.h"
#include "enhanced_video_capture_window.h"
#include "video_source_dialog.h"
#include "../core/model_manager.h"
#include <chrono>
// Prefer header presence over build macros; include GTK headers only
// when they are actually available to the compiler/indexer.
#if __has_include(<gtk/gtk.h>)
#  include <gtk/gtk.h>
#  if !defined(DUOROU_HAVE_GTK)
#    define DUOROU_HAVE_GTK 1
#  endif
#endif
#ifndef DUOROU_HAVE_GTK
// Lightweight GTK/GLib stubs to help indexers that lack GTK headers.
// They do not change runtime behavior; real builds still use GTK.
typedef void GtkWidget; typedef void GtkDialog; typedef void GtkButton; typedef void GtkToggleButton; typedef void GtkStyleContext; typedef void GtkCssProvider;
typedef void GtkEntry; typedef void GtkScrolledWindow; typedef void GtkDropDown; typedef void GtkLabel; typedef void GtkImage;
typedef int gint; typedef void* gpointer;
typedef unsigned int guint;
typedef int gboolean;
typedef void GdkFrameClock;
#ifndef GTK_IS_WIDGET
#define GTK_IS_WIDGET(x) (true)
#endif
#ifndef GTK_IS_TOGGLE_BUTTON
#define GTK_IS_TOGGLE_BUTTON(x) (true)
#endif
#ifndef GTK_IS_BUTTON
#define GTK_IS_BUTTON(x) (true)
#endif
#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif
#ifndef GTK_ORIENTATION_VERTICAL
#define GTK_ORIENTATION_VERTICAL 0
#endif
#ifndef GTK_ORIENTATION_HORIZONTAL
#define GTK_ORIENTATION_HORIZONTAL 1
#endif
#ifndef GTK_ALIGN_START
#define GTK_ALIGN_START 0
#endif
#ifndef GTK_ALIGN_END
#define GTK_ALIGN_END 1
#endif
#ifndef GTK_ALIGN_CENTER
#define GTK_ALIGN_CENTER 2
#endif
#ifndef GTK_ALIGN_FILL
#define GTK_ALIGN_FILL 3
#endif
#ifndef GTK_WRAP_WORD_CHAR
#define GTK_WRAP_WORD_CHAR 0
#endif
#ifndef G_SOURCE_REMOVE
#define G_SOURCE_REMOVE 0
#endif
#ifndef G_SOURCE_CONTINUE
#define G_SOURCE_CONTINUE 1
#endif
// Additional common constants
#ifndef GTK_STYLE_PROVIDER_PRIORITY_APPLICATION
#define GTK_STYLE_PROVIDER_PRIORITY_APPLICATION 600
#endif
#ifndef GTK_POLICY_AUTOMATIC
#define GTK_POLICY_AUTOMATIC 1
#endif
#ifndef GTK_INPUT_PURPOSE_FREE_FORM
#define GTK_INPUT_PURPOSE_FREE_FORM 0
#endif
#ifndef GTK_INPUT_HINT_NONE
#define GTK_INPUT_HINT_NONE 0
#endif
#endif // DUOROU_HAVE_GTK
#include <memory>
#include <string>
#include <vector>
#include <deque>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <thread>

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
class MarkdownView;

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
  MarkdownView *add_assistant_placeholder(const std::string &text);

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
  GtkWidget *voice_button_;        // Voice button
  GtkWidget *play_button_;         // Play button
  GtkWidget *upload_image_button_; // Upload image button
  GtkWidget *upload_file_button_;  // Upload file button
  GtkWidget *upload_video_button_; // Upload video file button
  GtkWidget *video_record_button_; // Video record button (GtkToggleButton)
  GtkWidget *file_preview_label_;  // Label to show selected file

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
  MarkdownView *streaming_md_ = nullptr; // 当前流式助手气泡的 MarkdownView
  bool is_streaming_ = false;            // Streaming flag
  std::string streaming_buffer_;         // Accumulated streamed text
  std::string last_assistant_message_;
  int tts_pid_ = -1;
  std::mutex tts_mutex_;
  std::mutex tts_queue_mutex_;
  std::condition_variable tts_queue_cv_;
  std::deque<std::string> tts_queue_;
  std::thread tts_worker_;
  std::atomic<bool> tts_worker_stop_{false};
  bool tts_worker_started_ = false;
  bool voice_response_tts_ = false;
  std::string tts_stream_pending_;

  bool is_voice_recording_ = false;
  bool voice_send_pending_ = false;
  std::unique_ptr<media::AudioCapture> voice_audio_capture_;
  std::mutex voice_audio_mutex_;
  std::vector<float> voice_audio_buffer_;
  int voice_audio_sample_rate_ = 0;
  int voice_audio_channels_ = 0;

  // Video frame cache related
  std::shared_ptr<duorou::media::VideoFrame>
      cached_video_frame_;                                  // Cached video frame
  std::chrono::steady_clock::time_point last_video_update_; // Last video update time
  static constexpr int VIDEO_UPDATE_INTERVAL_MS =
      66; // Video update interval (about 15fps, reduce flicker)
  std::chrono::steady_clock::time_point
      last_video_analysis_time_; // Last video analysis time
  static constexpr int VIDEO_ANALYSIS_INTERVAL_MS =
      3000; // Video analysis interval (ms)

  // Audio frame cache related
  std::vector<duorou::media::AudioFrame> cached_audio_frames_; // Cached audio frames
  std::mutex cached_audio_mutex_;
  std::chrono::steady_clock::time_point last_audio_update_; // Last audio update time
  static constexpr int AUDIO_UPDATE_INTERVAL_MS = 100;      // Audio update interval

  guint live_speech_timer_id_ = 0;
  std::atomic<bool> live_speech_inflight_{false};
  std::string live_last_speech_text_;
  bool live_speech_error_reported_ = false;
  std::atomic<bool> voice_qa_enabled_{false};

  std::thread voice_qa_worker_;
  std::atomic<bool> voice_qa_worker_stop_{false};
  bool voice_qa_worker_started_ = false;
  std::mutex voice_qa_mutex_;
  std::condition_variable voice_qa_cv_;
  std::deque<std::string> voice_qa_queue_;
  std::mutex voice_qa_response_mutex_;
  std::condition_variable voice_qa_response_cv_;
  bool voice_qa_waiting_response_ = false;

  std::mutex live_speech_mutex_;
  std::vector<float> live_speech_audio_buffer_;
  size_t live_speech_audio_read_pos_ = 0;
  int live_speech_sample_rate_ = 0;
  int live_speech_channels_ = 0;

  struct PendingLiveChunk {
    std::vector<float> audio;
    int sample_rate = 0;
    int channels = 0;
    std::shared_ptr<duorou::media::VideoFrame> video;
    double timestamp = 0.0;
  };

  std::mutex live_mutex_;
  std::vector<float> live_audio_buffer_;
  size_t live_audio_read_pos_ = 0;
  int live_audio_sample_rate_ = 0;
  int live_audio_channels_ = 0;
  bool live_utterance_active_ = false;
  int live_utterance_voice_ms_ = 0;
  int live_utterance_silence_ms_ = 0;
  std::vector<float> live_utterance_audio_;
  double live_noise_floor_ = 0.0;
  bool live_noise_floor_initialized_ = false;
  std::deque<PendingLiveChunk> pending_live_chunks_;
  std::vector<std::string> live_generated_files_;
  std::atomic<bool> live_idle_scheduled_{false};
  std::mutex cached_video_mutex_;
  std::atomic<bool> live_mnn_omni_enabled_{false};
  std::atomic<bool> live_inference_enabled_{false};
  std::atomic<bool> live_omni_listening_{false};
  std::atomic<bool> live_omni_processing_{false};
  guint bubble_width_tick_id_ = 0;
  int bubble_width_last_px_ = 0;

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

  void push_live_audio_frame(const duorou::media::AudioFrame &frame);
  void push_live_speech_audio_frame(const duorou::media::AudioFrame &frame);
  void schedule_try_send_live_chunks();
  void try_send_live_chunks_on_main_thread();
  void start_live_speech_transcription_timer();
  void stop_live_speech_transcription_timer();
  void tick_live_speech_transcription_on_main_thread();
  void ensure_voice_qa_worker_started();
  void enqueue_voice_qa_question(const std::string &text);
  void voice_qa_worker_loop();
  void send_user_message(const std::string &message);
  bool current_model_is_mnn_omni() const;
  void cleanup_live_generated_files_locked();

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
  static void on_voice_button_toggled(GtkToggleButton *toggle_button,
                                      gpointer user_data);
  static void on_play_button_clicked(GtkWidget *widget, gpointer user_data);
  static void on_upload_image_button_clicked(GtkWidget *widget,
                                             gpointer user_data);
  static void on_upload_file_button_clicked(GtkWidget *widget,
                                            gpointer user_data);
  static void on_upload_video_button_clicked(GtkWidget *widget,
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
  static void on_video_file_dialog_response(GtkDialog *dialog, gint response_id,
                                            gpointer user_data);

  // Video capture methods
  void show_video_source_dialog();
  void start_desktop_capture();
  void start_camera_capture();
  void stop_recording();

  // Periodic video analysis
  void analyze_video_frame(const duorou::media::VideoFrame &frame);

  // Video source selection callback
  void on_video_source_selected(VideoSourceDialog::VideoSource source);

  // State management methods
  void verify_button_state();
  void play_last_assistant_message();
  void start_voice_recording();
  void stop_voice_recording();
  void enqueue_tts_segment(const std::string &text);
  void stop_all_tts();
  void ensure_tts_worker_started();
  void tts_worker_loop();
  void feed_streaming_tts(const std::string &delta, bool finished);

  /**
   * Reset all states to prevent segmentation faults
   * Clean up recording state, button state, etc.
   */
  void reset_state();

private:
  // Recompute and apply bubble max-width based on right area width (70%)
  void update_bubble_max_width();
  // Callback when the scrolled window gets a new size allocation
  static void on_scrolled_size_allocate(GtkWidget *widget, gpointer allocation, gpointer user_data);
};

} // namespace gui
} // namespace duorou

#endif
#endif // DUOROU_GUI_CHAT_VIEW_H
