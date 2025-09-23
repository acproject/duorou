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
      session_manager_(nullptr), model_manager_(nullptr), cached_video_frame_(nullptr),
      last_video_update_(std::chrono::steady_clock::now()),
      last_audio_update_(std::chrono::steady_clock::now()) {
  // Initialize enhanced video window
  if (enhanced_video_window_) {
    enhanced_video_window_->initialize();
  }

  // Set video window close callback
  enhanced_video_window_->set_close_callback([this]() {
    stop_recording();
    // Ensure button is re-enabled
    if (video_record_button_) {
      gtk_widget_set_sensitive(video_record_button_, TRUE);
    }
  });

  // Set window selection callback (desktop capture mode)
  enhanced_video_window_->set_window_selection_callback(
      [this](const EnhancedVideoCaptureWindow::WindowInfo &window_info) {
        std::cout << "Window selected: " << window_info.title
                  << " (ID: " << window_info.window_id << ")" << std::endl;
        if (video_capture_) {
          video_capture_->set_capture_window_id(window_info.window_id);
          std::cout << "Capture window ID set: " << window_info.window_id
                    << std::endl;

          // If recording, use dynamic window update feature
          if (video_capture_->is_capturing()) {
            std::cout << "Dynamically updating screen capture window..." << std::endl;
#ifdef __APPLE__
            duorou::media::update_macos_screen_capture_window(
                window_info.window_id);
#endif
          }
        }
      });

  // Set device selection callback (camera mode)
  enhanced_video_window_->set_device_selection_callback(
      [this](const EnhancedVideoCaptureWindow::DeviceInfo &device_info) {
        std::cout << "Device selected: " << device_info.name
                  << " (Index: " << device_info.device_index << ")" << std::endl;
        if (video_capture_) {
          // Record current recording status
          bool was_capturing = video_capture_->is_capturing();

          // If recording, stop current capture first
          if (was_capturing) {
            std::cout << "Stopping current camera capture to apply new device selection..." << std::endl;
            video_capture_->stop_capture();
          }

          video_capture_->set_camera_device_index(device_info.device_index);
          std::cout << "Capture device index set: " << device_info.device_index
                    << std::endl;

          // If valid device selected (index>=0), always try to start camera
          if (device_info.device_index >= 0) {
            std::cout << "Reinitializing and starting camera capture..." << std::endl;

            // Recreate video capture object to ensure complete reset
            video_capture_.reset();
            video_capture_ = std::make_unique<media::VideoCapture>();

            // Set video frame callback
            video_capture_->set_frame_callback(
                [this](const media::VideoFrame &frame) {
                  static int camera_frame_count = 0;
                  camera_frame_count++;

                  if (camera_frame_count <= 5 || camera_frame_count % 30 == 0) {
                    std::cout << "Received camera video frame #" << camera_frame_count
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
                                  std::cout << "Showing camera video window..."
                                            << std::endl;
                                  chat_view->enhanced_video_window_->show(
                                      EnhancedVideoCaptureWindow::CaptureMode::
                                          CAMERA);
                                }
                              } catch (const std::exception &e) {
                                std::cout
                                    << "Error updating camera video frame: " << e.what()
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

            // Reinitialize to apply new device index
            if (video_capture_->initialize(duorou::media::VideoSource::CAMERA,
                                           device_info.device_index) &&
                video_capture_->start_capture()) {
              is_recording_ = true;
              std::cout << "Camera capture started, new device selection applied" << std::endl;
            } else {
              std::cout << "Failed to start camera capture" << std::endl;
            }
          } else if (device_info.device_index == -1) {
            is_recording_ = false;
            std::cout << "Camera disabled, stopping capture" << std::endl;
          }
        }
      });
}

ChatView::~ChatView() {
  std::cout << "ChatView destruction started..." << std::endl;

  // 1. First stop all recording activities to avoid triggering callbacks during destruction
  if (is_recording_) {
    std::cout << "Recording detected during destruction, forcing stop..." << std::endl;
    is_recording_ = false;

    // Immediately stop video and audio capture without waiting for callbacks
    if (video_capture_) {
      try {
        video_capture_->stop_capture();
      } catch (const std::exception &e) {
        std::cout << "Exception stopping video capture during destruction: " << e.what() << std::endl;
      }
    }

    if (audio_capture_) {
      try {
        audio_capture_->stop_capture();
      } catch (const std::exception &e) {
        std::cout << "Exception stopping audio capture during destruction: " << e.what() << std::endl;
      }
    }
  }

  // 2. Clear video window close callback to avoid triggering during destruction
  if (enhanced_video_window_) {
    try {
      enhanced_video_window_->set_close_callback(nullptr);
      enhanced_video_window_->hide();
    } catch (const std::exception &e) {
      std::cout << "Exception handling video window during destruction: " << e.what() << std::endl;
    }
  }

  // 3. Reset all states to ensure proper resource cleanup
  try {
    reset_state();
  } catch (const std::exception &e) {
    std::cout << "Exception resetting state during destruction: " << e.what() << std::endl;
  }

  // 4. Clean up video display window
  if (enhanced_video_window_) {
    try {
      enhanced_video_window_.reset();
    } catch (const std::exception &e) {
      std::cout << "Exception cleaning up video window during destruction: " << e.what() << std::endl;
    }
  }

  std::cout << "ChatView destruction completed" << std::endl;
  // GTK4 will automatically clean up child components
}

bool ChatView::initialize() {
  // Create main container
  main_widget_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
  if (!main_widget_) {
    std::cerr << "Failed to create chat view main container" << std::endl;
    return false;
  }

  gtk_widget_set_margin_start(main_widget_, 10);
  gtk_widget_set_margin_end(main_widget_, 10);
  gtk_widget_set_margin_top(main_widget_, 10);
  gtk_widget_set_margin_bottom(main_widget_, 10);

  // Create chat display area
  create_chat_area();

  // Create input area
  create_input_area();

  // Connect signals
  connect_signals();

  // Initialize video source selection dialog
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

  // Add user message to chat display
  add_message(message, true);
  
  // Save user message to current session
  if (session_manager_) {
    session_manager_->add_message_to_current_session(message, true);
  }

  // Show AI thinking indicator
  add_message("AI is thinking...", false);
  
  // Disable send button to prevent duplicate sending
  if (send_button_) {
    gtk_widget_set_sensitive(send_button_, FALSE);
  }
  if (input_entry_) {
    gtk_widget_set_sensitive(input_entry_, FALSE);
  }

  // Call AI model to process message in background thread
  std::thread([this, message]() {
    std::string ai_response = generate_ai_response(message);
    
    // Create data structure to pass to main thread
    struct CallbackData {
      ChatView* chat_view;
      std::string* response;
    };
    
    CallbackData* data = new CallbackData{this, new std::string(ai_response)};
    
    // Use g_idle_add to update UI in main thread
    g_idle_add([](gpointer user_data) -> gboolean {
      CallbackData* data = static_cast<CallbackData*>(user_data);
      ChatView* chat_view = data->chat_view;
      std::string* response = data->response;
      
      if (chat_view && response) {
        // Remove "AI is thinking..." message
        chat_view->remove_last_message();
        
        // Add AI response
        chat_view->add_message(*response, false);
        
        // Save AI response to current session
        if (chat_view->session_manager_) {
          chat_view->session_manager_->add_message_to_current_session(*response, false);
        }
        
        // Re-enable send button
        if (chat_view->send_button_) {
          gtk_widget_set_sensitive(chat_view->send_button_, TRUE);
        }
        if (chat_view->input_entry_) {
          gtk_widget_set_sensitive(chat_view->input_entry_, TRUE);
        }
      }
      
      // Clean up memory
      delete response;
      delete data;
      return G_SOURCE_REMOVE;
    }, data);
  }).detach();
}

void ChatView::add_message(const std::string &message, bool is_user) {
  if (!chat_box_) {
    return;
  }

  // Create message container - use horizontal layout for alignment
  GtkWidget *message_container = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
  gtk_widget_set_margin_start(message_container, 10);
  gtk_widget_set_margin_end(message_container, 10);
  gtk_widget_set_margin_top(message_container, 4);
  gtk_widget_set_margin_bottom(message_container, 4);

  // Create message label - display message content directly without prefix
  GtkWidget *message_label = gtk_label_new(message.c_str());
  gtk_label_set_wrap(GTK_LABEL(message_label), TRUE);
  gtk_label_set_wrap_mode(GTK_LABEL(message_label), PANGO_WRAP_WORD_CHAR);
  gtk_label_set_max_width_chars(GTK_LABEL(message_label), 50); // Limit maximum character count
  gtk_label_set_xalign(GTK_LABEL(message_label), 0.0);         // Left align text

  // Create bubble frame container
  GtkWidget *bubble_frame = gtk_frame_new(NULL);
  gtk_frame_set_child(GTK_FRAME(bubble_frame), message_label);

  // Create bubble container
  GtkWidget *bubble_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);

  if (is_user) {
    // User message: right-aligned, set background color directly
    gtk_widget_add_css_class(bubble_frame, "user-bubble");
    // Set background color and style directly
    GtkCssProvider *provider = gtk_css_provider_new();
    gtk_css_provider_load_from_string(
        provider, "frame { background: #48bb78; color: white; border-radius: "
                  "18px; padding: 12px 16px; margin: 4px; border: none; }");
    gtk_style_context_add_provider(gtk_widget_get_style_context(bubble_frame),
                                   GTK_STYLE_PROVIDER(provider),
                                   GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
    g_object_unref(provider);
    gtk_widget_set_halign(bubble_box, GTK_ALIGN_END);

    // Add left spacer to achieve right alignment effect
    GtkWidget *spacer = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    gtk_widget_set_hexpand(spacer, TRUE);
    gtk_box_append(GTK_BOX(message_container), spacer);

    gtk_box_append(GTK_BOX(bubble_box), bubble_frame);
    gtk_box_append(GTK_BOX(message_container), bubble_box);
  } else {
    // AI assistant message: left-aligned, set background color directly
    gtk_widget_add_css_class(bubble_frame, "assistant-bubble");
    // Directly set background color and style
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

    // Add right spacer to achieve left alignment effect
    GtkWidget *spacer = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    gtk_widget_set_hexpand(spacer, TRUE);
    gtk_box_append(GTK_BOX(message_container), spacer);
  }

  gtk_box_append(GTK_BOX(chat_box_), message_container);

  // Scroll to bottom
  scroll_to_bottom();
}

void ChatView::clear_chat() {
  if (chat_box_) {
    // Remove all child components
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
    // Get the last child component
    GtkWidget *last_child = gtk_widget_get_last_child(chat_box_);
    if (last_child) {
      gtk_box_remove(GTK_BOX(chat_box_), last_child);
    }
  }
}

void ChatView::create_chat_area() {
  // Create chat message container
  chat_box_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
  gtk_widget_set_valign(chat_box_, GTK_ALIGN_START);

  // Create welcome screen
  create_welcome_screen();

  // Create scrolled window
  chat_scrolled_ = gtk_scrolled_window_new();
  gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(chat_scrolled_),
                                 GTK_POLICY_AUTOMATIC, GTK_POLICY_AUTOMATIC);
  gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(chat_scrolled_), chat_box_);

  // Set scrolled window size
  gtk_widget_set_vexpand(chat_scrolled_, TRUE);
  gtk_widget_set_hexpand(chat_scrolled_, TRUE);

  // Add to main container
  gtk_box_append(GTK_BOX(main_widget_), chat_scrolled_);
}

void ChatView::create_input_area() {
  // Create main input container
  input_box_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
  gtk_widget_set_margin_start(input_box_, 20);
  gtk_widget_set_margin_end(input_box_, 20);
  gtk_widget_set_margin_bottom(input_box_, 20);
  gtk_widget_set_margin_top(input_box_, 10);

  // Create model selector container
  GtkWidget *model_container = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
  gtk_widget_set_halign(model_container, GTK_ALIGN_CENTER);

  // Create model selector (initially empty, filled later via update_model_selector)
  model_selector_ = gtk_drop_down_new_from_strings(
      (const char *[]){"No models available", NULL});
  gtk_widget_add_css_class(model_selector_, "model-selector");

  // Create model label
  GtkWidget *model_label = gtk_label_new("Model:");
  gtk_widget_add_css_class(model_label, "model-label");

  gtk_box_append(GTK_BOX(model_container), model_label);
  gtk_box_append(GTK_BOX(model_container), model_selector_);

  // Create input box container
  input_container_ = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
  gtk_widget_add_css_class(input_container_, "input-container");
  gtk_widget_set_hexpand(input_container_, TRUE);

  // Create message input box
  input_entry_ = gtk_entry_new();
  gtk_entry_set_placeholder_text(GTK_ENTRY(input_entry_), "Send a message...");
  gtk_widget_set_hexpand(input_entry_, TRUE);
  gtk_widget_add_css_class(input_entry_, "message-input");

  // Set input method related properties to avoid Pango errors
  gtk_entry_set_input_purpose(GTK_ENTRY(input_entry_),
                              GTK_INPUT_PURPOSE_FREE_FORM);
  gtk_entry_set_input_hints(GTK_ENTRY(input_entry_), GTK_INPUT_HINT_NONE);

  // Disable some features that may cause Pango errors
  gtk_entry_set_has_frame(GTK_ENTRY(input_entry_), TRUE);
  gtk_entry_set_activates_default(GTK_ENTRY(input_entry_), FALSE);

  // Set maximum length to avoid buffer overflow
  gtk_entry_set_max_length(GTK_ENTRY(input_entry_), 1000);

  // Set overwrite mode to FALSE to avoid cursor position issues
  gtk_entry_set_overwrite_mode(GTK_ENTRY(input_entry_), FALSE);

  // Enable input method support
  gtk_widget_set_can_focus(input_entry_, TRUE);
  gtk_widget_set_focusable(input_entry_, TRUE);

  // Create upload image button
  upload_image_button_ = gtk_button_new_with_label("Image");
  gtk_widget_add_css_class(upload_image_button_, "upload-button");
  gtk_widget_set_size_request(upload_image_button_, 40, 40);
  gtk_widget_set_tooltip_text(upload_image_button_, "Upload Image");

  // Create upload file button
  upload_file_button_ = gtk_button_new_with_label("File");
  gtk_widget_add_css_class(upload_file_button_, "upload-button");
  gtk_widget_set_size_request(upload_file_button_, 40, 40);
  gtk_widget_set_tooltip_text(upload_file_button_,
                              "Upload File (MD, DOC, Excel, PPT, PDF)");

  // Create video recording button icon - using relative path
  std::string icon_path_base = "src/gui/";
  // video_off_image_ =
  // gtk_picture_new_for_filename((icon_path_base +
  // "video-off.png").c_str());
  video_off_image_ =
      gtk_picture_new_for_filename((icon_path_base + "video-on.png").c_str());

  // Check if icon loaded successfully
  if (!video_off_image_ || !video_off_image_) {
    std::cout << "Warning: Unable to load recording button icon, using text alternative" << std::endl;
    // If icon loading fails, create text label as alternative
    if (!video_off_image_) {
      video_off_image_ = gtk_label_new("⏹");
    }
    if (!video_off_image_) {
      video_off_image_ = gtk_label_new("⏺");
    }
  }

  // Set icon size
  gtk_widget_set_size_request(video_off_image_, 24, 24);
  gtk_widget_set_size_request(video_off_image_, 24, 24);

  // Create video recording button (using GtkToggleButton)
  video_record_button_ = gtk_toggle_button_new();
  gtk_button_set_child(GTK_BUTTON(video_record_button_),
                       video_off_image_); // Default to show off state
  gtk_widget_add_css_class(video_record_button_, "upload-button");
  gtk_widget_set_size_request(video_record_button_, 40, 40);
  gtk_widget_set_tooltip_text(video_record_button_, "Start video recording/desktop capture");

  // Set toggle state change callback
  g_signal_connect(video_record_button_, "toggled",
                   G_CALLBACK(on_video_record_button_toggled), this);

  // Create send button
  send_button_ = gtk_button_new_with_label("↑");
  gtk_widget_add_css_class(send_button_, "send-button");
  gtk_widget_set_size_request(send_button_, 40, 40);

  // Add to input container
  gtk_box_append(GTK_BOX(input_container_), upload_image_button_);
  gtk_box_append(GTK_BOX(input_container_), upload_file_button_);
  gtk_box_append(GTK_BOX(input_container_), input_entry_);
  gtk_box_append(GTK_BOX(input_container_), video_record_button_);
  gtk_box_append(GTK_BOX(input_container_), send_button_);

  // Add to main input container
  gtk_box_append(GTK_BOX(input_box_), model_container);
  gtk_box_append(GTK_BOX(input_box_), input_container_);

  // Add to main container
  gtk_box_append(GTK_BOX(main_widget_), input_box_);
}

void ChatView::create_welcome_screen() {
  // Create welcome screen container
  GtkWidget *welcome_container = gtk_box_new(GTK_ORIENTATION_VERTICAL, 20);
  gtk_widget_set_halign(welcome_container, GTK_ALIGN_CENTER);
  gtk_widget_set_valign(welcome_container, GTK_ALIGN_CENTER);
  gtk_widget_set_vexpand(welcome_container, TRUE);
  gtk_widget_set_hexpand(welcome_container, TRUE);

  // Create application icon (using duorou01.png image)
  // Use absolute path to ensure image file can be found
  const char *icon_path =
      "/Users/acproject/workspace/cpp_projects/duorou/src/gui/duorou01.png";
  GtkWidget *icon_picture = gtk_picture_new_for_filename(icon_path);

  // If absolute path fails, try relative path
  if (!gtk_picture_get_file(GTK_PICTURE(icon_picture))) {
    g_object_unref(icon_picture);
    icon_picture = gtk_picture_new_for_filename("src/gui/duorou01.png");
  }

  gtk_picture_set_content_fit(GTK_PICTURE(icon_picture),
                              GTK_CONTENT_FIT_CONTAIN);
  gtk_widget_set_size_request(icon_picture, 16, 16);
  gtk_widget_add_css_class(icon_picture, "welcome-icon");

  // Create welcome text
  GtkWidget *welcome_title = gtk_label_new("Welcome to Duorou");
  gtk_widget_add_css_class(welcome_title, "welcome-title");

  GtkWidget *welcome_subtitle = gtk_label_new("Your AI Desktop Assistant");
  gtk_widget_add_css_class(welcome_subtitle, "welcome-subtitle");

  GtkWidget *welcome_hint =
      gtk_label_new("Start a conversation by typing a message below");
  gtk_widget_add_css_class(welcome_hint, "welcome-hint");

  // Add to container
  gtk_box_append(GTK_BOX(welcome_container), icon_picture);
  gtk_box_append(GTK_BOX(welcome_container), welcome_title);
  gtk_box_append(GTK_BOX(welcome_container), welcome_subtitle);
  gtk_box_append(GTK_BOX(welcome_container), welcome_hint);

  // Add to chat container
  gtk_box_append(GTK_BOX(chat_box_), welcome_container);
}

void ChatView::connect_signals() {
  // Connect send button signal
  g_signal_connect(send_button_, "clicked", G_CALLBACK(on_send_button_clicked),
                   this);

  // Connect upload image button signal
  g_signal_connect(upload_image_button_, "clicked",
                   G_CALLBACK(on_upload_image_button_clicked), this);

  // Connect upload file button signal
  g_signal_connect(upload_file_button_, "clicked",
                   G_CALLBACK(on_upload_file_button_clicked), this);

  // Connect video record button signal
  g_signal_connect(video_record_button_, "clicked",
                   G_CALLBACK(on_video_record_button_clicked), this);

  // Connect Enter key to send message
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

  // Use gtk_editable_get_text to get text directly, avoiding Pango errors from buffer operations
  const char *text_ptr =
      gtk_editable_get_text(GTK_EDITABLE(chat_view->input_entry_));
  std::string message_text = text_ptr ? std::string(text_ptr) : "";

  // Check if there is text message or selected files
  bool has_text = !message_text.empty();
  bool has_image = !chat_view->selected_image_path_.empty();
  bool has_file = !chat_view->selected_file_path_.empty();

  if (has_text || has_image || has_file) {
    // Use gtk_editable_set_text to clear input box
    gtk_editable_set_text(GTK_EDITABLE(chat_view->input_entry_), "");

    // Clear welcome screen only on first message send
    if (!chat_view->welcome_cleared_) {
      chat_view->clear_chat();
      chat_view->welcome_cleared_ = true;
    }

    // Build complete message
    std::string full_message = message_text;

    // Add image information
    if (has_image) {
      if (!full_message.empty())
        full_message += "\n";
      full_message +=
          "Image: " + std::string(g_path_get_basename(
                            chat_view->selected_image_path_.c_str()));
    }

    // Add document information
    if (has_file) {
      if (!full_message.empty())
        full_message += "\n";
      full_message +=
          "File: " + std::string(g_path_get_basename(
                            chat_view->selected_file_path_.c_str()));
    }

    // Send message
    chat_view->send_message(full_message);

    // Clear selected file paths and reset button tooltips
    if (has_image) {
      chat_view->selected_image_path_.clear();
      gtk_widget_set_tooltip_text(chat_view->upload_image_button_, "Upload Image");
    }
    if (has_file) {
      chat_view->selected_file_path_.clear();
      gtk_widget_set_tooltip_text(chat_view->upload_file_button_, "Upload Document");
    }
  }
}

void ChatView::on_input_entry_activate(GtkWidget *widget, gpointer user_data) {
  ChatView *chat_view = static_cast<ChatView *>(user_data);

  if (!widget) {
    return;
  }

  // Use gtk_editable_get_text to get text directly, avoiding Pango errors from buffer operations
  const char *text_ptr = gtk_editable_get_text(GTK_EDITABLE(widget));
  std::string message_text = text_ptr ? std::string(text_ptr) : "";

  // Check if there's text message or selected files
  bool has_text = !message_text.empty();
  bool has_image = !chat_view->selected_image_path_.empty();
  bool has_file = !chat_view->selected_file_path_.empty();

  if (has_text || has_image || has_file) {
    // Use gtk_editable_set_text to clear input box
    gtk_editable_set_text(GTK_EDITABLE(widget), "");

    // Clear welcome screen only on first message send
    if (!chat_view->welcome_cleared_) {
      chat_view->clear_chat();
      chat_view->welcome_cleared_ = true;
    }

    // Build complete message
    std::string full_message = message_text;

    // Add image information
    if (has_image) {
      if (!full_message.empty())
        full_message += "\n";
      full_message +=
          "Image: " + std::string(g_path_get_basename(
                            chat_view->selected_image_path_.c_str()));
    }

    // Add document information
    if (has_file) {
      if (!full_message.empty())
        full_message += "\n";
      full_message +=
          "File: " + std::string(g_path_get_basename(
                            chat_view->selected_file_path_.c_str()));
    }

    // Send message
    chat_view->send_message(full_message);

    // Clear selected file paths and reset button tooltips
    if (has_image) {
      chat_view->selected_image_path_.clear();
      gtk_widget_set_tooltip_text(chat_view->upload_image_button_, "Upload Image");
    }
    if (has_file) {
      chat_view->selected_file_path_.clear();
      gtk_widget_set_tooltip_text(chat_view->upload_file_button_, "Upload Document");
    }
  }
}

void ChatView::on_upload_image_button_clicked(GtkWidget *widget,
                                              gpointer user_data) {
  ChatView *chat_view = static_cast<ChatView *>(user_data);

  // Create file selection dialog
  GtkWidget *dialog = gtk_file_chooser_dialog_new(
      "Select Image", GTK_WINDOW(gtk_widget_get_root(widget)),
      GTK_FILE_CHOOSER_ACTION_OPEN, "_Cancel", GTK_RESPONSE_CANCEL, "_Open",
      GTK_RESPONSE_ACCEPT, NULL);

  // Set image file filter
  GtkFileFilter *filter = gtk_file_filter_new();
  gtk_file_filter_set_name(filter, "Image files");
  gtk_file_filter_add_mime_type(filter, "image/png");
  gtk_file_filter_add_mime_type(filter, "image/jpeg");
  gtk_file_filter_add_mime_type(filter, "image/gif");
  gtk_file_filter_add_mime_type(filter, "image/bmp");
  gtk_file_filter_add_mime_type(filter, "image/webp");
  gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dialog), filter);

  // Show dialog
  gtk_widget_show(dialog);

  // Store chat_view pointer in dialog data
  g_object_set_data(G_OBJECT(dialog), "chat_view", chat_view);

  // Connect response signal
  g_signal_connect(dialog, "response", G_CALLBACK(on_image_dialog_response),
                   NULL);
}

void ChatView::on_upload_file_button_clicked(GtkWidget *widget,
                                             gpointer user_data) {
  ChatView *chat_view = static_cast<ChatView *>(user_data);

  // Create file selection dialog
  GtkWidget *dialog = gtk_file_chooser_dialog_new(
      "Select Document", GTK_WINDOW(gtk_widget_get_root(widget)),
      GTK_FILE_CHOOSER_ACTION_OPEN, "_Cancel", GTK_RESPONSE_CANCEL, "_Open",
      GTK_RESPONSE_ACCEPT, NULL);

  // Set document file filter
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
        // Store selected image path, don't send directly
        chat_view->selected_image_path_ = std::string(filename);

        // Update upload button tooltip text or style to indicate file selected
        gtk_widget_set_tooltip_text(
            chat_view->upload_image_button_,
            ("Image selected: " + std::string(g_path_get_basename(filename)))
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
        // Store selected document path, don't send directly
        chat_view->selected_file_path_ = std::string(filename);

        // Update upload button tooltip text or style to indicate file selected
        gtk_widget_set_tooltip_text(
            chat_view->upload_file_button_,
            ("Document selected: " + std::string(g_path_get_basename(filename)))
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

  // If button is disabled during recording, only allow stopping recording
  if (!gtk_widget_get_sensitive(widget) && !chat_view->is_recording_) {
    return;
  }

  // Toggle function: stop if recording, otherwise show selection dialog
  if (chat_view->is_recording_) {
    chat_view->stop_recording();
  } else {
    chat_view->show_video_source_dialog();
  }
}

void ChatView::on_video_record_button_toggled(GtkToggleButton *toggle_button,
                                              gpointer user_data) {
  ChatView *chat_view = static_cast<ChatView *>(user_data);

  // Prevent signal processing when program is closing
  if (!chat_view || !chat_view->video_record_button_) {
    return;
  }

  // Prevent recursive calls
  if (chat_view->updating_button_state_) {
    return;
  }

  // Disable button for 1 second to prevent rapid repeated clicks
  gtk_widget_set_sensitive(GTK_WIDGET(toggle_button), FALSE);

  // Re-enable button after 1 second
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
  std::cout << "Video record button state change: "
            << (is_active ? "active(on)" : "inactive(off)") << std::endl;

  if (is_active) {
    // Button is activated, but don't start recording directly, show selection dialog instead
    if (!chat_view->is_recording_) {
      // Reset button state first to avoid button staying active when user cancels
      chat_view->updating_button_state_ = true;
      gtk_toggle_button_set_active(toggle_button, FALSE);
      chat_view->updating_button_state_ = false;

      // Show video source selection dialog
      chat_view->show_video_source_dialog();
    }
  } else {
    // Toggle button inactive = off state = show video-off icon
    if (chat_view->video_off_image_ &&
        GTK_IS_WIDGET(chat_view->video_off_image_)) {
      gtk_widget_set_visible(chat_view->video_off_image_, TRUE);
      gtk_button_set_child(GTK_BUTTON(toggle_button),
                           chat_view->video_off_image_);
      // Remove recording state CSS class
      gtk_widget_remove_css_class(GTK_WIDGET(toggle_button), "recording");
      // Ensure base style class exists
      if (!gtk_widget_has_css_class(GTK_WIDGET(toggle_button),
                                    "upload-button")) {
        gtk_widget_add_css_class(GTK_WIDGET(toggle_button), "upload-button");
      }
      gtk_widget_set_tooltip_text(GTK_WIDGET(toggle_button),
                                  "Start video recording/desktop capture");
    }
    std::cout << "Icon switched to video-off (off state)" << std::endl;

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

  // Prevent duplicate initialization, set flag
  static bool initializing = false;
  if (initializing) {
    std::cout << "Desktop capture is initializing, please wait..." << std::endl;
    return;
  }
  initializing = true;

  // Ensure previous resources are cleaned up
  if (video_capture_) {
    std::cout << "Stopping previous video capture..." << std::endl;
    video_capture_->stop_capture();
    // Wait for a while to ensure resources are fully released
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    video_capture_.reset();
    std::cout << "Previous video capture stopped" << std::endl;
  }
  if (audio_capture_) {
    std::cout << "Stopping previous audio capture..." << std::endl;
    audio_capture_->stop_capture();
    audio_capture_.reset();
    std::cout << "Previous audio capture stopped" << std::endl;
  }

  // On macOS, ensure ScreenCaptureKit resources are fully cleaned up
#ifdef __APPLE__
  std::cout << "Cleaning up macOS screen capture resources..." << std::endl;
  media::cleanup_macos_screen_capture();
  // Additional wait time to ensure macOS resources are fully released
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  std::cout << "macOS screen capture resource cleanup completed" << std::endl;
#endif

  // Initialize video capture
  video_capture_ = std::make_unique<media::VideoCapture>();

  // Initialize audio capture
  audio_capture_ = std::make_unique<media::AudioCapture>();

  // Set video frame callback - use caching mechanism to reduce flickering
  video_capture_->set_frame_callback([this](const media::VideoFrame &frame) {
    // Static counter, only output frame info at the beginning
    static int frame_count = 0;
    frame_count++;

    if (frame_count <= 5 || frame_count % 30 == 0) { // Only output first 5 frames and every 30th frame
      std::cout << "Received video frame #" << frame_count << ": " << frame.width << "x"
                << frame.height << std::endl;
    }

    // Check if video frame needs to be updated (based on time interval)
    auto now = std::chrono::steady_clock::now();
    auto time_since_last_update =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_video_update_)
            .count();

    if (time_since_last_update >= VIDEO_UPDATE_INTERVAL_MS) {
      last_video_update_ = now;

      // Update video display window directly, avoiding complex memory allocation
      if (enhanced_video_window_) {
        // Create frame copy for asynchronous update
        media::VideoFrame *frame_copy = new media::VideoFrame(frame);

        g_idle_add(
            [](gpointer user_data) -> gboolean {
              auto *data =
                  static_cast<std::pair<ChatView *, media::VideoFrame *> *>(
                      user_data);
              ChatView *chat_view = data->first;
              media::VideoFrame *frame_ptr = data->second;

              // Check if ChatView object is still valid
              if (chat_view && chat_view->enhanced_video_window_) {
                try {
                  chat_view->enhanced_video_window_->update_frame(*frame_ptr);

                  // Only output log on first display
                  if (!chat_view->enhanced_video_window_->is_visible()) {
                    std::cout << "Showing video window..." << std::endl;
                    chat_view->enhanced_video_window_->show(
                        EnhancedVideoCaptureWindow::CaptureMode::DESKTOP);
                  }
                } catch (const std::exception &e) {
                  std::cout << "Error updating video frame: " << e.what() << std::endl;
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

  // Set audio frame callback - use caching mechanism to reduce processing frequency
  audio_capture_->set_frame_callback([this](const media::AudioFrame &frame) {
    // Static counter, only output frame info at the beginning
    static int audio_frame_count = 0;
    audio_frame_count++;

    if (audio_frame_count <= 3 ||
        audio_frame_count % 100 == 0) { // Only output first 3 frames and every 100th frame
      std::cout << "Received audio frame #" << audio_frame_count << ": "
                << frame.frame_count << " samples, " << frame.sample_rate << "Hz"
                << std::endl;
    }

    // Check if audio frame needs to be processed (based on time interval)
    auto now = std::chrono::steady_clock::now();
    auto time_since_last_update =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_audio_update_)
            .count();

    if (time_since_last_update >= AUDIO_UPDATE_INTERVAL_MS) {
      // Cache audio frames (keep recent frames)
      cached_audio_frames_.push_back(frame);

      // Limit cache size, only keep the most recent 10 frames
      if (cached_audio_frames_.size() > 10) {
        cached_audio_frames_.erase(cached_audio_frames_.begin());
      }

      last_audio_update_ = now;
    }
  });

  // Initialize desktop capture
  if (video_capture_->initialize(media::VideoSource::DESKTOP_CAPTURE)) {
    if (video_capture_->start_capture()) {
      // Initialize microphone audio capture
      if (audio_capture_->initialize(media::AudioSource::MICROPHONE)) {
        if (audio_capture_->start_capture()) {
          is_recording_ = true;

          // Only update button state, icon is handled by toggle callback
          if (video_record_button_) {
            // Only set to active state when button is not active, avoid recursion
            if (!gtk_toggle_button_get_active(
                    GTK_TOGGLE_BUTTON(video_record_button_))) {
              gtk_toggle_button_set_active(
                  GTK_TOGGLE_BUTTON(video_record_button_), TRUE);
            }
            std::cout << "Button state switched to active" << std::endl;
          }

          std::cout << "Desktop recording started - capturing desktop video and microphone audio"
                    << std::endl;

          // Reset initialization flag
          initializing = false;
        } else {
          std::cout << "Audio capture startup failed" << std::endl;
          // Reset button state, use flag to prevent recursive calls
          if (video_record_button_) {
            updating_button_state_ = true;
            gtk_toggle_button_set_active(
                GTK_TOGGLE_BUTTON(video_record_button_), FALSE);
            // Directly update icon to off state
            if (video_off_image_) {
              gtk_button_set_child(GTK_BUTTON(video_record_button_),
                                   video_off_image_);
              gtk_widget_set_tooltip_text(GTK_WIDGET(video_record_button_),
                                          "Start video recording/desktop capture");
            }
            updating_button_state_ = false;
          }
          initializing = false;
        }
      } else {
        std::cout << "Audio capture initialization failed" << std::endl;
        // Reset button state, use flag to prevent recursive calls
        if (video_record_button_) {
          updating_button_state_ = true;
          gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(video_record_button_),
                                       FALSE);
          // Directly update icon to off state
          if (video_off_image_) {
            gtk_button_set_child(GTK_BUTTON(video_record_button_),
                                 video_off_image_);
            gtk_widget_set_tooltip_text(GTK_WIDGET(video_record_button_),
                                        "Start video recording/desktop capture");
          }
          updating_button_state_ = false;
        }
        initializing = false;
      }
    } else {
      std::cout << "Video capture startup failed" << std::endl;
      // Reset button state, use flag to prevent recursive calls
      if (video_record_button_) {
        updating_button_state_ = true;
        gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(video_record_button_),
                                     FALSE);
        // Directly update icon to off state
        if (video_off_image_) {
          gtk_button_set_child(GTK_BUTTON(video_record_button_),
                               video_off_image_);
          gtk_widget_set_tooltip_text(GTK_WIDGET(video_record_button_),
                                      "Start video recording/desktop capture");
        }
        updating_button_state_ = false;
      }
      initializing = false;
    }
  } else {
    std::cout << "Video capture initialization failed" << std::endl;

    // Reset button state, icon is handled by toggle callback
    if (video_record_button_) {
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(video_record_button_),
                                   FALSE);
    }

    // Show error message
    GtkWidget *dialog = gtk_message_dialog_new(
        GTK_WINDOW(gtk_widget_get_root(main_widget_)), GTK_DIALOG_MODAL,
        GTK_MESSAGE_ERROR, GTK_BUTTONS_OK,
        "Desktop capture initialization failed\n\nPlease check system permission settings.");

    // Ensure dialog is on top
    gtk_window_set_modal(GTK_WINDOW(dialog), TRUE);
    gtk_window_present(GTK_WINDOW(dialog));
    gtk_widget_show(dialog);
    g_signal_connect(dialog, "response", G_CALLBACK(gtk_window_destroy), NULL);

    // Reset initialization flag
    initializing = false;
  }
}

void ChatView::start_camera_capture() {
  std::cout << "Starting camera capture..." << std::endl;

  if (is_recording_) {
    // Stop current recording
    stop_recording();
    return;
  }

  // Check if camera is available
  if (!media::VideoCapture::is_camera_available()) {
    // Show camera unavailable message and provide fallback to desktop capture option
    GtkWidget *dialog = gtk_message_dialog_new(
        GTK_WINDOW(gtk_widget_get_root(main_widget_)), GTK_DIALOG_MODAL,
        GTK_MESSAGE_WARNING, GTK_BUTTONS_NONE,
        "No available camera device detected\n\nUse desktop capture as alternative?");

    gtk_dialog_add_button(GTK_DIALOG(dialog), "Use Desktop Capture", GTK_RESPONSE_YES);
    gtk_dialog_add_button(GTK_DIALOG(dialog), "Cancel", GTK_RESPONSE_NO);

    // Ensure dialog is on top
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

  // Ensure previous resources are cleaned up
  if (video_capture_) {
    std::cout << "Stopping previous video capture..." << std::endl;
    video_capture_->stop_capture();
    // Wait for a while to ensure resources are fully released
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    video_capture_.reset();
    std::cout << "Previous video capture stopped" << std::endl;
  }
  if (audio_capture_) {
    std::cout << "Stopping previous audio capture..." << std::endl;
    audio_capture_->stop_capture();
    audio_capture_.reset();
    std::cout << "Previous audio capture stopped" << std::endl;
  }

  // On macOS, ensure ScreenCaptureKit resources are fully cleaned up
#ifdef __APPLE__
  std::cout << "Cleaning up macOS screen capture resources..." << std::endl;
  media::cleanup_macos_screen_capture();
  // Additional wait time to ensure macOS resources are fully released
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  std::cout << "macOS screen capture resource cleanup completed" << std::endl;
#endif

  // Initialize video capture
  video_capture_ = std::make_unique<media::VideoCapture>();

  // Initialize audio capture
  audio_capture_ = std::make_unique<media::AudioCapture>();

  // Set video frame callback - use caching mechanism to reduce flickering
  video_capture_->set_frame_callback([this](const media::VideoFrame &frame) {
    // Static counter and flag, reduce log output
    static int camera_frame_count = 0;
    static bool window_shown_logged = false;
    camera_frame_count++;

    if (camera_frame_count <= 5 ||
        camera_frame_count % 30 == 0) { // Only output first 5 frames and every 30th frame
      std::cout << "Received camera video frame #" << camera_frame_count << ": "
                << frame.width << "x" << frame.height << std::endl;
    }

    // Check if video frame needs to be updated (based on time interval)
    auto now = std::chrono::steady_clock::now();
    auto time_since_last_update =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_video_update_)
            .count();

    if (time_since_last_update >= VIDEO_UPDATE_INTERVAL_MS) {
      last_video_update_ = now;

      // Update video display window directly, avoiding complex memory allocation
      if (enhanced_video_window_) {
        // Create frame copy for asynchronous update
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

                  // Only output log on first display
                  if (!chat_view->enhanced_video_window_->is_visible()) {
                    std::cout << "Showing camera video window..." << std::endl;
                    chat_view->enhanced_video_window_->show(
                        EnhancedVideoCaptureWindow::CaptureMode::CAMERA);
                  }
                } catch (const std::exception &e) {
                  std::cout << "Error updating camera video frame: " << e.what()
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

  // Set audio frame callback - use caching mechanism to reduce processing frequency
  audio_capture_->set_frame_callback([this](const media::AudioFrame &frame) {
    // Static counter, only output frame info at the beginning
    static int camera_audio_frame_count = 0;
    camera_audio_frame_count++;

    if (camera_audio_frame_count <= 3 ||
        camera_audio_frame_count % 100 == 0) { // Only output first 3 frames and every 100th frame
      std::cout << "Received camera audio frame #" << camera_audio_frame_count << ": "
                << frame.frame_count << " samples, " << frame.sample_rate << "Hz"
                << std::endl;
    }

    // Check if audio frame needs to be processed (based on time interval)
    auto now = std::chrono::steady_clock::now();
    auto time_since_last_update =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_audio_update_)
            .count();

    if (time_since_last_update >= AUDIO_UPDATE_INTERVAL_MS) {
      // Cache audio frames (keep recent frames)
      cached_audio_frames_.push_back(frame);

      // Limit cache size, only keep the most recent 10 frames
      if (cached_audio_frames_.size() > 10) {
        cached_audio_frames_.erase(cached_audio_frames_.begin());
      }

      last_audio_update_ = now;
    }
  });

  // Initialize camera capture
  if (video_capture_->initialize(media::VideoSource::CAMERA, 0)) {
    if (video_capture_->start_capture()) {
      // Initialize microphone audio capture
      if (audio_capture_->initialize(media::AudioSource::MICROPHONE)) {
        if (audio_capture_->start_capture()) {
          is_recording_ = true;

          // Only update button state, icon is handled by toggle callback
          if (video_record_button_) {
            // Only set to active state when button is not active, avoid recursion
            if (!gtk_toggle_button_get_active(
                    GTK_TOGGLE_BUTTON(video_record_button_))) {
              gtk_toggle_button_set_active(
                  GTK_TOGGLE_BUTTON(video_record_button_), TRUE);
            }
          }

          std::cout << "Camera recording started - capturing camera video and microphone audio"
                    << std::endl;
        } else {
          std::cout << "Audio capture startup failed" << std::endl;
          // Reset button state, use flag to prevent recursive calls
          if (video_record_button_) {
            updating_button_state_ = true;
            gtk_toggle_button_set_active(
                GTK_TOGGLE_BUTTON(video_record_button_), FALSE);
            // Directly update icon to off state
            if (video_off_image_) {
              gtk_button_set_child(GTK_BUTTON(video_record_button_),
                                   video_off_image_);
              gtk_widget_set_tooltip_text(GTK_WIDGET(video_record_button_),
                                          "Start video recording/desktop capture");
            }
            updating_button_state_ = false;
          }
        }
      } else {
        std::cout << "Audio capture initialization failed" << std::endl;
        // Reset button state, use flag to prevent recursive calls
        if (video_record_button_) {
          updating_button_state_ = true;
          gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(video_record_button_),
                                       FALSE);
          // Directly update icon to off state
          if (video_off_image_) {
            gtk_button_set_child(GTK_BUTTON(video_record_button_),
                                 video_off_image_);
            gtk_widget_set_tooltip_text(GTK_WIDGET(video_record_button_),
                                        "Start video recording/desktop capture");
          }
          updating_button_state_ = false;
        }
      }
    } else {
      std::cout << "Camera capture startup failed" << std::endl;
      // Reset button state, use flag to prevent recursive calls
      if (video_record_button_) {
        updating_button_state_ = true;
        gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(video_record_button_),
                                     FALSE);
        // Directly update icon to off state
        if (video_off_image_) {
          gtk_button_set_child(GTK_BUTTON(video_record_button_),
                               video_off_image_);
          gtk_widget_set_tooltip_text(GTK_WIDGET(video_record_button_),
                                      "Start video recording/desktop capture");
        }
        updating_button_state_ = false;
      }
    }
  } else {
    std::cout << "Camera capture initialization failed" << std::endl;

    // Reset button state, use flag to prevent recursive calls
    if (video_record_button_) {
      updating_button_state_ = true;
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(video_record_button_),
                                   FALSE);
      // Directly update icon to off state
      if (video_off_image_) {
        gtk_button_set_child(GTK_BUTTON(video_record_button_),
                             video_off_image_);
        gtk_widget_set_tooltip_text(GTK_WIDGET(video_record_button_),
                                    "Start video recording/desktop capture");
      }
      updating_button_state_ = false;
    }

    // Show error message
    GtkWidget *dialog = gtk_message_dialog_new(
        GTK_WINDOW(gtk_widget_get_root(main_widget_)), GTK_DIALOG_MODAL,
        GTK_MESSAGE_ERROR, GTK_BUTTONS_OK,
        "Camera capture initialization failed\n\nPlease check camera permission settings.");

    // Ensure dialog is on top
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

  // Prevent duplicate calls
  static bool stopping = false;
  if (stopping) {
    std::cout << "Stop recording already in progress, skipping" << std::endl;
    return;
  }
  stopping = true;

  // First set recording status to false
  is_recording_ = false;

  // Stop video capture
  if (video_capture_) {
    try {
      std::cout << "Stopping video capture..." << std::endl;
      video_capture_->stop_capture();
      // Wait a short time to ensure all pending callbacks complete
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      video_capture_.reset(); // Reset video capture object
      std::cout << "Video capture stopped" << std::endl;
    } catch (const std::exception &e) {
      std::cout << "Error stopping video capture: " << e.what() << std::endl;
    }
  }

  // Stop audio capture
  if (audio_capture_) {
    try {
      std::cout << "Stopping audio capture..." << std::endl;
      audio_capture_->stop_capture();
      audio_capture_.reset(); // Reset audio capture object
      std::cout << "Audio capture stopped" << std::endl;
    } catch (const std::exception &e) {
      std::cout << "Error stopping audio capture: " << e.what() << std::endl;
    }
  }

  // On macOS, ensure ScreenCaptureKit resources are completely cleaned up
#ifdef __APPLE__
  try {
    std::cout << "Cleaning up macOS screen capture resources..." << std::endl;
    media::cleanup_macos_screen_capture();
    // Additional wait time to ensure macOS resources are fully released
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    std::cout << "macOS screen capture resource cleanup completed" << std::endl;
  } catch (const std::exception &e) {
    std::cout << "Error cleaning up macOS screen capture: " << e.what()
              << std::endl;
  }
#endif

  // Update button state and icon
  if (video_record_button_) {
    // Set flag to prevent recursive calls
    updating_button_state_ = true;

    // Force set button to inactive state
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(video_record_button_),
                                 FALSE);

    // Ensure icon object is valid and reset
    if (video_off_image_ && GTK_IS_WIDGET(video_off_image_)) {
      // Ensure icon is visible
      gtk_widget_set_visible(video_off_image_, TRUE);
      gtk_button_set_child(GTK_BUTTON(video_record_button_), video_off_image_);

      // Reapply CSS classes to ensure correct styling
      gtk_widget_remove_css_class(video_record_button_, "recording");
      if (!gtk_widget_has_css_class(video_record_button_, "upload-button")) {
        gtk_widget_add_css_class(video_record_button_, "upload-button");
      }

      gtk_widget_set_tooltip_text(GTK_WIDGET(video_record_button_),
                                  "Start video recording/desktop capture");
    } else {
      // If icon object is invalid, recreate it
      std::cout << "Warning: video_off_image_ is invalid, recreating icon" << std::endl;
      std::string icon_path_base = "src/gui/";
      video_off_image_ = gtk_picture_new_for_filename(
          (icon_path_base + "video-off.png").c_str());
      if (!video_off_image_) {
        video_off_image_ = gtk_label_new("⏹");
      }
      gtk_widget_set_size_request(video_off_image_, 24, 24);
      gtk_widget_set_visible(video_off_image_, TRUE);
      gtk_button_set_child(GTK_BUTTON(video_record_button_), video_off_image_);

      // Reapply CSS classes
      gtk_widget_remove_css_class(video_record_button_, "recording");
      if (!gtk_widget_has_css_class(video_record_button_, "upload-button")) {
        gtk_widget_add_css_class(video_record_button_, "upload-button");
      }

      gtk_widget_set_tooltip_text(GTK_WIDGET(video_record_button_),
                                  "Start video recording/desktop capture");
    }

    // Re-enable button
    gtk_widget_set_sensitive(video_record_button_, TRUE);
    updating_button_state_ = false;
    std::cout
        << "Button state switched to inactive, icon updated to video-off, button re-enabled"
        << std::endl;
  }

  // Hide video display window
  if (enhanced_video_window_) {
    try {
      enhanced_video_window_->hide();
    } catch (const std::exception &e) {
      std::cout << "Error hiding video window: " << e.what() << std::endl;
    }
  }

  std::cout << "Recording stopped - video and audio capture ended" << std::endl;

  // Reset stop flag
  stopping = false;

  // Verify state synchronization
  verify_button_state();
}

void ChatView::verify_button_state() {
  if (!video_record_button_) {
    return;
  }

  gboolean button_active =
      gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(video_record_button_));

  // Check if button state matches actual recording state
  if (button_active != is_recording_) {
    std::cout << "State inconsistency detected: button state="
              << (button_active ? "active(on)" : "inactive(off)")
              << ", recording state=" << (is_recording_ ? "recording" : "stopped")
              << std::endl;

    // Synchronize button state to actual recording state
    updating_button_state_ = true;
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(video_record_button_),
                                 is_recording_);

    // Update icon based on toggle state: active=video-on, inactive=video-off
    if (is_recording_) {
      // Recording, button should be active, show video-on icon
      if (video_off_image_) {
        gtk_button_set_child(GTK_BUTTON(video_record_button_),
                             video_off_image_);
        gtk_widget_set_tooltip_text(GTK_WIDGET(video_record_button_),
                                    "Stop video recording/desktop capture");
      }
      std::cout << "Sync: set to active state, show video-on icon" << std::endl;
    } else {
      // Not recording, button should be inactive, show video-off icon
      if (video_off_image_) {
        gtk_button_set_child(GTK_BUTTON(video_record_button_),
                             video_off_image_);
        gtk_widget_set_tooltip_text(GTK_WIDGET(video_record_button_),
                                    "Start video recording/desktop capture");
      }
      std::cout << "Sync: set to inactive state, show video-off icon" << std::endl;
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

  // Show dialog, pass callback function
  video_source_dialog_->show(main_widget_,
                             [this](VideoSourceDialog::VideoSource source) {
                               on_video_source_selected(source);
                             });
}

void ChatView::on_video_source_selected(VideoSourceDialog::VideoSource source) {
  switch (source) {
  case VideoSourceDialog::VideoSource::DESKTOP_CAPTURE:
    std::cout << "User selected: desktop recording" << std::endl;
    // Activate button and update icon
    if (video_record_button_) {
      updating_button_state_ = true;
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(video_record_button_),
                                   TRUE);
      if (video_off_image_ && GTK_IS_WIDGET(video_off_image_)) {
        gtk_widget_set_visible(video_off_image_, TRUE);
        gtk_button_set_child(GTK_BUTTON(video_record_button_),
                             video_off_image_);
        // Add recording state CSS class
        gtk_widget_add_css_class(video_record_button_, "recording");
        gtk_widget_set_tooltip_text(GTK_WIDGET(video_record_button_),
                                    "Stop recording");
      }
      // Disable button to prevent clicks during recording
      gtk_widget_set_sensitive(video_record_button_, FALSE);
      updating_button_state_ = false;
    }
    start_desktop_capture();
    break;
  case VideoSourceDialog::VideoSource::CAMERA:
    std::cout << "User selected: camera" << std::endl;
    // Activate button and update icon
    if (video_record_button_) {
      updating_button_state_ = true;
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(video_record_button_),
                                   TRUE);
      if (video_off_image_ && GTK_IS_WIDGET(video_off_image_)) {
        gtk_widget_set_visible(video_off_image_, TRUE);
        gtk_button_set_child(GTK_BUTTON(video_record_button_),
                             video_off_image_);
        // Add recording state CSS class
        gtk_widget_add_css_class(video_record_button_, "recording");
        gtk_widget_set_tooltip_text(GTK_WIDGET(video_record_button_),
                                    "Stop recording");
      }
      // Disable button to prevent clicks during recording
      gtk_widget_set_sensitive(video_record_button_, FALSE);
      updating_button_state_ = false;
    }
    start_camera_capture();
    break;
  case VideoSourceDialog::VideoSource::CANCEL:
    std::cout << "User cancelled selection" << std::endl;
    // Reset button state
    if (video_record_button_) {
      updating_button_state_ = true;
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(video_record_button_),
                                   FALSE);
      if (video_off_image_) {
        gtk_button_set_child(GTK_BUTTON(video_record_button_),
                             video_off_image_);
        gtk_widget_set_tooltip_text(GTK_WIDGET(video_record_button_),
                                    "Start video recording/desktop capture");
      }
      updating_button_state_ = false;
    }
    break;
  }
}

void ChatView::reset_state() {
  std::cout << "Starting to reset ChatView state..." << std::endl;

  // Directly stop recording activity, don't call stop_recording to avoid duplicate cleanup
  if (is_recording_) {
    is_recording_ = false;

    // Directly stop video and audio capture, don't call macOS cleanup functions
    if (video_capture_) {
      try {
        video_capture_->stop_capture();
      } catch (const std::exception &e) {
        std::cout << "Exception stopping video capture during state reset: " << e.what() << std::endl;
      }
    }

    if (audio_capture_) {
      try {
        audio_capture_->stop_capture();
      } catch (const std::exception &e) {
        std::cout << "Exception stopping audio capture during state reset: " << e.what() << std::endl;
      }
    }
  }

  // Reset recording state
  is_recording_ = false;
  updating_button_state_ = false;

  // Clean up video capture
  if (video_capture_) {
    video_capture_.reset();
  }

  // Clean up audio capture
  if (audio_capture_) {
    audio_capture_.reset();
  }

  // reset button status
  if (video_record_button_) {
    updating_button_state_ = true;
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(video_record_button_),
                                 FALSE);
    gtk_widget_set_sensitive(video_record_button_, TRUE);

    // Reset button icon to off state
    if (video_off_image_ && GTK_IS_WIDGET(video_off_image_)) {
      gtk_widget_set_visible(video_off_image_, TRUE);
      gtk_button_set_child(GTK_BUTTON(video_record_button_), video_off_image_);
      gtk_widget_remove_css_class(video_record_button_, "recording");
      gtk_widget_set_tooltip_text(GTK_WIDGET(video_record_button_), "Start recording");
    }
    updating_button_state_ = false;
  }

  // Hide video display window
  if (enhanced_video_window_) {
    try {
      enhanced_video_window_->hide();
    } catch (const std::exception &e) {
      std::cout << "Error hiding video window: " << e.what() << std::endl;
    }
  }

  // Clean up cached frame data
  cached_video_frame_.reset();
  cached_audio_frames_.clear();

  // Reset timestamps
  last_video_update_ = std::chrono::steady_clock::now();
  last_audio_update_ = std::chrono::steady_clock::now();

  std::cout << "ChatView state reset completed" << std::endl;
}

void ChatView::set_session_manager(ChatSessionManager *session_manager) {
  session_manager_ = session_manager;
}

void ChatView::load_session_messages(const std::string &session_id) {
  if (!session_manager_) {
    std::cerr << "Session manager not set" << std::endl;
    return;
  }

  // Clear current chat history
  clear_chat();

  // Get messages for specified session
  auto session = session_manager_->get_session(session_id);
  if (!session) {
    std::cout << "Session not found: " << session_id << std::endl;
    return;
  }

  // Load all messages in the session
  for (const auto &message : session->get_messages()) {
    add_message(message.content, message.is_user);
  }

  std::cout << "Loaded " << session->get_messages().size()
            << " messages for session: " << session_id << std::endl;
}

void ChatView::set_model_manager(core::ModelManager *model_manager) {
  model_manager_ = model_manager;
  // Update model selector immediately after setting model manager
  update_model_selector();
}

void ChatView::update_model_selector() {
  if (!model_manager_ || !model_selector_) {
    return;
  }

  // Get available model list
  auto available_models = model_manager_->getAllModels();
  
  if (available_models.empty()) {
    // If no models available, show prompt message
    const char *no_models[] = {"No models available", NULL};
    GtkStringList *string_list = gtk_string_list_new(no_models);
    gtk_drop_down_set_model(GTK_DROP_DOWN(model_selector_), G_LIST_MODEL(string_list));
    return;
  }

  // Create model name array
  std::vector<const char*> model_names;
  for (const auto &model : available_models) {
    model_names.push_back(model.name.c_str());
  }
  model_names.push_back(NULL); // End with NULL

  // Update dropdown menu
  GtkStringList *string_list = gtk_string_list_new(model_names.data());
  gtk_drop_down_set_model(GTK_DROP_DOWN(model_selector_), G_LIST_MODEL(string_list));
  
  // Select first model by default
  if (!available_models.empty()) {
    gtk_drop_down_set_selected(GTK_DROP_DOWN(model_selector_), 0);
  }
  
  std::cout << "Updated model selector with " << available_models.size() << " models" << std::endl;
}

std::string ChatView::generate_ai_response(const std::string &message) {
  std::cout << "[DEBUG] ChatView::generate_ai_response() called with message: " << message.substr(0, 50) << "..." << std::endl;
  
  if (!model_manager_) {
    std::cout << "[DEBUG] ChatView: Model manager not available" << std::endl;
    return "Error: Model manager not available.";
  }
  std::cout << "[DEBUG] ChatView: Model manager is available" << std::endl;

  // Get currently selected model
  if (!model_selector_) {
    std::cout << "[DEBUG] ChatView: Model selector not available" << std::endl;
    return "Error: Model selector not available.";
  }
  std::cout << "[DEBUG] ChatView: Model selector is available" << std::endl;

  // Get selected model index
  guint selected_index = gtk_drop_down_get_selected(GTK_DROP_DOWN(model_selector_));
  std::cout << "[DEBUG] ChatView: Selected model index: " << selected_index << std::endl;
  
  // Get available model list
  auto available_models = model_manager_->getAllModels();
  std::cout << "[DEBUG] ChatView: Available models count: " << available_models.size() << std::endl;
  
  if (available_models.empty()) {
    std::cout << "[DEBUG] ChatView: No models available" << std::endl;
    return "Error: No models available for text generation.";
  }
  
  if (selected_index >= available_models.size()) {
    std::cout << "[DEBUG] ChatView: Invalid model selection - index " << selected_index << " >= " << available_models.size() << std::endl;
    return "Error: Invalid model selection.";
  }

  // Get selected model
  const auto& selected_model = available_models[selected_index];
  std::string model_id = selected_model.name;
  std::cout << "[DEBUG] ChatView: Selected model ID: " << model_id << std::endl;

  try {
    // First try to load model
    std::cout << "[DEBUG] ChatView: Attempting to load model: " << model_id << std::endl;
    bool model_loaded = model_manager_->loadModel(model_id);
    if (!model_loaded) {
      std::cout << "[DEBUG] ChatView: Failed to load model: " << model_id << std::endl;
      return "Error: Failed to load model: " + model_id;
    }
    std::cout << "[DEBUG] ChatView: Model loaded successfully: " << model_id << std::endl;
    
    // Get text generator
     std::cout << "[DEBUG] ChatView: Getting text generator for model: " << model_id << std::endl;
     core::TextGenerator* text_generator = model_manager_->getTextGenerator(model_id);
     if (!text_generator) {
       std::cout << "[DEBUG] ChatView: Failed to get text generator for model: " << model_id << std::endl;
       return "Error: Failed to get text generator for model: " + model_id;
     }
     
     // Check if text generator is available
     if (!text_generator->canGenerate()) {
       std::cout << "[DEBUG] ChatView: Text generator is not ready for generation" << std::endl;
       return "Error: Text generator is not ready for generation";
     }
     
     // Set generation parameters
     core::GenerationParams params;
     params.max_tokens = 512;  // Maximum 512 tokens
     params.temperature = 0.7f;  // Moderate randomness
     params.top_p = 0.9f;
     params.top_k = 40;
     params.repeat_penalty = 1.1f;
     
     // Generate response
     std::cout << "[DEBUG] ChatView: Starting text generation..." << std::endl;
     core::GenerationResult result = text_generator->generate(message, params);
     
     if (result.finished && !result.text.empty()) {
       std::cout << "[DEBUG] ChatView: Text generation completed successfully" << std::endl;
       return result.text;
     } else {
       std::cout << "[DEBUG] ChatView: Text generation failed or returned empty result" << std::endl;
       return "Error: Text generation failed or returned empty result. Stop reason: " + result.stop_reason;
     }
  } catch (const std::exception &e) {
    std::cout << "[DEBUG] ChatView: Exception caught: " << e.what() << std::endl;
    return "Error generating response: " + std::string(e.what());
  }
}

} // namespace gui
} // namespace duorou