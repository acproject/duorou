#ifndef DUOROU_GUI_MAIN_WINDOW_H
#define DUOROU_GUI_MAIN_WINDOW_H

#if defined(__has_include) && __has_include(<gtk/gtk.h>)
#include <gtk/gtk.h>
#define DUOROU_HAS_GTK 1
#else
#define DUOROU_HAS_GTK 0
// GTK 占位类型，便于在未安装 GTK 头文件的环境下编译通过
typedef void* GtkWidget;
typedef void* GtkWindow;
typedef void* GtkDialog;
typedef void* GtkGestureClick;
typedef void* gpointer;
typedef int gint;
typedef double gdouble;
typedef int gboolean;
typedef unsigned int guint;
// 额外的占位类型以便信号回调声明
typedef void* GObject;
typedef void* GParamSpec;
 typedef void* GFile;

 // 常用 GTK 类型转换宏占位
 #ifndef GTK_WINDOW
 #define GTK_WINDOW(x) (x)
 #endif
 #ifndef GTK_STACK
 #define GTK_STACK(x) (x)
 #endif
 #ifndef GTK_LABEL
 #define GTK_LABEL(x) (x)
 #endif
 #ifndef GTK_BOX
 #define GTK_BOX(x) (x)
 #endif
 #ifndef GTK_PANED
 #define GTK_PANED(x) (x)
 #endif

 // 常用 GTK4 函数占位（编辑器/索引器友好，运行时无影响）
 #ifndef gtk_window_new
 #define gtk_window_new(...) ((GtkWidget*)nullptr)
 #endif
 #ifndef gtk_window_set_title
 #define gtk_window_set_title(...) ((void)0)
 #endif
 #ifndef gtk_window_set_default_size
 #define gtk_window_set_default_size(...) ((void)0)
 #endif
 #ifndef gtk_window_set_child
 #define gtk_window_set_child(...) ((void)0)
 #endif
 #ifndef gtk_window_present
 #define gtk_window_present(...) ((void)0)
 #endif
 #ifndef gtk_window_destroy
 #define gtk_window_destroy(...) ((void)0)
 #endif

 #ifndef gtk_box_new
 #define gtk_box_new(...) ((GtkWidget*)nullptr)
 #endif
 #ifndef gtk_box_append
 #define gtk_box_append(...) ((void)0)
 #endif

 #ifndef gtk_stack_new
 #define gtk_stack_new(...) ((GtkWidget*)nullptr)
 #endif
 #ifndef gtk_stack_add_named
 #define gtk_stack_add_named(...) ((void)0)
 #endif
 #ifndef gtk_stack_set_visible_child_name
 #define gtk_stack_set_visible_child_name(...) ((void)0)
 #endif

 #ifndef gtk_label_new
 #define gtk_label_new(...) ((GtkWidget*)nullptr)
 #endif
 #ifndef gtk_label_set_text
 #define gtk_label_set_text(...) ((void)0)
 #endif

 #ifndef gtk_paned_new
 #define gtk_paned_new(...) ((GtkWidget*)nullptr)
 #endif
 #ifndef gtk_paned_set_start_child
 #define gtk_paned_set_start_child(...) ((void)0)
 #endif
 #ifndef gtk_paned_set_end_child
 #define gtk_paned_set_end_child(...) ((void)0)
 #endif
 #ifndef gtk_paned_set_position
 #define gtk_paned_set_position(...) ((void)0)
 #endif

 #ifndef gtk_widget_set_visible
 #define gtk_widget_set_visible(...) ((void)0)
 #endif
 #ifndef gtk_widget_hide
 #define gtk_widget_hide(...) ((void)0)
 #endif
 #ifndef gtk_widget_show
 #define gtk_widget_show(...) ((void)0)
 #endif

 // CSS/Display/GLib 占位
 typedef void* GtkCssProvider; typedef void* GtkStyleProvider;
 #ifndef GTK_STYLE_PROVIDER
 #define GTK_STYLE_PROVIDER(x) ((GtkStyleProvider*)(x))
 #endif
 #ifndef GTK_STYLE_PROVIDER_PRIORITY_APPLICATION
 #define GTK_STYLE_PROVIDER_PRIORITY_APPLICATION 600
 #endif
 #ifndef gdk_display_get_default
 #define gdk_display_get_default(...) ((void*)nullptr)
 #endif
 #ifndef gtk_css_provider_new
 #define gtk_css_provider_new(...) ((GtkCssProvider*)nullptr)
 #endif
 #ifndef gtk_css_provider_load_from_file
 #define gtk_css_provider_load_from_file(...) ((void)0)
 #endif
 #ifndef g_file_new_for_path
 #define g_file_new_for_path(...) ((GFile*)nullptr)
 #endif
 #ifndef g_object_unref
 #define g_object_unref(...) ((void)0)
 #endif

 // 信号占位
 #ifndef g_signal_connect
 #define g_signal_connect(...) (0)
 #endif
 #ifndef G_CALLBACK
 #define G_CALLBACK(f) ((void*)(f))
 #endif
#endif
#include <memory>
#include <string>
#include <vector>

#ifdef __APPLE__
#include "../platform/macos_tray.h"
#endif

namespace duorou {
namespace core {
class Application;
}
namespace gui {

class ChatView;
class ImageView;
class SettingsDialog;
class ChatSessionManager;
class SystemTray;

/**
 * Main window class - manages the main interface of the entire application
 * Contains switching between chat interface, image generation interface and settings panel
 */
class MainWindow {
public:
  MainWindow();
  explicit MainWindow(core::Application *app);
  ~MainWindow();

  // Disable copy constructor and assignment
  MainWindow(const MainWindow &) = delete;
  MainWindow &operator=(const MainWindow &) = delete;

  /**
   * Initialize main window
   * @return Returns true on success, false on failure
   */
  bool initialize();

  /**
   * Show main window
   */
  void show();

  /**
   * Hide main window
   */
  void hide();

  /**
   * Get GTK window pointer
   * @return GTK window pointer
   */
  GtkWidget *get_window() const { return window_; }

  /**
   * Set window title
   * @param title Window title
   */
  void set_title(const std::string &title);

  /**
   * Switch to chat interface
   */
  void switch_to_chat();

  /**
   * Switch to image generation interface
   */
  void switch_to_image_generation();

  /**
   * Show settings dialog
   */
  void show_settings();

  /**
   * Quit application
   */
  void quit_application();

  /**
   * Set application instance reference
   * @param app Application instance pointer
   */
  void set_application(core::Application *app);

  /**
   * Set system tray status
   * @param status Status description (e.g.: "idle", "processing", "error")
   */
  void set_tray_status(const std::string &status);

  /**
   * Restore window display from system tray
   */
  void restore_from_tray();

  /**
   * Create new chat session
   */
  void create_new_chat();

  /**
   * Switch to specified chat session
   * @param session_id Session ID
   */
  void switch_to_chat_session(const std::string &session_id);

private:
  // GTK components
  GtkWidget *window_;        // Main window
  GtkWidget *header_bar_;    // Header bar
  GtkWidget *main_box_;      // Main container
  GtkWidget *sidebar_;       // Sidebar
  GtkWidget *content_stack_; // Content stack
  GtkWidget *status_bar_;    // Status bar
  GtkWidget *paned_;         // Draggable paned container
  GtkWidget *toggle_sidebar_button_; // Toggle sidebar button

  // Sidebar buttons
  GtkWidget *new_chat_button_;  // New chat button
  GtkWidget *image_button_;     // Image generation button
  GtkWidget *settings_button_;  // Settings button
  GtkWidget *chat_history_box_; // Chat history container

  // UI components
  std::unique_ptr<ChatView> chat_view_;
  std::unique_ptr<ImageView> image_view_;
  std::unique_ptr<SettingsDialog> settings_dialog_;
  std::unique_ptr<ChatSessionManager> session_manager_;
  std::unique_ptr<SystemTray> system_tray_;

#ifdef __APPLE__
  std::unique_ptr<MacOSTray> macos_tray_;
#endif

  // Current view state
  std::string current_view_;

  // Application instance reference
  core::Application *application_;

  // Remember last non-zero sidebar width for restore
  int last_sidebar_width_ = 300;

  /**
   * Create header bar
   */
  void create_header_bar();

  /**
   * Create sidebar
   */
  void create_sidebar();

  /**
   * Create content area
   */
  void create_content_area();

  /**
   * Create status bar
   */
  void create_status_bar();

  /**
   * Setup window styling
   */
  void setup_styling();

  /**
   * Connect signal handlers
   */
  void connect_signals();

  /**
   * Update sidebar button state
   * @param active_button Currently active button
   */
  void update_sidebar_buttons(GtkWidget *active_button);

  /**
   * Update chat history list
   */
  void update_chat_history_list();

  /**
   * Session change callback
   * @param session_id New session ID
   */
  void on_session_changed(const std::string &session_id);

  /**
   * Session list change callback
   */
  void on_session_list_changed();

  // Static callback functions
  static void on_new_chat_button_clicked(GtkWidget *widget, gpointer user_data);
  static void on_chat_history_item_clicked(GtkWidget *widget,
                                           gpointer user_data);
  static void on_chat_history_item_right_clicked(GtkGestureClick *gesture,
                                                 gint n_press, gdouble x,
                                                 gdouble y, gpointer user_data);
  static void on_context_menu_rename_clicked(GtkWidget *widget,
                                             gpointer user_data);
  static void on_rename_dialog_response(GtkDialog *dialog, gint response_id,
                                        gpointer user_data);
  static void on_context_menu_delete_clicked(GtkWidget *widget,
                                             gpointer user_data);
  static void on_delete_chat_button_clicked(GtkWidget *widget,
                                            gpointer user_data);
  static void on_image_button_clicked(GtkWidget *widget, gpointer user_data);
  static void on_settings_button_clicked(GtkWidget *widget, gpointer user_data);
  static gboolean on_window_delete_event(GtkWindow *window, gpointer user_data);
  static void on_window_destroy(GtkWidget *widget, gpointer user_data);
  static void on_toggle_sidebar_button_clicked(GtkWidget *widget, gpointer user_data);
  static void on_paned_position_notify(GObject *object, GParamSpec *pspec, gpointer user_data);
};

} // namespace gui
} // namespace duorou

#endif // DUOROU_GUI_MAIN_WINDOW_H