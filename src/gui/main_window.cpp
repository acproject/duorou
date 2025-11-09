#include "main_window.h"
#include "../core/application.h"
#include "../core/logger.h"
#include "chat_session_manager.h"
#include "chat_view.h"
#include "image_view.h"
#include "settings_dialog.h"
#include "system_tray.h"

#include <iostream>

namespace duorou {
namespace gui {

MainWindow::MainWindow()
    : window_(nullptr), header_bar_(nullptr), main_box_(nullptr),
      sidebar_(nullptr), content_stack_(nullptr), status_bar_(nullptr),
      paned_(nullptr), toggle_sidebar_button_(nullptr),
      new_chat_button_(nullptr), image_button_(nullptr),
      settings_button_(nullptr), chat_history_box_(nullptr),
  current_view_("chat"), application_(nullptr)
#ifdef __APPLE__
      ,
      macos_tray_(std::make_unique<MacOSTray>())
#endif
#ifdef _WIN32
      ,
      windows_tray_(std::make_unique<WindowsTray>())
#endif
{
}

MainWindow::MainWindow(core::Application *app)
    : window_(nullptr), header_bar_(nullptr), main_box_(nullptr),
      sidebar_(nullptr), content_stack_(nullptr), status_bar_(nullptr),
      paned_(nullptr), toggle_sidebar_button_(nullptr),
      new_chat_button_(nullptr), image_button_(nullptr),
      settings_button_(nullptr), chat_history_box_(nullptr),
  current_view_("chat"), application_(app)
#ifdef __APPLE__
      ,
      macos_tray_(std::make_unique<MacOSTray>())
#endif
#ifdef _WIN32
      ,
      windows_tray_(std::make_unique<WindowsTray>())
#endif
{
}

MainWindow::~MainWindow() {
  // Window may have been destroyed in quit_application
  if (window_) {
    gtk_window_destroy(GTK_WINDOW(window_));
    window_ = nullptr;
  }
}

bool MainWindow::initialize() {
  // Create main window
  window_ = gtk_window_new();
  if (!window_) {
    std::cerr << "Failed to create main window" << std::endl;
    return false;
  }

  // Set window properties
  gtk_window_set_title(GTK_WINDOW(window_), "Duorou - AI Desktop Assistant");
  gtk_window_set_default_size(GTK_WINDOW(window_), 1200, 800);

  // Create main container
  main_box_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
  gtk_window_set_child(GTK_WINDOW(window_), main_box_);

  // Create components
  create_header_bar();
  create_sidebar();
  create_content_area();
  create_status_bar();

  // Setup styling
  setup_styling();

  // Connect signals
  connect_signals();

  // Initialize session manager
  session_manager_ = std::make_unique<ChatSessionManager>();

  // Set session manager callbacks
  session_manager_->set_session_change_callback(
      [this](const std::string &session_id) {
        on_session_changed(session_id);
      });
  session_manager_->set_session_list_change_callback(
      [this]() { on_session_list_changed(); });

  // Initialize sub views
  chat_view_ = std::make_unique<ChatView>();
  image_view_ = std::make_unique<ImageView>();
  settings_dialog_ = std::make_unique<SettingsDialog>(application_);

  if (!chat_view_->initialize() || !image_view_->initialize() ||
      !settings_dialog_->initialize()) {
    std::cerr << "Failed to initialize sub views" << std::endl;
    return false;
  }

  // Set session manager for ChatView
  if (chat_view_ && session_manager_) {
    chat_view_->set_session_manager(session_manager_.get());
  }

  // Set model manager for ChatView
  if (chat_view_ && application_) {
    chat_view_->set_model_manager(application_->getModelManager());
    chat_view_->set_config_manager(application_->getConfigManager());
  }

  // Add sub views to stack
  gtk_stack_add_named(GTK_STACK(content_stack_), chat_view_->get_widget(),
                      "chat");
  gtk_stack_add_named(GTK_STACK(content_stack_), image_view_->get_widget(),
                      "image");

  // Show chat interface by default
  switch_to_chat();

  // Initialize system tray
#ifdef __APPLE__
  if (macos_tray_ && macos_tray_->initialize()) {
    std::cout << "macOS system tray initialized successfully" << std::endl;

    // Use system icon instead of emoji (emoji causes crashes)
    macos_tray_->setSystemIcon();
    macos_tray_->setTooltip("Duorou - AI Desktop Assistant");

    // Set left click callback to show window
    macos_tray_->setLeftClickCallback([this]() { restore_from_tray(); });

    // Set right click callback to hide window
    macos_tray_->setRightClickCallback([this]() { hide(); });

    // Add menu items
    macos_tray_->addMenuItemWithId("show_window", "Show Window",
                                   [this]() { restore_from_tray(); });

    macos_tray_->addMenuItemWithId("hide_window", "Hide Window", [this]() {
      std::cout << "[MainWindow] Hide Window menu item clicked" << std::endl;
      hide();
    });

    macos_tray_->addSeparator();

    macos_tray_->addMenuItemWithId("new_chat", "New Chat", [this]() {
      restore_from_tray();
      create_new_chat();
    });

    macos_tray_->addMenuItemWithId("settings", "Settings", [this]() {
      restore_from_tray();
      show_settings();
    });

    macos_tray_->addSeparator();

    macos_tray_->addMenuItemWithId("quit", "Quit Duorou", [this]() {
      std::cout << "[MainWindow] Quit Duorou menu item clicked" << std::endl;
      quit_application();
    });

    // Set quit callback function
    macos_tray_->setQuitCallback([this]() { quit_application(); });

    macos_tray_->show();

    // Initialize menu state (window is currently shown)
    macos_tray_->updateWindowStateMenu(true);
  } else {
    std::cerr << "Failed to initialize macOS system tray" << std::endl;
  }
#elif defined(_WIN32)
  if (windows_tray_ && windows_tray_->initialize()) {
    std::cout << "Windows system tray initialized successfully" << std::endl;

    windows_tray_->setSystemIcon();
    windows_tray_->setIconFromFile("src/gui/seo_page_browser_web_window_view_icon.ico");
    windows_tray_->setTooltip("Duorou - AI Desktop Assistant");

    windows_tray_->setLeftClickCallback([this]() { restore_from_tray(); });
    windows_tray_->setRightClickCallback([this]() { hide(); });

    windows_tray_->addMenuItemWithId("show_window", "Show Window",
                                     [this]() { restore_from_tray(); });
    windows_tray_->addMenuItemWithId("hide_window", "Hide Window",
                                     [this]() { hide(); });

    windows_tray_->addSeparator();

    windows_tray_->addMenuItemWithId("new_chat", "New Chat", [this]() {
      restore_from_tray();
      create_new_chat();
    });

    windows_tray_->addMenuItemWithId("settings", "Settings", [this]() {
      restore_from_tray();
      show_settings();
    });

    windows_tray_->addSeparator();

    windows_tray_->addMenuItemWithId("quit", "Quit Duorou", [this]() {
      quit_application();
    });

    windows_tray_->setQuitCallback([this]() { quit_application(); });
    windows_tray_->show();
    windows_tray_->updateWindowStateMenu(true);
  } else {
    std::cerr << "Failed to initialize Windows system tray" << std::endl;
  }
#else
  // Other platforms: not implemented
  std::cout << "System tray feature not implemented for this platform" << std::endl;
#endif

  // Load existing sessions and update chat history list
  if (session_manager_) {
    session_manager_->load_sessions();
    update_chat_history_list();
  }

  std::cout << "Main window initialized successfully" << std::endl;
  return true;
}

void MainWindow::set_application(core::Application *app) { application_ = app; }

void MainWindow::show() {
  if (window_) {
    gtk_window_present(GTK_WINDOW(window_));

    // Update system tray menu state
#ifdef __APPLE__
    if (macos_tray_ && macos_tray_->isAvailable()) {
      macos_tray_->updateWindowStateMenu(true);
    }
#endif
#ifdef _WIN32
    if (windows_tray_ && windows_tray_->isAvailable()) {
      windows_tray_->updateWindowStateMenu(true);
    }
#endif
  }
}

void MainWindow::hide() {
  std::cout << "[MainWindow] hide() method called" << std::endl;
  if (window_) {
    gtk_widget_set_visible(window_, FALSE);
    std::cout << "[MainWindow] Window hidden" << std::endl;

    // Update system tray menu state
#ifdef __APPLE__
    if (macos_tray_ && macos_tray_->isAvailable()) {
      macos_tray_->updateWindowStateMenu(false);
      std::cout << "[MainWindow] Updated tray menu state to hidden"
                << std::endl;
    }
#endif
#ifdef _WIN32
    if (windows_tray_ && windows_tray_->isAvailable()) {
      windows_tray_->updateWindowStateMenu(false);
    }
#endif
  }
}

void MainWindow::set_title(const std::string &title) {
  if (window_) {
    gtk_window_set_title(GTK_WINDOW(window_), title.c_str());
  }
}

void MainWindow::switch_to_chat() {
  if (content_stack_) {
    gtk_stack_set_visible_child_name(GTK_STACK(content_stack_), "chat");
    current_view_ = "chat";
    update_sidebar_buttons(new_chat_button_);

    // Update status label
    if (status_bar_) {
      gtk_label_set_text(GTK_LABEL(status_bar_),
                         "Chat Mode - Ready for conversation");
    }
  }
}

void MainWindow::switch_to_image_generation() {
  if (content_stack_) {
    gtk_stack_set_visible_child_name(GTK_STACK(content_stack_), "image");
    current_view_ = "image";
    update_sidebar_buttons(image_button_);

    // Update status label
    if (status_bar_) {
      gtk_label_set_text(GTK_LABEL(status_bar_),
                         "Image Generation Mode - Ready to create");
    }
  }
}

void MainWindow::show_settings() {
  if (settings_dialog_) {
    settings_dialog_->show(window_);
  }
}

void MainWindow::quit_application() {
  std::cout << "[MainWindow] quit_application() method called" << std::endl;

  // Save session data first
  if (session_manager_) {
    std::cout << "[MainWindow] Saving session data" << std::endl;
    session_manager_->save_sessions();
  }

  // Destroy window
  if (window_) {
    std::cout << "[MainWindow] Destroying window" << std::endl;
    gtk_window_destroy(GTK_WINDOW(window_));
    window_ = nullptr;
  }

  // Call Application's stop method to properly exit the application
  if (application_) {
    std::cout << "[MainWindow] Calling Application::stop()" << std::endl;
    application_->stop();
  } else {
    std::cout << "[MainWindow] Warning: No application instance available"
              << std::endl;
  }

  std::cout << "[MainWindow] Application should exit now" << std::endl;
}

void MainWindow::create_new_chat() {
  if (session_manager_) {
    session_manager_->create_new_session();
    switch_to_chat();
  }
}

void MainWindow::switch_to_chat_session(const std::string &session_id) {
  if (session_manager_) {
    session_manager_->switch_to_session(session_id);
    switch_to_chat();
  }
}

void MainWindow::update_chat_history_list() {
  if (!chat_history_box_)
    return;

  // Clear existing chat history items
  GtkWidget *child = gtk_widget_get_first_child(chat_history_box_);
  while (child) {
    GtkWidget *next = gtk_widget_get_next_sibling(child);
    gtk_box_remove(GTK_BOX(chat_history_box_), child);
    child = next;
  }

  // Add new chat history items
  if (session_manager_) {
    auto sessions = session_manager_->get_all_sessions();
    for (const auto &session : sessions) {
      // Create horizontal container for chat button and delete button
      GtkWidget *item_container = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
      gtk_widget_set_size_request(item_container, -1, 40);

      // Create chat item button
      GtkWidget *chat_item = gtk_button_new();
      gtk_widget_add_css_class(chat_item, "chat-history-item");
      gtk_widget_set_hexpand(chat_item, TRUE);

      std::string display_name = session->get_display_name();
      if (display_name.empty()) {
        display_name = "New Chat";
      }
      gtk_button_set_label(GTK_BUTTON(chat_item), display_name.c_str());
      gtk_widget_set_halign(chat_item, GTK_ALIGN_FILL);

      // Store session ID as data
      g_object_set_data_full(G_OBJECT(chat_item), "session_id",
                             g_strdup(session->get_id().c_str()), g_free);

      // Connect click signal
      g_signal_connect(chat_item, "clicked",
                       G_CALLBACK(on_chat_history_item_clicked), this);

      // Add right click gesture
      GtkGesture *right_click_gesture = gtk_gesture_click_new();
      gtk_gesture_single_set_button(GTK_GESTURE_SINGLE(right_click_gesture),
                                    GDK_BUTTON_SECONDARY);
      g_signal_connect(right_click_gesture, "pressed",
                       G_CALLBACK(on_chat_history_item_right_clicked), this);

      // Store session ID for gesture
      g_object_set_data_full(G_OBJECT(right_click_gesture), "session_id",
                             g_strdup(session->get_id().c_str()), g_free);

      gtk_widget_add_controller(chat_item,
                                GTK_EVENT_CONTROLLER(right_click_gesture));

      // Create delete button
      GtkWidget *delete_button = gtk_button_new_with_label("Delete");
      gtk_widget_add_css_class(delete_button, "delete-button");
      gtk_widget_set_size_request(delete_button, 30, -1);
      gtk_widget_set_tooltip_text(delete_button, "Delete this chat");

      // Store session ID for delete button
      g_object_set_data_full(G_OBJECT(delete_button), "session_id",
                             g_strdup(session->get_id().c_str()), g_free);

      // Connect delete button signal
      g_signal_connect(delete_button, "clicked",
                       G_CALLBACK(on_delete_chat_button_clicked), this);

      // Add buttons to container
      gtk_box_append(GTK_BOX(item_container), chat_item);
      gtk_box_append(GTK_BOX(item_container), delete_button);

      gtk_box_append(GTK_BOX(chat_history_box_), item_container);
    }
  }
}

void MainWindow::on_session_changed(const std::string &session_id) {
  // Handle session switching
  std::cout << "Session changed to: " << session_id << std::endl;

  // Update chat view to display current session messages
  if (chat_view_) {
    chat_view_->load_session_messages(session_id);
  }
}

void MainWindow::on_session_list_changed() {
  // Update UI when session list changes
  update_chat_history_list();
}

void MainWindow::create_header_bar() {
  header_bar_ = gtk_header_bar_new();
  gtk_header_bar_set_show_title_buttons(GTK_HEADER_BAR(header_bar_), TRUE);
  // Create title box with sidebar toggle button and title label
  GtkWidget *title_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
  toggle_sidebar_button_ = gtk_button_new_with_label("Hide Sidebar");
  GtkWidget *title_label = gtk_label_new("Duorou - AI Desktop Assistant");
  gtk_box_append(GTK_BOX(title_box), toggle_sidebar_button_);
  gtk_box_append(GTK_BOX(title_box), title_label);

  gtk_header_bar_set_title_widget(
      GTK_HEADER_BAR(header_bar_),
      title_box);

  gtk_window_set_titlebar(GTK_WINDOW(window_), header_bar_);
}

void MainWindow::create_sidebar() {
  // Create sidebar container
  sidebar_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
  // Allow full collapse by not enforcing minimum width
  gtk_widget_set_size_request(sidebar_, -1, -1);
  gtk_widget_add_css_class(sidebar_, "sidebar");
  gtk_widget_set_margin_start(sidebar_, 10);
  gtk_widget_set_margin_end(sidebar_, 10);
  gtk_widget_set_margin_top(sidebar_, 10);
  gtk_widget_set_margin_bottom(sidebar_, 10);

  // Create "New Chat" button
  new_chat_button_ = gtk_button_new_with_label("New Chat");
  gtk_widget_set_size_request(new_chat_button_, -1, 45);
  gtk_widget_add_css_class(new_chat_button_, "new-chat-button");
  gtk_widget_set_margin_bottom(new_chat_button_, 15);
  gtk_box_append(GTK_BOX(sidebar_), new_chat_button_);

  // Create chat history title
  GtkWidget *history_label = gtk_label_new("Recent Chats");
  gtk_widget_set_halign(history_label, GTK_ALIGN_START);
  gtk_widget_add_css_class(history_label, "section-title");
  gtk_widget_set_margin_bottom(history_label, 10);
  gtk_box_append(GTK_BOX(sidebar_), history_label);

  // Create chat history scroll area
  GtkWidget *history_scrolled = gtk_scrolled_window_new();
  gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(history_scrolled),
                                 GTK_POLICY_NEVER, GTK_POLICY_AUTOMATIC);
  gtk_widget_set_vexpand(history_scrolled, TRUE);

  chat_history_box_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
  gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(history_scrolled),
                                chat_history_box_);
  gtk_box_append(GTK_BOX(sidebar_), history_scrolled);

  // Don't add example items during initialization, session manager will add dynamically

  // Add separator
  GtkWidget *separator = gtk_separator_new(GTK_ORIENTATION_HORIZONTAL);
  gtk_widget_set_margin_top(separator, 15);
  gtk_widget_set_margin_bottom(separator, 15);
  gtk_box_append(GTK_BOX(sidebar_), separator);

  // Create bottom function buttons
  image_button_ = gtk_button_new_with_label("Image Generation");
  settings_button_ = gtk_button_new_with_label("Settings");

  gtk_widget_set_size_request(image_button_, -1, 40);
  gtk_widget_set_size_request(settings_button_, -1, 40);
  gtk_widget_add_css_class(image_button_, "sidebar-button");
  gtk_widget_add_css_class(settings_button_, "sidebar-button");

  gtk_box_append(GTK_BOX(sidebar_), image_button_);
  gtk_box_append(GTK_BOX(sidebar_), settings_button_);
}

void MainWindow::create_content_area() {
  // Create content stack first
  content_stack_ = gtk_stack_new();
  gtk_stack_set_transition_type(GTK_STACK(content_stack_),
                                GTK_STACK_TRANSITION_TYPE_SLIDE_LEFT_RIGHT);
  gtk_stack_set_transition_duration(GTK_STACK(content_stack_), 300);
  gtk_widget_set_hexpand(content_stack_, TRUE);
  gtk_widget_set_vexpand(content_stack_, TRUE);

  // Create a horizontal GtkPaned so the divider is draggable
  paned_ = gtk_paned_new(GTK_ORIENTATION_HORIZONTAL);
  gtk_paned_set_start_child(GTK_PANED(paned_), sidebar_);
  gtk_paned_set_end_child(GTK_PANED(paned_), content_stack_);

  // Set initial position similar to previous fixed width (≈280px)
  gtk_paned_set_position(GTK_PANED(paned_), 300);

  // Add to main container
  gtk_box_append(GTK_BOX(main_box_), paned_);
}

void MainWindow::create_status_bar() {
  status_bar_ = gtk_label_new("Ready");
  gtk_box_append(GTK_BOX(main_box_), status_bar_);
}

void MainWindow::setup_styling() {
  // Load CSS style file
  GtkCssProvider *css_provider = gtk_css_provider_new();

  // Try to load CSS file
  const char *css_file_path = "src/gui/styles.css";

  GFile *css_file = g_file_new_for_path(css_file_path);
  gtk_css_provider_load_from_file(css_provider, css_file);
  g_object_unref(css_file);

  // Apply CSS styles
  gtk_style_context_add_provider_for_display(
      gdk_display_get_default(), GTK_STYLE_PROVIDER(css_provider),
      GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);

  g_object_unref(css_provider);
}

void MainWindow::connect_signals() {
  // Connect window signals
  g_signal_connect(window_, "close-request", G_CALLBACK(on_window_delete_event),
                   this);
  g_signal_connect(window_, "destroy", G_CALLBACK(on_window_destroy), this);

  // Connect button signals
  g_signal_connect(new_chat_button_, "clicked",
                   G_CALLBACK(on_new_chat_button_clicked), this);
  g_signal_connect(image_button_, "clicked",
                   G_CALLBACK(on_image_button_clicked), this);
  g_signal_connect(settings_button_, "clicked",
                   G_CALLBACK(on_settings_button_clicked), this);

  // Toggle sidebar button
  if (toggle_sidebar_button_) {
    g_signal_connect(toggle_sidebar_button_, "clicked",
                     G_CALLBACK(on_toggle_sidebar_button_clicked), this);
  }

  // Track paned position to remember last non-zero width
  if (paned_) {
    g_signal_connect(paned_, "notify::position",
                     G_CALLBACK(on_paned_position_notify), this);
  }
}

void MainWindow::update_sidebar_buttons(GtkWidget *active_button) {
  // Reset all button states
  gtk_widget_remove_css_class(new_chat_button_, "active");
  gtk_widget_remove_css_class(image_button_, "active");

  // Set active button state
  if (active_button) {
    gtk_widget_add_css_class(active_button, "active");
  }
}

// Static callback function implementations
void MainWindow::on_new_chat_button_clicked(GtkWidget *widget,
                                            gpointer user_data) {
  MainWindow *main_window = static_cast<MainWindow *>(user_data);
  main_window->create_new_chat();
}

void MainWindow::on_chat_history_item_clicked(GtkWidget *widget,
                                              gpointer user_data) {
  MainWindow *main_window = static_cast<MainWindow *>(user_data);
  const char *session_id = static_cast<const char *>(
      g_object_get_data(G_OBJECT(widget), "session_id"));
  if (session_id) {
    main_window->switch_to_chat_session(session_id);
  }
}

void MainWindow::on_chat_history_item_right_clicked(GtkGestureClick *gesture,
                                                    gint n_press, gdouble x,
                                                    gdouble y,
                                                    gpointer user_data) {
  MainWindow *main_window = static_cast<MainWindow *>(user_data);
  const char *session_id = static_cast<const char *>(
      g_object_get_data(G_OBJECT(gesture), "session_id"));

  if (session_id) {
    // Create popup menu
    GtkWidget *popover = gtk_popover_new();
    GtkWidget *menu_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);

    // Create rename menu item
    GtkWidget *rename_item = gtk_button_new_with_label("Rename Chat");
    gtk_widget_add_css_class(rename_item, "context-menu-item");
    gtk_widget_set_size_request(rename_item, 150, 35);

    // Store session ID and main window pointer for rename menu item
    g_object_set_data_full(G_OBJECT(rename_item), "session_id",
                           g_strdup(session_id), g_free);
    g_object_set_data(G_OBJECT(rename_item), "popover", popover);
    g_object_set_data(G_OBJECT(rename_item), "main_window", main_window);

    // Connect rename menu item signal
    g_signal_connect(rename_item, "clicked",
                     G_CALLBACK(on_context_menu_rename_clicked), nullptr);

    gtk_box_append(GTK_BOX(menu_box), rename_item);

    // Create delete menu item
    GtkWidget *delete_item = gtk_button_new_with_label("Delete Chat");
    gtk_widget_add_css_class(delete_item, "context-menu-item");
    gtk_widget_set_size_request(delete_item, 150, 35);

    // Store session ID and main window pointer for delete menu item
    g_object_set_data_full(G_OBJECT(delete_item), "session_id",
                           g_strdup(session_id), g_free);
    g_object_set_data(G_OBJECT(delete_item), "popover", popover);
    g_object_set_data(G_OBJECT(delete_item), "main_window", main_window);

    // Connect delete menu item signal
    g_signal_connect(delete_item, "clicked",
                     G_CALLBACK(on_context_menu_delete_clicked), nullptr);

    gtk_box_append(GTK_BOX(menu_box), delete_item);
    gtk_popover_set_child(GTK_POPOVER(popover), menu_box);

    // Set popup menu position
    GtkWidget *chat_item =
        gtk_event_controller_get_widget(GTK_EVENT_CONTROLLER(gesture));
    gtk_widget_set_parent(popover, chat_item);

    // Show popup menu
    gtk_popover_popup(GTK_POPOVER(popover));
  }
}

void MainWindow::on_context_menu_rename_clicked(GtkWidget *widget,
                                                gpointer user_data) {
  MainWindow *main_window = static_cast<MainWindow *>(
      g_object_get_data(G_OBJECT(widget), "main_window"));
  const char *session_id = static_cast<const char *>(
      g_object_get_data(G_OBJECT(widget), "session_id"));
  GtkWidget *popover =
      static_cast<GtkWidget *>(g_object_get_data(G_OBJECT(widget), "popover"));

  if (session_id && main_window && main_window->session_manager_) {
    // Close popup menu
    if (popover) {
      gtk_popover_popdown(GTK_POPOVER(popover));
    }

    // Get current session
    auto session = main_window->session_manager_->get_session(session_id);
    if (!session) {
      return;
    }

    // Create rename dialog
    GtkWidget *dialog = gtk_dialog_new_with_buttons(
        "Rename Chat Session", GTK_WINDOW(main_window->window_),
        GTK_DIALOG_MODAL, "Cancel", GTK_RESPONSE_CANCEL, "OK",
        GTK_RESPONSE_OK, nullptr);

    // Create input box
    GtkWidget *entry = gtk_entry_new();
    std::string current_name = session->get_custom_name();
    if (current_name.empty()) {
      current_name = session->get_title();
    }
    gtk_editable_set_text(GTK_EDITABLE(entry), current_name.c_str());
    gtk_entry_set_placeholder_text(GTK_ENTRY(entry), "Enter new name...");

    // Add to dialog
    GtkWidget *content_area = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
    gtk_box_append(GTK_BOX(content_area), entry);

    // Store data for callback
    struct RenameData {
      MainWindow *main_window;
      std::string session_id;
      GtkWidget *entry;
    };
    RenameData *rename_data = new RenameData{main_window, session_id, entry};
    g_object_set_data_full(G_OBJECT(dialog), "rename_data", rename_data,
                           [](gpointer data) { delete static_cast<RenameData *>(data); });

    // Show dialog and handle response
    gtk_widget_show(dialog);
    g_signal_connect(dialog, "response",
                     G_CALLBACK(on_rename_dialog_response), nullptr);
  }
}

void MainWindow::on_rename_dialog_response(GtkDialog *dialog, gint response_id,
                                           gpointer user_data) {
  struct RenameData {
    MainWindow *main_window;
    std::string session_id;
    GtkWidget *entry;
  };
  RenameData *data = static_cast<RenameData *>(
      g_object_get_data(G_OBJECT(dialog), "rename_data"));

  if (data && response_id == GTK_RESPONSE_OK) {
    const char *new_name = gtk_editable_get_text(GTK_EDITABLE(data->entry));
    if (new_name && strlen(new_name) > 0) {
      data->main_window->session_manager_->set_session_custom_name(
          data->session_id, new_name);
      data->main_window->update_chat_history_list();
    }
  }

  gtk_window_destroy(GTK_WINDOW(dialog));
}

void MainWindow::on_context_menu_delete_clicked(GtkWidget *widget,
                                                gpointer user_data) {
  MainWindow *main_window = static_cast<MainWindow *>(
      g_object_get_data(G_OBJECT(widget), "main_window"));
  const char *session_id = static_cast<const char *>(
      g_object_get_data(G_OBJECT(widget), "session_id"));
  GtkWidget *popover =
      static_cast<GtkWidget *>(g_object_get_data(G_OBJECT(widget), "popover"));

  if (session_id && main_window && main_window->session_manager_) {
    // Delete session
    main_window->session_manager_->delete_session(session_id);

    // Update chat history list
    main_window->update_chat_history_list();

    // If deleted session is current session, create new session
    if (main_window->session_manager_->get_current_session_id() == session_id) {
      main_window->create_new_chat();
    }
  }

  // Close popup menu
  if (popover) {
    gtk_popover_popdown(GTK_POPOVER(popover));
  }
}

void MainWindow::on_delete_chat_button_clicked(GtkWidget *widget,
                                               gpointer user_data) {
  MainWindow *main_window = static_cast<MainWindow *>(user_data);
  const char *session_id = static_cast<const char *>(
      g_object_get_data(G_OBJECT(widget), "session_id"));

  if (session_id && main_window->session_manager_) {
    // Delete session
    main_window->session_manager_->delete_session(session_id);

    // Update chat history list
    main_window->update_chat_history_list();

    // If deleted session is current session, create new session
    if (main_window->session_manager_->get_current_session_id() == session_id) {
      main_window->create_new_chat();
    }
  }
}

void MainWindow::on_image_button_clicked(GtkWidget *widget,
                                         gpointer user_data) {
  MainWindow *main_window = static_cast<MainWindow *>(user_data);
  main_window->switch_to_image_generation();
}

void MainWindow::on_settings_button_clicked(GtkWidget *widget,
                                            gpointer user_data) {
  MainWindow *main_window = static_cast<MainWindow *>(user_data);
  main_window->show_settings();
}

void MainWindow::on_toggle_sidebar_button_clicked(GtkWidget *widget,
                                                  gpointer user_data) {
  MainWindow *self = static_cast<MainWindow *>(user_data);
  if (!self || !self->paned_)
    return;

  int pos = gtk_paned_get_position(GTK_PANED(self->paned_));
  if (pos > 0) {
    // Remember current width and collapse to 0
    self->last_sidebar_width_ = pos;
    gtk_paned_set_position(GTK_PANED(self->paned_), 0);
    if (self->toggle_sidebar_button_) {
      gtk_button_set_label(GTK_BUTTON(self->toggle_sidebar_button_), "Show Sidebar");
    }
  } else {
    // Restore to last width or default if invalid
    int target = self->last_sidebar_width_ > 50 ? self->last_sidebar_width_ : 300;
    gtk_paned_set_position(GTK_PANED(self->paned_), target);
    if (self->toggle_sidebar_button_) {
      gtk_button_set_label(GTK_BUTTON(self->toggle_sidebar_button_), "Hide Sidebar");
    }
  }
}

void MainWindow::on_paned_position_notify(GObject *object, GParamSpec *pspec,
                                          gpointer user_data) {
  MainWindow *self = static_cast<MainWindow *>(user_data);
  if (!self || !self->paned_)
    return;

  int pos = gtk_paned_get_position(GTK_PANED(self->paned_));
  if (pos > 0) {
    self->last_sidebar_width_ = pos;
    if (self->toggle_sidebar_button_) {
      gtk_button_set_label(GTK_BUTTON(self->toggle_sidebar_button_), "Hide Sidebar");
    }
  } else {
    if (self->toggle_sidebar_button_) {
      gtk_button_set_label(GTK_BUTTON(self->toggle_sidebar_button_), "Show Sidebar");
    }
  }
}

gboolean MainWindow::on_window_delete_event(GtkWindow *window,
                                            gpointer user_data) {
  MainWindow *main_window = static_cast<MainWindow *>(user_data);

  // Save session data
  if (main_window->session_manager_) {
    main_window->session_manager_->save_sessions();
  }

#ifdef __APPLE__
  // On macOS, if system tray is available, hide window instead of quitting
  if (main_window->macos_tray_ && main_window->macos_tray_->isAvailable()) {
    main_window->hide();
    return TRUE; // Prevent window closing, just hide
  }
#endif
#ifdef _WIN32
  // On Windows, if system tray is available, hide window instead of quitting
  if (main_window->windows_tray_ && main_window->windows_tray_->isAvailable()) {
    main_window->hide();
    return TRUE;
  }
#endif

  // If system tray is not available, exit normally
  return FALSE; // Allow window to close normally
}

void MainWindow::on_window_destroy(GtkWidget *widget, gpointer user_data) {
  // In GTK4, usually no need to manually call exit function
  // Application will handle automatically
}

void MainWindow::restore_from_tray() {
  if (window_) {
    show();
    gtk_window_present(GTK_WINDOW(window_));

    // Ensure window gets focus
    gtk_window_set_focus_visible(GTK_WINDOW(window_), TRUE);

    // Update system tray menu state
#ifdef __APPLE__
    if (macos_tray_ && macos_tray_->isAvailable()) {
      macos_tray_->updateWindowStateMenu(true);
    }
#endif
#ifdef _WIN32
    if (windows_tray_ && windows_tray_->isAvailable()) {
      windows_tray_->updateWindowStateMenu(true);
    }
#endif
  }
}

void MainWindow::set_tray_status(const std::string &status) {
#ifdef __APPLE__
  if (macos_tray_ && macos_tray_->isAvailable()) {
    if (status == "idle") {
      macos_tray_->setIcon("Flower"); // Flower indicates idle
      macos_tray_->setTooltip("Duorou - Ready");
    } else if (status == "processing") {
      macos_tray_->setIcon("Lightning"); // Lightning indicates processing
      macos_tray_->setTooltip("Duorou - Processing...");
    } else if (status == "error") {
      macos_tray_->setIcon("Error"); // Red X indicates error
      macos_tray_->setTooltip("Duorou - Error occurred");
    } else if (status == "success") {
      macos_tray_->setIcon("Success"); // Green check indicates success
      macos_tray_->setTooltip("Duorou - Task completed");
    } else {
      macos_tray_->setIcon("Flower"); // Default icon
      macos_tray_->setTooltip("Duorou - AI Desktop Assistant");
    }
  }
#endif
#ifdef _WIN32
  if (windows_tray_ && windows_tray_->isAvailable()) {
    if (status == "idle") {
      windows_tray_->setIcon("Flower");
      windows_tray_->setTooltip("Duorou - Ready");
    } else if (status == "processing") {
      windows_tray_->setIcon("Lightning");
      windows_tray_->setTooltip("Duorou - Processing...");
    } else if (status == "error") {
      windows_tray_->setIcon("Error");
      windows_tray_->setTooltip("Duorou - Error occurred");
    } else if (status == "success") {
      windows_tray_->setIcon("Success");
      windows_tray_->setTooltip("Duorou - Task completed");
    } else {
      windows_tray_->setIcon("Flower");
      windows_tray_->setTooltip("Duorou - AI Desktop Assistant");
    }
  }
#endif
}

} // namespace gui
} // namespace duorou