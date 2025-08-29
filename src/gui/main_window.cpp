#include "main_window.h"
#include "chat_view.h"
#include "image_view.h"
#include "settings_dialog.h"
#include "chat_session_manager.h"
#include "system_tray.h"
#include "../core/logger.h"
#include "../core/application.h"

#include <iostream>

namespace duorou {
namespace gui {

MainWindow::MainWindow()
    : window_(nullptr)
    , header_bar_(nullptr)
    , main_box_(nullptr)
    , sidebar_(nullptr)
    , content_stack_(nullptr)
    , status_bar_(nullptr)
    , new_chat_button_(nullptr)
    , image_button_(nullptr)
    , settings_button_(nullptr)
    , chat_history_box_(nullptr)
    , current_view_("chat")
    , application_(nullptr)
#ifdef __APPLE__
    , macos_tray_(std::make_unique<MacOSTray>())
#endif
{
}

MainWindow::MainWindow(core::Application* app)
    : window_(nullptr)
    , header_bar_(nullptr)
    , main_box_(nullptr)
    , sidebar_(nullptr)
    , content_stack_(nullptr)
    , status_bar_(nullptr)
    , new_chat_button_(nullptr)
    , image_button_(nullptr)
    , settings_button_(nullptr)
    , chat_history_box_(nullptr)
    , current_view_("chat")
    , application_(app)
#ifdef __APPLE__
    , macos_tray_(std::make_unique<MacOSTray>())
#endif
{
}

MainWindow::~MainWindow() {
    // 窗口可能已经在quit_application中被销毁
    if (window_) {
        gtk_window_destroy(GTK_WINDOW(window_));
        window_ = nullptr;
    }
}

bool MainWindow::initialize() {
    // 创建主窗口
    window_ = gtk_window_new();
    if (!window_) {
        std::cerr << "Failed to create main window" << std::endl;
        return false;
    }

    // 设置窗口属性
    gtk_window_set_title(GTK_WINDOW(window_), "Duorou - AI Desktop Assistant");
    gtk_window_set_default_size(GTK_WINDOW(window_), 1200, 800);

    // 创建主容器
    main_box_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_window_set_child(GTK_WINDOW(window_), main_box_);

    // 创建各个组件
    create_header_bar();
    create_sidebar();
    create_content_area();
    create_status_bar();

    // 设置样式
    setup_styling();

    // 连接信号
    connect_signals();

    // 初始化会话管理器
    session_manager_ = std::make_unique<ChatSessionManager>();
    
    // 设置会话管理器回调
    session_manager_->set_session_change_callback(
        [this](const std::string& session_id) {
            on_session_changed(session_id);
        }
    );
    session_manager_->set_session_list_change_callback(
        [this]() {
            on_session_list_changed();
        }
    );

    // 初始化子视图
    chat_view_ = std::make_unique<ChatView>();
    image_view_ = std::make_unique<ImageView>();
    settings_dialog_ = std::make_unique<SettingsDialog>();

    if (!chat_view_->initialize() || !image_view_->initialize() || !settings_dialog_->initialize()) {
        std::cerr << "Failed to initialize sub views" << std::endl;
        return false;
    }

    // 将子视图添加到堆栈
    gtk_stack_add_named(GTK_STACK(content_stack_), chat_view_->get_widget(), "chat");
    gtk_stack_add_named(GTK_STACK(content_stack_), image_view_->get_widget(), "image");

    // 默认显示聊天界面
    switch_to_chat();

    // 初始化系统托盘
#ifdef __APPLE__
    if (macos_tray_ && macos_tray_->initialize()) {
        std::cout << "macOS system tray initialized successfully" << std::endl;
        
        // 使用系统图标而不是emoji（emoji会导致崩溃）
        macos_tray_->setSystemIcon();
        macos_tray_->setTooltip("Duorou - AI Desktop Assistant");
        
        // 设置左键回调为显示窗口
        macos_tray_->setLeftClickCallback([this]() {
            restore_from_tray();
        });
        
        // 设置右键回调为隐藏窗口
        macos_tray_->setRightClickCallback([this]() {
            hide();
        });
        
        // 添加菜单项
        macos_tray_->addMenuItemWithId("show_window", "Show Window", [this]() {
            restore_from_tray();
        });
        
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
        
        // 设置退出回调函数
        macos_tray_->setQuitCallback([this]() {
            quit_application();
        });
        
        macos_tray_->show();
        
        // 初始化菜单状态（窗口当前是显示的）
        macos_tray_->updateWindowStateMenu(true);
    } else {
        std::cerr << "Failed to initialize macOS system tray" << std::endl;
    }
#else
    // 在其他平台上使用GTK系统托盘（如果支持）
    std::cout << "System tray feature not implemented for this platform" << std::endl;
#endif

    // 加载现有会话并更新聊天历史列表
    if (session_manager_) {
        session_manager_->load_sessions_from_file("chat_sessions.txt");
        update_chat_history_list();
    }

    std::cout << "Main window initialized successfully" << std::endl;
    return true;
}

void MainWindow::set_application(core::Application* app) {
    application_ = app;
}

void MainWindow::show() {
    if (window_) {
        gtk_widget_show(window_);
        
        // 更新系统托盘菜单状态
#ifdef __APPLE__
        if (macos_tray_ && macos_tray_->isAvailable()) {
            macos_tray_->updateWindowStateMenu(true);
        }
#endif
    }
}

void MainWindow::hide() {
    std::cout << "[MainWindow] hide() method called" << std::endl;
    if (window_) {
        gtk_widget_hide(window_);
        std::cout << "[MainWindow] Window hidden" << std::endl;
        
        // 更新系统托盘菜单状态
#ifdef __APPLE__
        if (macos_tray_ && macos_tray_->isAvailable()) {
            macos_tray_->updateWindowStateMenu(false);
            std::cout << "[MainWindow] Updated tray menu state to hidden" << std::endl;
        }
#endif
    }
}

void MainWindow::set_title(const std::string& title) {
    if (window_) {
        gtk_window_set_title(GTK_WINDOW(window_), title.c_str());
    }
}

void MainWindow::switch_to_chat() {
    if (content_stack_) {
        gtk_stack_set_visible_child_name(GTK_STACK(content_stack_), "chat");
        current_view_ = "chat";
        update_sidebar_buttons(new_chat_button_);
        
        // 更新状态栏
        if (status_bar_) {
            gtk_statusbar_pop(GTK_STATUSBAR(status_bar_), 1);
            gtk_statusbar_push(GTK_STATUSBAR(status_bar_), 1, "Chat Mode - Ready for conversation");
        }
    }
}

void MainWindow::switch_to_image_generation() {
    if (content_stack_) {
        gtk_stack_set_visible_child_name(GTK_STACK(content_stack_), "image");
        current_view_ = "image";
        update_sidebar_buttons(image_button_);
        
        // 更新状态栏
        if (status_bar_) {
            gtk_statusbar_pop(GTK_STATUSBAR(status_bar_), 1);
            gtk_statusbar_push(GTK_STATUSBAR(status_bar_), 1, "Image Generation Mode - Ready to create");
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
    
    // 先保存会话数据
    if (session_manager_) {
        std::cout << "[MainWindow] Saving session data" << std::endl;
        session_manager_->save_sessions_to_file("chat_sessions.txt");
    }
    
    // 销毁窗口
    if (window_) {
        std::cout << "[MainWindow] Destroying window" << std::endl;
        gtk_window_destroy(GTK_WINDOW(window_));
        window_ = nullptr;
    }
    
    // 调用Application的stop方法来正确退出应用程序
    if (application_) {
        std::cout << "[MainWindow] Calling Application::stop()" << std::endl;
        application_->stop();
    } else {
        std::cout << "[MainWindow] Warning: No application instance available" << std::endl;
    }
    
    std::cout << "[MainWindow] Application should exit now" << std::endl;
}

void MainWindow::create_new_chat() {
    if (session_manager_) {
        session_manager_->create_new_session();
        switch_to_chat();
    }
}

void MainWindow::switch_to_chat_session(const std::string& session_id) {
    if (session_manager_) {
        session_manager_->switch_to_session(session_id);
        switch_to_chat();
    }
}

void MainWindow::update_chat_history_list() {
    if (!chat_history_box_) return;
    
    // 清空现有的聊天历史项
    GtkWidget* child = gtk_widget_get_first_child(chat_history_box_);
    while (child) {
        GtkWidget* next = gtk_widget_get_next_sibling(child);
        gtk_box_remove(GTK_BOX(chat_history_box_), child);
        child = next;
    }
    
    // 添加新的聊天历史项
    if (session_manager_) {
        auto sessions = session_manager_->get_all_sessions();
        for (const auto& session : sessions) {
            GtkWidget* chat_item = gtk_button_new();
            gtk_widget_add_css_class(chat_item, "chat-history-item");
            gtk_widget_set_size_request(chat_item, -1, 40);
            
            std::string title = session->get_title();
            if (title.empty()) {
                title = "New Chat";
            }
            gtk_button_set_label(GTK_BUTTON(chat_item), title.c_str());
            gtk_widget_set_halign(chat_item, GTK_ALIGN_FILL);
            
            // 存储会话ID作为数据
            g_object_set_data_full(G_OBJECT(chat_item), "session_id", 
                                 g_strdup(session->get_id().c_str()), g_free);
            
            // 连接点击信号
            g_signal_connect(chat_item, "clicked", G_CALLBACK(on_chat_history_item_clicked), this);
            
            gtk_box_append(GTK_BOX(chat_history_box_), chat_item);
        }
    }
}

void MainWindow::on_session_changed(const std::string& session_id) {
    // 会话切换时的处理
    // 这里可以更新聊天视图显示当前会话的消息
    std::cout << "Session changed to: " << session_id << std::endl;
}

void MainWindow::on_session_list_changed() {
    // 会话列表变更时更新UI
    update_chat_history_list();
}

void MainWindow::create_header_bar() {
    header_bar_ = gtk_header_bar_new();
    gtk_header_bar_set_show_title_buttons(GTK_HEADER_BAR(header_bar_), TRUE);
    gtk_header_bar_set_title_widget(GTK_HEADER_BAR(header_bar_), gtk_label_new("Duorou - AI Desktop Assistant"));
    
    gtk_window_set_titlebar(GTK_WINDOW(window_), header_bar_);
}

void MainWindow::create_sidebar() {
    // 创建侧边栏容器
    sidebar_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_widget_set_size_request(sidebar_, 280, -1);
    gtk_widget_add_css_class(sidebar_, "sidebar");
    gtk_widget_set_margin_start(sidebar_, 10);
    gtk_widget_set_margin_end(sidebar_, 10);
    gtk_widget_set_margin_top(sidebar_, 10);
    gtk_widget_set_margin_bottom(sidebar_, 10);

    // 创建"New Chat"按钮
    new_chat_button_ = gtk_button_new_with_label("✏️ New Chat");
    gtk_widget_set_size_request(new_chat_button_, -1, 45);
    gtk_widget_add_css_class(new_chat_button_, "new-chat-button");
    gtk_widget_set_margin_bottom(new_chat_button_, 15);
    gtk_box_append(GTK_BOX(sidebar_), new_chat_button_);

    // 创建聊天历史标题
    GtkWidget* history_label = gtk_label_new("Recent Chats");
    gtk_widget_set_halign(history_label, GTK_ALIGN_START);
    gtk_widget_add_css_class(history_label, "section-title");
    gtk_widget_set_margin_bottom(history_label, 10);
    gtk_box_append(GTK_BOX(sidebar_), history_label);

    // 创建聊天历史滚动区域
    GtkWidget* history_scrolled = gtk_scrolled_window_new();
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(history_scrolled), 
                                   GTK_POLICY_NEVER, GTK_POLICY_AUTOMATIC);
    gtk_widget_set_vexpand(history_scrolled, TRUE);
    
    chat_history_box_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(history_scrolled), chat_history_box_);
    gtk_box_append(GTK_BOX(sidebar_), history_scrolled);

    // 初始化时不添加示例项，会话管理器会动态添加

    // 添加分隔符
    GtkWidget* separator = gtk_separator_new(GTK_ORIENTATION_HORIZONTAL);
    gtk_widget_set_margin_top(separator, 15);
    gtk_widget_set_margin_bottom(separator, 15);
    gtk_box_append(GTK_BOX(sidebar_), separator);

    // 创建底部功能按钮
    image_button_ = gtk_button_new_with_label("🎨 Image Generation");
    settings_button_ = gtk_button_new_with_label("⚙️ Settings");

    gtk_widget_set_size_request(image_button_, -1, 40);
    gtk_widget_set_size_request(settings_button_, -1, 40);
    gtk_widget_add_css_class(image_button_, "sidebar-button");
    gtk_widget_add_css_class(settings_button_, "sidebar-button");

    gtk_box_append(GTK_BOX(sidebar_), image_button_);
    gtk_box_append(GTK_BOX(sidebar_), settings_button_);


}

void MainWindow::create_content_area() {
    // 创建水平容器
    GtkWidget* content_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    
    // 添加侧边栏
    gtk_box_append(GTK_BOX(content_box), sidebar_);
    
    // 添加分隔符
    GtkWidget* separator = gtk_separator_new(GTK_ORIENTATION_VERTICAL);
    gtk_box_append(GTK_BOX(content_box), separator);
    
    // 创建内容堆栈
    content_stack_ = gtk_stack_new();
    gtk_stack_set_transition_type(GTK_STACK(content_stack_), GTK_STACK_TRANSITION_TYPE_SLIDE_LEFT_RIGHT);
    gtk_stack_set_transition_duration(GTK_STACK(content_stack_), 300);
    gtk_widget_set_hexpand(content_stack_, TRUE);
    gtk_widget_set_vexpand(content_stack_, TRUE);
    
    gtk_box_append(GTK_BOX(content_box), content_stack_);
    gtk_box_append(GTK_BOX(main_box_), content_box);
}

void MainWindow::create_status_bar() {
    status_bar_ = gtk_statusbar_new();
    gtk_statusbar_push(GTK_STATUSBAR(status_bar_), 1, "Ready");
    gtk_box_append(GTK_BOX(main_box_), status_bar_);
}

void MainWindow::setup_styling() {
    // 加载CSS样式文件
    GtkCssProvider* css_provider = gtk_css_provider_new();
    
    // 尝试加载CSS文件
    const char* css_file_path = "src/gui/styles.css";
    
    GFile* css_file = g_file_new_for_path(css_file_path);
    gtk_css_provider_load_from_file(css_provider, css_file);
    g_object_unref(css_file);
    
    // 应用CSS样式
    gtk_style_context_add_provider_for_display(
        gdk_display_get_default(),
        GTK_STYLE_PROVIDER(css_provider),
        GTK_STYLE_PROVIDER_PRIORITY_APPLICATION
    );
    
    g_object_unref(css_provider);
}

void MainWindow::connect_signals() {
    // 连接窗口信号
    g_signal_connect(window_, "close-request", G_CALLBACK(on_window_delete_event), this);
    g_signal_connect(window_, "destroy", G_CALLBACK(on_window_destroy), this);
    
    // 连接按钮信号
    g_signal_connect(new_chat_button_, "clicked", G_CALLBACK(on_new_chat_button_clicked), this);
    g_signal_connect(image_button_, "clicked", G_CALLBACK(on_image_button_clicked), this);
    g_signal_connect(settings_button_, "clicked", G_CALLBACK(on_settings_button_clicked), this);
}

void MainWindow::update_sidebar_buttons(GtkWidget* active_button) {
    // 重置所有按钮状态
    gtk_widget_remove_css_class(new_chat_button_, "active");
    gtk_widget_remove_css_class(image_button_, "active");
    
    // 设置活动按钮状态
    if (active_button) {
        gtk_widget_add_css_class(active_button, "active");
    }
}

// 静态回调函数实现
void MainWindow::on_new_chat_button_clicked(GtkWidget* widget, gpointer user_data) {
    MainWindow* main_window = static_cast<MainWindow*>(user_data);
    main_window->create_new_chat();
}

void MainWindow::on_chat_history_item_clicked(GtkWidget* widget, gpointer user_data) {
    MainWindow* main_window = static_cast<MainWindow*>(user_data);
    const char* session_id = static_cast<const char*>(g_object_get_data(G_OBJECT(widget), "session_id"));
    if (session_id) {
        main_window->switch_to_chat_session(session_id);
    }
}

void MainWindow::on_image_button_clicked(GtkWidget* widget, gpointer user_data) {
    MainWindow* main_window = static_cast<MainWindow*>(user_data);
    main_window->switch_to_image_generation();
}

void MainWindow::on_settings_button_clicked(GtkWidget* widget, gpointer user_data) {
    MainWindow* main_window = static_cast<MainWindow*>(user_data);
    main_window->show_settings();
}

gboolean MainWindow::on_window_delete_event(GtkWindow* window, gpointer user_data) {
    MainWindow* main_window = static_cast<MainWindow*>(user_data);
    
    // 保存会话数据
    if (main_window->session_manager_) {
        main_window->session_manager_->save_sessions_to_file("chat_sessions.txt");
    }
    
#ifdef __APPLE__
    // 在macOS上，如果系统托盘可用，隐藏窗口而不是退出
    if (main_window->macos_tray_ && main_window->macos_tray_->isAvailable()) {
        main_window->hide();
        return TRUE; // 阻止窗口关闭，只是隐藏
    }
#endif
    
    // 如果系统托盘不可用，正常退出
    return FALSE; // 允许窗口正常关闭
}

void MainWindow::on_window_destroy(GtkWidget* widget, gpointer user_data) {
    // 在GTK4中，通常不需要手动调用退出函数
    // 应用程序会自动处理
}

void MainWindow::restore_from_tray() {
    if (window_) {
        show();
        gtk_window_present(GTK_WINDOW(window_));
        
        // 确保窗口获得焦点
        gtk_window_set_focus_visible(GTK_WINDOW(window_), TRUE);
        
        // 更新系统托盘菜单状态
#ifdef __APPLE__
        if (macos_tray_ && macos_tray_->isAvailable()) {
            macos_tray_->updateWindowStateMenu(true);
        }
#endif
    }
}

void MainWindow::set_tray_status(const std::string& status) {
#ifdef __APPLE__
    if (macos_tray_ && macos_tray_->isAvailable()) {
        if (status == "idle") {
            macos_tray_->setIcon("🌸");  // 花朵表示空闲
            macos_tray_->setTooltip("Duorou - Ready");
        } else if (status == "processing") {
            macos_tray_->setIcon("⚡");  // 闪电表示处理中
            macos_tray_->setTooltip("Duorou - Processing...");
        } else if (status == "error") {
            macos_tray_->setIcon("❌");  // 红叉表示错误
            macos_tray_->setTooltip("Duorou - Error occurred");
        } else if (status == "success") {
            macos_tray_->setIcon("✅");  // 绿勾表示成功
            macos_tray_->setTooltip("Duorou - Task completed");
        } else {
            macos_tray_->setIcon("🌸");  // 默认图标
            macos_tray_->setTooltip("Duorou - AI Desktop Assistant");
        }
    }
#endif
}

} // namespace gui
} // namespace duorou