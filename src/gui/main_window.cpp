#include "main_window.h"
#include "chat_view.h"
#include "image_view.h"
#include "settings_dialog.h"
#include "../core/logger.h"

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
    , chat_button_(nullptr)
    , image_button_(nullptr)
    , settings_button_(nullptr)
    , current_view_("chat")
{
}

MainWindow::~MainWindow() {
    if (window_) {
        gtk_window_destroy(GTK_WINDOW(window_));
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

    std::cout << "Main window initialized successfully" << std::endl;
    return true;
}

void MainWindow::show() {
    if (window_) {
        gtk_widget_show(window_);
    }
}

void MainWindow::hide() {
    if (window_) {
        gtk_widget_hide(window_);
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
        update_sidebar_buttons(chat_button_);
        
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
    // 简单的退出方式，适用于没有GApplication的情况
    if (window_) {
        gtk_window_close(GTK_WINDOW(window_));
    }
}

void MainWindow::create_header_bar() {
    header_bar_ = gtk_header_bar_new();
    gtk_header_bar_set_show_title_buttons(GTK_HEADER_BAR(header_bar_), TRUE);
    gtk_header_bar_set_title_widget(GTK_HEADER_BAR(header_bar_), gtk_label_new("Duorou - AI Desktop Assistant"));
    
    gtk_window_set_titlebar(GTK_WINDOW(window_), header_bar_);
}

void MainWindow::create_sidebar() {
    // 创建侧边栏容器
    sidebar_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_widget_set_size_request(sidebar_, 200, -1);
    gtk_widget_add_css_class(sidebar_, "sidebar");

    // 创建按钮
    chat_button_ = gtk_button_new_with_label("💬 Chat");
    image_button_ = gtk_button_new_with_label("🎨 Image");
    settings_button_ = gtk_button_new_with_label("⚙️ Settings");

    // 设置按钮样式
    gtk_widget_set_size_request(chat_button_, -1, 50);
    gtk_widget_set_size_request(image_button_, -1, 50);
    gtk_widget_set_size_request(settings_button_, -1, 50);

    // 添加按钮到侧边栏
    gtk_box_append(GTK_BOX(sidebar_), chat_button_);
    gtk_box_append(GTK_BOX(sidebar_), image_button_);
    gtk_box_append(GTK_BOX(sidebar_), gtk_separator_new(GTK_ORIENTATION_HORIZONTAL));
    gtk_box_append(GTK_BOX(sidebar_), settings_button_);

    // 添加弹性空间
    GtkWidget* spacer = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_widget_set_vexpand(spacer, TRUE);
    gtk_box_append(GTK_BOX(sidebar_), spacer);
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
    // 添加CSS样式
    GtkCssProvider* css_provider = gtk_css_provider_new();
    const char* css_data = 
        ".sidebar { "
        "  background-color: #f5f5f5; "
        "  border-right: 1px solid #ddd; "
        "  padding: 10px; "
        "} "
        ".sidebar button { "
        "  margin: 2px 0; "
        "  border-radius: 8px; "
        "} "
        ".sidebar button:checked { "
        "  background-color: #007acc; "
        "  color: white; "
        "}";
    
    GError* error = nullptr;
    gtk_css_provider_load_from_string(css_provider, css_data);
    // GTK4中gtk_css_provider_load_from_string不返回错误，如果有问题会在运行时显示警告
    
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
    g_signal_connect(chat_button_, "clicked", G_CALLBACK(on_chat_button_clicked), this);
    g_signal_connect(image_button_, "clicked", G_CALLBACK(on_image_button_clicked), this);
    g_signal_connect(settings_button_, "clicked", G_CALLBACK(on_settings_button_clicked), this);
}

void MainWindow::update_sidebar_buttons(GtkWidget* active_button) {
    // 重置所有按钮状态
    gtk_widget_remove_css_class(chat_button_, "active");
    gtk_widget_remove_css_class(image_button_, "active");
    
    // 设置活动按钮状态
    if (active_button) {
        gtk_widget_add_css_class(active_button, "active");
    }
}

// 静态回调函数实现
void MainWindow::on_chat_button_clicked(GtkWidget* widget, gpointer user_data) {
    MainWindow* main_window = static_cast<MainWindow*>(user_data);
    main_window->switch_to_chat();
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
    main_window->quit_application();
    return FALSE;
}

void MainWindow::on_window_destroy(GtkWidget* widget, gpointer user_data) {
    // 在GTK4中，通常不需要手动调用退出函数
    // 应用程序会自动处理
}

} // namespace gui
} // namespace duorou