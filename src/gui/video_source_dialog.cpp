#include "video_source_dialog.h"
#include "../core/logger.h"
#include <iostream>

namespace duorou {
namespace gui {

VideoSourceDialog::VideoSourceDialog()
    : dialog_(nullptr)
    , content_box_(nullptr)
    , title_label_(nullptr)
    , desktop_button_(nullptr)
    , camera_button_(nullptr)
    , cancel_button_(nullptr)
    , button_box_(nullptr)
{
}

VideoSourceDialog::~VideoSourceDialog() {
    if (dialog_) {
        gtk_window_destroy(GTK_WINDOW(dialog_));
        dialog_ = nullptr;
    }
}

bool VideoSourceDialog::initialize() {
    // 创建对话框
    dialog_ = gtk_window_new();
    if (!dialog_) {
        std::cerr << "Failed to create video source dialog" << std::endl;
        return false;
    }

    // 设置对话框属性
    gtk_window_set_title(GTK_WINDOW(dialog_), "选择视频源");
    gtk_window_set_default_size(GTK_WINDOW(dialog_), 400, 200);
    gtk_window_set_resizable(GTK_WINDOW(dialog_), FALSE);
    gtk_window_set_modal(GTK_WINDOW(dialog_), TRUE);

    // 创建内容
    create_content();

    // 设置样式
    setup_styling();

    // 连接信号
    connect_signals();

    std::cout << "Video source dialog initialized successfully" << std::endl;
    return true;
}

void VideoSourceDialog::show(GtkWidget* parent_window, std::function<void(VideoSource)> callback) {
    if (!dialog_) {
        std::cerr << "Dialog not initialized" << std::endl;
        return;
    }

    // 保存回调函数
    selection_callback_ = callback;

    // 设置父窗口
    if (parent_window) {
        GtkWindow* parent = GTK_WINDOW(gtk_widget_get_root(parent_window));
        if (parent) {
            gtk_window_set_transient_for(GTK_WINDOW(dialog_), parent);
        }
    }

    // 显示对话框
    gtk_widget_show(dialog_);
    gtk_window_present(GTK_WINDOW(dialog_));
}

void VideoSourceDialog::hide() {
    if (dialog_) {
        gtk_widget_hide(dialog_);
    }
}

void VideoSourceDialog::create_content() {
    // 创建主容器
    content_box_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 20);
    gtk_widget_set_margin_top(content_box_, 20);
    gtk_widget_set_margin_bottom(content_box_, 20);
    gtk_widget_set_margin_start(content_box_, 20);
    gtk_widget_set_margin_end(content_box_, 20);
    gtk_window_set_child(GTK_WINDOW(dialog_), content_box_);

    // 创建标题标签
    title_label_ = gtk_label_new("请选择视频源：");
    gtk_widget_add_css_class(title_label_, "title");
    gtk_box_append(GTK_BOX(content_box_), title_label_);

    // 创建按钮容器
    button_box_ = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    gtk_widget_set_halign(button_box_, GTK_ALIGN_CENTER);
    gtk_box_append(GTK_BOX(content_box_), button_box_);

    // 创建桌面录制按钮
    desktop_button_ = gtk_button_new_with_label("Desktop Recording");
    gtk_widget_set_size_request(desktop_button_, 120, 50);
    gtk_widget_add_css_class(desktop_button_, "suggested-action");
    gtk_box_append(GTK_BOX(button_box_), desktop_button_);

    // 创建摄像头按钮
    camera_button_ = gtk_button_new_with_label("Camera");
    gtk_widget_set_size_request(camera_button_, 120, 50);
    gtk_widget_add_css_class(camera_button_, "suggested-action");
    gtk_box_append(GTK_BOX(button_box_), camera_button_);

    // 创建取消按钮容器
    GtkWidget* cancel_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    gtk_widget_set_halign(cancel_box, GTK_ALIGN_CENTER);
    gtk_box_append(GTK_BOX(content_box_), cancel_box);

    // 创建取消按钮
    cancel_button_ = gtk_button_new_with_label("取消");
    gtk_widget_set_size_request(cancel_button_, 80, 35);
    gtk_box_append(GTK_BOX(cancel_box), cancel_button_);
}

void VideoSourceDialog::setup_styling() {
    // 添加CSS样式
    const char* css_data = 
        ".title { "
        "  font-size: 16px; "
        "  font-weight: bold; "
        "  margin-bottom: 10px; "
        "} "
        "button { "
        "  font-size: 14px; "
        "  padding: 8px 16px; "
        "  border-radius: 6px; "
        "} "
        "button.suggested-action { "
        "  background: linear-gradient(to bottom, #4a90e2, #357abd); "
        "  color: white; "
        "  border: 1px solid #2968a3; "
        "} "
        "button.suggested-action:hover { "
        "  background: linear-gradient(to bottom, #5ba0f2, #4a90e2); "
        "} ";

    GtkCssProvider* css_provider = gtk_css_provider_new();
    gtk_css_provider_load_from_data(css_provider, css_data, -1);
    
    GtkStyleContext* style_context = gtk_widget_get_style_context(dialog_);
    gtk_style_context_add_provider(style_context, 
                                   GTK_STYLE_PROVIDER(css_provider), 
                                   GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
    
    g_object_unref(css_provider);
}

void VideoSourceDialog::connect_signals() {
    // 连接按钮点击信号
    g_signal_connect(desktop_button_, "clicked", 
                     G_CALLBACK(on_desktop_button_clicked), this);
    g_signal_connect(camera_button_, "clicked", 
                     G_CALLBACK(on_camera_button_clicked), this);
    g_signal_connect(cancel_button_, "clicked", 
                     G_CALLBACK(on_cancel_button_clicked), this);
    
    // 连接窗口关闭信号
    g_signal_connect(dialog_, "close-request", 
                     G_CALLBACK(on_dialog_delete_event), this);
}

void VideoSourceDialog::handle_selection(VideoSource source) {
    // 隐藏对话框
    hide();
    
    // 调用回调函数
    if (selection_callback_) {
        selection_callback_(source);
    }
}

// 静态回调函数实现
void VideoSourceDialog::on_desktop_button_clicked(GtkWidget* widget, gpointer user_data) {
    VideoSourceDialog* dialog = static_cast<VideoSourceDialog*>(user_data);
    dialog->handle_selection(VideoSource::DESKTOP_CAPTURE);
}

void VideoSourceDialog::on_camera_button_clicked(GtkWidget* widget, gpointer user_data) {
    VideoSourceDialog* dialog = static_cast<VideoSourceDialog*>(user_data);
    dialog->handle_selection(VideoSource::CAMERA);
}

void VideoSourceDialog::on_cancel_button_clicked(GtkWidget* widget, gpointer user_data) {
    VideoSourceDialog* dialog = static_cast<VideoSourceDialog*>(user_data);
    dialog->handle_selection(VideoSource::CANCEL);
}

gboolean VideoSourceDialog::on_dialog_delete_event(GtkWindow* window, gpointer user_data) {
    VideoSourceDialog* dialog = static_cast<VideoSourceDialog*>(user_data);
    dialog->handle_selection(VideoSource::CANCEL);
    return TRUE; // 阻止默认的关闭行为，由handle_selection处理
}

} // namespace gui
} // namespace duorou