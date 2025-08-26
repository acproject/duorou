#include "image_view.h"
#include "../core/logger.h"

#include <iostream>

namespace duorou {
namespace gui {

ImageView::ImageView()
    : main_widget_(nullptr)
    , prompt_box_(nullptr)
    , prompt_entry_(nullptr)
    , generate_button_(nullptr)
    , image_scrolled_(nullptr)
    , image_widget_(nullptr)
    , progress_bar_(nullptr)
    , status_label_(nullptr)
{
}

ImageView::~ImageView() {
    // GTK4会自动清理子组件
}

bool ImageView::initialize() {
    // 创建主容器
    main_widget_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    if (!main_widget_) {
        std::cerr << "Failed to create image view main container" << std::endl;
        return false;
    }

    gtk_widget_set_margin_start(main_widget_, 10);
    gtk_widget_set_margin_end(main_widget_, 10);
    gtk_widget_set_margin_top(main_widget_, 10);
    gtk_widget_set_margin_bottom(main_widget_, 10);

    // 创建各个区域
    create_prompt_area();
    create_image_area();
    create_status_area();
    
    // 连接信号
    connect_signals();

    std::cout << "Image view initialized successfully" << std::endl;
    return true;
}

void ImageView::generate_image(const std::string& prompt) {
    if (prompt.empty()) {
        update_status("Please enter a prompt");
        return;
    }

    // 显示进度
    gtk_widget_show(progress_bar_);
    update_progress(0.0);
    update_status("Generating image...");
    
    // 禁用生成按钮
    gtk_widget_set_sensitive(generate_button_, FALSE);
    
    // TODO: 这里应该调用stable-diffusion模型生成图像
    // 暂时创建一个占位符图像
    
    // 模拟进度更新
    update_progress(0.3);
    update_status("Processing prompt...");
    
    update_progress(0.6);
    update_status("Generating pixels...");
    
    update_progress(1.0);
    update_status("Image generated successfully (placeholder)");
    
    // 显示占位符图像
    create_placeholder_image();
    
    // 隐藏进度条并重新启用按钮
    gtk_widget_hide(progress_bar_);
    gtk_widget_set_sensitive(generate_button_, TRUE);
}

void ImageView::display_image(const std::string& image_path) {
    if (!image_widget_ || image_path.empty()) {
        return;
    }

    // 尝试加载图像
    GError* error = nullptr;
    GdkPixbuf* pixbuf = gdk_pixbuf_new_from_file(image_path.c_str(), &error);
    
    if (error) {
        std::cerr << "Failed to load image: " << error->message << std::endl;
        g_error_free(error);
        update_status("Failed to load image: " + image_path);
        return;
    }
    
    if (pixbuf) {
        // 显示图像
        gtk_image_set_from_pixbuf(GTK_IMAGE(image_widget_), pixbuf);
        g_object_unref(pixbuf);
        update_status("Image loaded: " + image_path);
    }
}

void ImageView::clear_image() {
    if (image_widget_) {
        gtk_image_clear(GTK_IMAGE(image_widget_));
        update_status("Image cleared");
    }
}

void ImageView::create_prompt_area() {
    // 创建提示词输入区域容器
    prompt_box_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_widget_set_margin_bottom(prompt_box_, 10);
    
    // 创建标签
    GtkWidget* prompt_label = gtk_label_new("Image Generation Prompt:");
    gtk_widget_set_halign(prompt_label, GTK_ALIGN_START);
    gtk_widget_add_css_class(prompt_label, "prompt-label");
    
    // 创建输入框
    prompt_entry_ = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(prompt_entry_), "Describe the image you want to generate...");
    gtk_widget_set_hexpand(prompt_entry_, TRUE);
    
    // 创建按钮容器
    GtkWidget* button_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    gtk_widget_set_halign(button_box, GTK_ALIGN_CENTER);
    gtk_widget_set_margin_top(button_box, 5);
    
    // 创建生成按钮
    generate_button_ = gtk_button_new_with_label("🎨 Generate Image");
    gtk_widget_set_size_request(generate_button_, 150, 40);
    gtk_widget_add_css_class(generate_button_, "generate-button");
    
    // 创建清空按钮
    GtkWidget* clear_button = gtk_button_new_with_label("🗑️ Clear");
    gtk_widget_set_size_request(clear_button, 100, 40);
    
    // 组装组件
    gtk_box_append(GTK_BOX(button_box), generate_button_);
    gtk_box_append(GTK_BOX(button_box), clear_button);
    
    gtk_box_append(GTK_BOX(prompt_box_), prompt_label);
    gtk_box_append(GTK_BOX(prompt_box_), prompt_entry_);
    gtk_box_append(GTK_BOX(prompt_box_), button_box);
    
    // 连接清空按钮信号
    g_signal_connect(clear_button, "clicked", G_CALLBACK(on_clear_button_clicked), this);
    
    // 添加到主容器
    gtk_box_append(GTK_BOX(main_widget_), prompt_box_);
}

void ImageView::create_image_area() {
    // 创建图像显示组件
    image_widget_ = gtk_image_new();
    gtk_widget_set_size_request(image_widget_, 512, 512);
    gtk_widget_set_halign(image_widget_, GTK_ALIGN_CENTER);
    gtk_widget_set_valign(image_widget_, GTK_ALIGN_CENTER);
    
    // 创建滚动窗口
    image_scrolled_ = gtk_scrolled_window_new();
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(image_scrolled_), 
                                   GTK_POLICY_AUTOMATIC, GTK_POLICY_AUTOMATIC);
    gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(image_scrolled_), image_widget_);
    
    // 设置滚动窗口大小
    gtk_widget_set_vexpand(image_scrolled_, TRUE);
    gtk_widget_set_hexpand(image_scrolled_, TRUE);
    
    // 添加边框样式
    gtk_widget_add_css_class(image_scrolled_, "image-display");
    
    // 添加到主容器
    gtk_box_append(GTK_BOX(main_widget_), image_scrolled_);
}

void ImageView::create_status_area() {
    // 创建状态区域容器
    GtkWidget* status_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_widget_set_margin_top(status_box, 10);
    
    // 创建进度条
    progress_bar_ = gtk_progress_bar_new();
    gtk_widget_hide(progress_bar_); // 初始时隐藏
    gtk_widget_set_margin_bottom(progress_bar_, 5);
    
    // 创建状态标签
    status_label_ = gtk_label_new("Ready to generate images");
    gtk_widget_set_halign(status_label_, GTK_ALIGN_START);
    gtk_widget_add_css_class(status_label_, "status-label");
    
    // 组装组件
    gtk_box_append(GTK_BOX(status_box), progress_bar_);
    gtk_box_append(GTK_BOX(status_box), status_label_);
    
    // 添加到主容器
    gtk_box_append(GTK_BOX(main_widget_), status_box);
}

void ImageView::connect_signals() {
    // 连接生成按钮信号
    g_signal_connect(generate_button_, "clicked", G_CALLBACK(on_generate_button_clicked), this);
    
    // 连接回车键生成图像
    g_signal_connect(prompt_entry_, "activate", G_CALLBACK(on_prompt_entry_activate), this);
}

void ImageView::update_progress(double progress) {
    if (progress_bar_) {
        gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(progress_bar_), progress);
    }
}

void ImageView::update_status(const std::string& status) {
    if (status_label_) {
        gtk_label_set_text(GTK_LABEL(status_label_), status.c_str());
    }
}

void ImageView::create_placeholder_image() {
    // 创建一个简单的占位符图像
    GdkPixbuf* pixbuf = gdk_pixbuf_new(GDK_COLORSPACE_RGB, FALSE, 8, 512, 512);
    if (pixbuf) {
        // 填充为浅灰色
        gdk_pixbuf_fill(pixbuf, 0xC0C0C0FF);
        
        // 显示图像
        gtk_image_set_from_pixbuf(GTK_IMAGE(image_widget_), pixbuf);
        g_object_unref(pixbuf);
    }
}

// 静态回调函数实现
void ImageView::on_generate_button_clicked(GtkWidget* widget, gpointer user_data) {
    ImageView* image_view = static_cast<ImageView*>(user_data);
    
    // 获取提示词
    const char* prompt_text = gtk_entry_buffer_get_text(gtk_entry_get_buffer(GTK_ENTRY(image_view->prompt_entry_)));
    std::string prompt = prompt_text ? prompt_text : "";
    
    image_view->generate_image(prompt);
}

void ImageView::on_prompt_entry_activate(GtkWidget* widget, gpointer user_data) {
    ImageView* image_view = static_cast<ImageView*>(user_data);
    
    // 获取提示词
    const char* prompt_text = gtk_entry_buffer_get_text(gtk_entry_get_buffer(GTK_ENTRY(widget)));
    std::string prompt = prompt_text ? prompt_text : "";
    
    image_view->generate_image(prompt);
}

void ImageView::on_clear_button_clicked(GtkWidget* widget, gpointer user_data) {
    ImageView* image_view = static_cast<ImageView*>(user_data);
    image_view->clear_image();
    
    // 清空输入框
    gtk_entry_buffer_set_text(gtk_entry_get_buffer(GTK_ENTRY(image_view->prompt_entry_)), "", 0);
}

} // namespace gui
} // namespace duorou