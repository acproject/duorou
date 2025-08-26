#include "video_display_window.h"
#include <iostream>
#include <cstring>

namespace duorou {
namespace gui {

VideoDisplayWindow::VideoDisplayWindow() 
    : window_(nullptr), video_area_(nullptr), info_label_(nullptr),
      frame_data_(nullptr), frame_width_(0), frame_height_(0), frame_channels_(4) {
    init_ui();
}

VideoDisplayWindow::~VideoDisplayWindow() {
    if (window_) {
        gtk_window_destroy(GTK_WINDOW(window_));
    }
}

void VideoDisplayWindow::init_ui() {
    // 创建窗口
    window_ = gtk_window_new();
    gtk_window_set_title(GTK_WINDOW(window_), "视频预览");
    gtk_window_set_default_size(GTK_WINDOW(window_), 640, 480);
    gtk_window_set_resizable(GTK_WINDOW(window_), TRUE);
    
    // 在GTK4中设置窗口层级，确保视频窗口在普通窗口之上但在模态对话框之下
    gtk_window_set_modal(GTK_WINDOW(window_), FALSE);
    gtk_window_set_transient_for(GTK_WINDOW(window_), nullptr);
    
    // 设置窗口类型提示，使其表现为工具窗口
    gtk_window_set_decorated(GTK_WINDOW(window_), TRUE);
    gtk_window_set_deletable(GTK_WINDOW(window_), TRUE);
    
    // 创建主容器
    GtkWidget* vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_widget_set_margin_start(vbox, 10);
    gtk_widget_set_margin_end(vbox, 10);
    gtk_widget_set_margin_top(vbox, 10);
    gtk_widget_set_margin_bottom(vbox, 10);
    
    // 创建信息标签
    info_label_ = gtk_label_new("等待视频数据...");
    gtk_widget_set_halign(info_label_, GTK_ALIGN_CENTER);
    
    // 创建视频显示区域
    video_area_ = gtk_drawing_area_new();
    gtk_widget_set_size_request(video_area_, 320, 240);
    gtk_widget_set_hexpand(video_area_, TRUE);
    gtk_widget_set_vexpand(video_area_, TRUE);
    
    // 设置绘制回调
    gtk_drawing_area_set_draw_func(GTK_DRAWING_AREA(video_area_), on_draw_area, this, nullptr);
    
    // 添加到容器
    gtk_box_append(GTK_BOX(vbox), info_label_);
    gtk_box_append(GTK_BOX(vbox), video_area_);
    
    // 设置窗口内容
    gtk_window_set_child(GTK_WINDOW(window_), vbox);
    
    // 连接窗口关闭信号
    g_signal_connect(window_, "close-request", G_CALLBACK(on_window_close), this);
}

void VideoDisplayWindow::show() {
    if (window_) {
        gtk_widget_set_visible(window_, TRUE);
        // 显示窗口但不强制获得焦点，避免干扰对话框
        gtk_window_present(GTK_WINDOW(window_));
    }
}

void VideoDisplayWindow::hide() {
    if (window_) {
        gtk_widget_set_visible(window_, FALSE);
    }
}

void VideoDisplayWindow::set_close_callback(std::function<void()> callback) {
    close_callback_ = callback;
}

bool VideoDisplayWindow::is_visible() const {
    return window_ && gtk_widget_get_visible(window_);
}

void VideoDisplayWindow::update_frame(const media::VideoFrame& frame) {
    // 更新帧数据
    frame_width_ = frame.width;
    frame_height_ = frame.height;
    frame_channels_ = frame.channels;
    
    // 分配内存存储帧数据
    size_t data_size = frame.width * frame.height * frame.channels;
    frame_data_ = std::make_unique<guchar[]>(data_size);
    std::memcpy(frame_data_.get(), frame.data.data(), data_size);
    
    // 更新信息标签 - 格式化时间戳为可读格式
    char info_text[256];
    
    // 将时间戳转换为可读的时间格式
    auto timestamp_ms = static_cast<int64_t>(frame.timestamp * 1000);
    auto timestamp_sec = timestamp_ms / 1000;
    auto ms_part = timestamp_ms % 1000;
    auto hours = timestamp_sec / 3600;
    auto minutes = (timestamp_sec % 3600) / 60;
    auto seconds = timestamp_sec % 60;
    
    char timestamp_str[64];
    snprintf(timestamp_str, sizeof(timestamp_str), "%02lld:%02lld:%02lld.%03lld", 
             hours, minutes, seconds, ms_part);
    
    snprintf(info_text, sizeof(info_text), "分辨率: %dx%d, 通道: %d, 时间戳: %s", 
             frame.width, frame.height, frame.channels, timestamp_str);
    gtk_label_set_text(GTK_LABEL(info_label_), info_text);
    
    // 触发重绘
    if (video_area_) {
        gtk_widget_queue_draw(video_area_);
    }
}

void VideoDisplayWindow::on_draw_area(GtkDrawingArea* area, cairo_t* cr, int width, int height, gpointer user_data) {
    VideoDisplayWindow* window = static_cast<VideoDisplayWindow*>(user_data);
    
    // 设置背景色
    cairo_set_source_rgb(cr, 0.1, 0.1, 0.1);
    cairo_paint(cr);
    
    // 如果有视频数据，绘制视频帧
    if (window->frame_data_ && window->frame_width_ > 0 && window->frame_height_ > 0) {
        // 创建图像表面
        cairo_format_t format = CAIRO_FORMAT_RGB24;
        if (window->frame_width_ > 0 && window->frame_height_ > 0) {
            // 计算缩放比例以适应显示区域
            double scale_x = (double)width / window->frame_width_;
            double scale_y = (double)height / window->frame_height_;
            double scale = std::min(scale_x, scale_y);
            
            // 计算居中位置
            int scaled_width = (int)(window->frame_width_ * scale);
            int scaled_height = (int)(window->frame_height_ * scale);
            int x_offset = (width - scaled_width) / 2;
            int y_offset = (height - scaled_height) / 2;
            
            // 保存当前状态
            cairo_save(cr);
            
            // 移动到居中位置并缩放
            cairo_translate(cr, x_offset, y_offset);
            cairo_scale(cr, scale, scale);
            
            // 创建图像表面来显示实际视频数据
            cairo_surface_t* surface = nullptr;
            
            // 根据通道数创建适当的图像表面
            if (window->frame_data_) {
                // 创建一个临时的RGBA数据缓冲区
                int stride = cairo_format_stride_for_width(CAIRO_FORMAT_RGB24, window->frame_width_);
                std::unique_ptr<guchar[]> rgba_data = std::make_unique<guchar[]>(stride * window->frame_height_);
                
                // 将视频帧数据转换为Cairo可用的格式
                // 使用存储的通道数信息，避免动态检测导致的不稳定
                int channels = window->frame_channels_;
                
                for (int y = 0; y < window->frame_height_; y++) {
                    for (int x = 0; x < window->frame_width_; x++) {
                        int src_idx = (y * window->frame_width_ + x) * channels;
                        int dst_idx = y * stride + x * 4;
                        
                        if (src_idx + (channels - 1) < (int)(window->frame_width_ * window->frame_height_ * channels)) {
                            if (channels == 4) {
                                // RGBA格式
                                rgba_data[dst_idx + 0] = window->frame_data_[src_idx + 2]; // B
                                rgba_data[dst_idx + 1] = window->frame_data_[src_idx + 1]; // G
                                rgba_data[dst_idx + 2] = window->frame_data_[src_idx + 0]; // R
                                rgba_data[dst_idx + 3] = window->frame_data_[src_idx + 3]; // A
                            } else {
                                // RGB格式
                                rgba_data[dst_idx + 0] = window->frame_data_[src_idx + 2]; // B
                                rgba_data[dst_idx + 1] = window->frame_data_[src_idx + 1]; // G
                                rgba_data[dst_idx + 2] = window->frame_data_[src_idx + 0]; // R
                                rgba_data[dst_idx + 3] = 255; // A
                            }
                        }
                    }
                }
                
                // 创建Cairo图像表面
                surface = cairo_image_surface_create_for_data(
                    rgba_data.get(), CAIRO_FORMAT_RGB24, 
                    window->frame_width_, window->frame_height_, stride);
                
                if (surface && cairo_surface_status(surface) == CAIRO_STATUS_SUCCESS) {
                    // 绘制图像
                    cairo_set_source_surface(cr, surface, 0, 0);
                    cairo_paint(cr);
                } else {
                    // 如果创建表面失败，显示错误信息
                    cairo_set_source_rgb(cr, 0.8, 0.2, 0.2);
                    cairo_rectangle(cr, 0, 0, window->frame_width_, window->frame_height_);
                    cairo_fill(cr);
                }
                
                // 清理表面
                if (surface) {
                    cairo_surface_destroy(surface);
                }
            } else {
                // 没有数据时显示灰色背景
                cairo_set_source_rgb(cr, 0.5, 0.5, 0.5);
                cairo_rectangle(cr, 0, 0, window->frame_width_, window->frame_height_);
                cairo_fill(cr);
            }
            
            // 恢复状态
            cairo_restore(cr);
        }
    } else {
        // 没有视频数据时显示提示文字
        cairo_set_source_rgb(cr, 0.7, 0.7, 0.7);
        cairo_select_font_face(cr, "Arial", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);
        cairo_set_font_size(cr, 16);
        
        const char* text = "等待视频数据...";
        cairo_text_extents_t extents;
        cairo_text_extents(cr, text, &extents);
        
        cairo_move_to(cr, (width - extents.width) / 2, (height + extents.height) / 2);
        cairo_show_text(cr, text);
    }
}

gboolean VideoDisplayWindow::on_window_close(GtkWidget* widget, gpointer user_data) {
    VideoDisplayWindow* window = static_cast<VideoDisplayWindow*>(user_data);
    
    // 如果设置了关闭回调，调用它来停止录制
    if (window->close_callback_) {
        window->close_callback_();
    }
    
    window->hide();
    return TRUE; // 阻止窗口销毁，只是隐藏
}

} // namespace gui
} // namespace duorou