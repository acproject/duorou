#include "video_display_window.h"
#include <iostream>
#include <cstring>

namespace duorou {
namespace gui {

VideoDisplayWindow::VideoDisplayWindow() 
    : window_(nullptr), video_area_(nullptr), info_label_(nullptr),
      frame_data_(nullptr), frame_width_(0), frame_height_(0), frame_channels_(4),
      cached_surface_(nullptr), cached_rgba_data_(nullptr), cached_width_(0), cached_height_(0) {
    init_ui();
}

VideoDisplayWindow::~VideoDisplayWindow() {
    // 清理缓存的Cairo表面
    if (cached_surface_) {
        cairo_surface_destroy(cached_surface_);
        cached_surface_ = nullptr;
    }
    
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
    // 检查是否需要重新创建缓存表面
    bool need_recreate_surface = (frame.width != cached_width_ || frame.height != cached_height_);
    
    // 更新帧数据
    frame_width_ = frame.width;
    frame_height_ = frame.height;
    frame_channels_ = frame.channels;
    
    // 分配内存存储帧数据
    size_t data_size = frame.width * frame.height * frame.channels;
    frame_data_ = std::make_unique<guchar[]>(data_size);
    std::memcpy(frame_data_.get(), frame.data.data(), data_size);
    
    // 如果尺寸改变，重新创建缓存表面
    if (need_recreate_surface) {
        // 清理旧的表面
        if (cached_surface_) {
            cairo_surface_destroy(cached_surface_);
            cached_surface_ = nullptr;
        }
        
        // 更新缓存尺寸
        cached_width_ = frame.width;
        cached_height_ = frame.height;
        
        // 创建新的RGBA数据缓冲区
        int stride = cairo_format_stride_for_width(CAIRO_FORMAT_RGB24, frame.width);
        cached_rgba_data_ = std::make_unique<guchar[]>(stride * frame.height);
        
        // 创建新的Cairo表面
        cached_surface_ = cairo_image_surface_create_for_data(
            cached_rgba_data_.get(), CAIRO_FORMAT_RGB24, 
            frame.width, frame.height, stride);
    }
    
    // 更新缓存表面的数据
    if (cached_surface_ && cached_rgba_data_) {
        int stride = cairo_format_stride_for_width(CAIRO_FORMAT_RGB24, frame.width);
        int channels = frame.channels;
        
        // 在修改表面数据前，先获取表面数据指针
        cairo_surface_flush(cached_surface_);
        
        for (int y = 0; y < frame.height; y++) {
            for (int x = 0; x < frame.width; x++) {
                int src_idx = (y * frame.width + x) * channels;
                int dst_idx = y * stride + x * 4;
                
                if (src_idx + (channels - 1) < (int)(frame.width * frame.height * channels)) {
                    if (channels == 4) {
                        // ScreenCaptureKit使用BGRA格式，转换为Cairo的RGB24格式
                        cached_rgba_data_[dst_idx + 0] = frame.data[src_idx + 0]; // B
                        cached_rgba_data_[dst_idx + 1] = frame.data[src_idx + 1]; // G
                        cached_rgba_data_[dst_idx + 2] = frame.data[src_idx + 2]; // R
                        cached_rgba_data_[dst_idx + 3] = frame.data[src_idx + 3]; // A
                    } else {
                        // RGB格式转换为Cairo RGB24
                        cached_rgba_data_[dst_idx + 0] = frame.data[src_idx + 2]; // B
                        cached_rgba_data_[dst_idx + 1] = frame.data[src_idx + 1]; // G
                        cached_rgba_data_[dst_idx + 2] = frame.data[src_idx + 0]; // R
                        cached_rgba_data_[dst_idx + 3] = 255; // A
                    }
                }
            }
        }
        
        // 标记表面数据已更新，确保Cairo知道数据已被修改
        cairo_surface_mark_dirty(cached_surface_);
    }
    
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
    
    // 如果有缓存的表面，直接绘制
    if (window->cached_surface_ && window->frame_width_ > 0 && window->frame_height_ > 0) {
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
        
        // 直接绘制缓存的表面
        cairo_set_source_surface(cr, window->cached_surface_, 0, 0);
        cairo_paint(cr);
        
        // 恢复状态
        cairo_restore(cr);
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
    // 安全检查：确保user_data不为空
    if (!user_data) {
        std::cout << "VideoDisplayWindow::on_window_close: user_data为空" << std::endl;
        return TRUE;
    }
    
    VideoDisplayWindow* window = static_cast<VideoDisplayWindow*>(user_data);
    
    // 安全检查：确保window对象有效
    if (!window) {
        std::cout << "VideoDisplayWindow::on_window_close: window对象为空" << std::endl;
        return TRUE;
    }
    
    std::cout << "VideoDisplayWindow关闭事件触发" << std::endl;
    
    // 如果设置了关闭回调，调用它来停止录制
    if (window->close_callback_) {
        try {
            std::cout << "调用视频窗口关闭回调..." << std::endl;
            window->close_callback_();
            std::cout << "视频窗口关闭回调执行完成" << std::endl;
        } catch (const std::exception &e) {
            std::cout << "视频窗口关闭回调异常: " << e.what() << std::endl;
        } catch (...) {
            std::cout << "视频窗口关闭回调发生未知异常" << std::endl;
        }
    }
    
    // 安全地隐藏窗口
    try {
        window->hide();
    } catch (const std::exception &e) {
        std::cout << "隐藏视频窗口异常: " << e.what() << std::endl;
    }
    
    return TRUE; // 阻止窗口销毁，只是隐藏
}

} // namespace gui
} // namespace duorou