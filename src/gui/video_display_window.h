#ifndef DUOROU_GUI_VIDEO_DISPLAY_WINDOW_H
#define DUOROU_GUI_VIDEO_DISPLAY_WINDOW_H

#include <gtk/gtk.h>
#include <memory>
#include "../media/video_capture.h"

namespace duorou {
namespace gui {

class VideoDisplayWindow {
public:
    VideoDisplayWindow();
    ~VideoDisplayWindow();
    
    // 显示窗口
    void show();
    
    // 隐藏窗口
    void hide();
    
    // 更新视频帧
    void update_frame(const media::VideoFrame& frame);
    
    // 检查窗口是否可见
    bool is_visible() const;
    
private:
    GtkWidget* window_;
    GtkWidget* video_area_;
    GtkWidget* info_label_;
    
    // 视频帧数据
    std::unique_ptr<guchar[]> frame_data_;
    int frame_width_;
    int frame_height_;
    
    // 初始化UI
    void init_ui();
    
    // 绘制回调
    static void on_draw_area(GtkDrawingArea* area, cairo_t* cr, int width, int height, gpointer user_data);
    
    // 窗口关闭回调
    static gboolean on_window_close(GtkWidget* widget, gpointer user_data);
};

} // namespace gui
} // namespace duorou

#endif // DUOROU_GUI_VIDEO_DISPLAY_WINDOW_H