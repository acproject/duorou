#ifndef DUOROU_GUI_ENHANCED_VIDEO_CAPTURE_WINDOW_H
#define DUOROU_GUI_ENHANCED_VIDEO_CAPTURE_WINDOW_H

#include <gtk/gtk.h>
#include <memory>
#include <functional>
#include "../media/video_capture.h"
#include "../media/video_frame.h"

#ifdef __cplusplus
#include <vector>
#include <string>
#endif

namespace duorou {
namespace gui {

/**
 * 增强的视频捕捉窗口
 * 实现左侧视频显示区域和右侧窗口/设备选择列表的布局
 */
class EnhancedVideoCaptureWindow {
public:
    enum class CaptureMode {
        DESKTOP,    // 桌面捕获模式
        CAMERA      // 摄像头模式
    };

    struct WindowInfo {
        std::string title;      // 窗口标题
        std::string app_name;   // 应用程序名称
        int window_id;          // 窗口ID
        bool is_desktop;        // 是否为桌面
    };

    struct DeviceInfo {
        std::string name;       // 设备名称
        std::string id;         // 设备ID
        int device_index;       // 设备索引
    };

    EnhancedVideoCaptureWindow();
    ~EnhancedVideoCaptureWindow();

    // 禁用拷贝构造和赋值
    EnhancedVideoCaptureWindow(const EnhancedVideoCaptureWindow&) = delete;
    EnhancedVideoCaptureWindow& operator=(const EnhancedVideoCaptureWindow&) = delete;

    /**
     * 初始化窗口
     * @return 成功返回true，失败返回false
     */
    bool initialize();

    /**
     * 显示窗口
     * @param mode 捕获模式
     */
    void show(CaptureMode mode);

    /**
     * 隐藏窗口
     */
    void hide();

    /**
     * 检查窗口是否可见
     */
    bool is_visible() const;

    /**
     * 更新视频帧
     * @param frame 视频帧数据
     */
    void update_frame(const media::VideoFrame& frame);

    /**
     * 设置窗口关闭回调
     * @param callback 关闭回调函数
     */
    void set_close_callback(std::function<void()> callback);

    /**
     * 设置窗口选择回调
     * @param callback 窗口选择回调函数
     */
    void set_window_selection_callback(std::function<void(const WindowInfo&)> callback);

    /**
     * 设置设备选择回调
     * @param callback 设备选择回调函数
     */
    void set_device_selection_callback(std::function<void(const DeviceInfo&)> callback);

private:
    // GTK组件
    GtkWidget* window_;                 // 主窗口
    GtkWidget* main_paned_;             // 主分割面板
    GtkWidget* left_frame_;             // 左侧框架
    GtkWidget* right_frame_;            // 右侧框架
    GtkWidget* video_area_;             // 视频显示区域
    GtkWidget* info_label_;             // 信息标签
    GtkWidget* source_list_;            // 源列表（窗口或设备）
    GtkWidget* source_scrolled_;        // 源列表滚动容器
    GtkWidget* mode_label_;             // 模式标签
    GtkWidget* refresh_button_;         // 刷新按钮

    // 状态变量
    CaptureMode current_mode_;
    std::vector<WindowInfo> available_windows_;
    std::vector<DeviceInfo> available_devices_;

    // 视频帧数据
    std::unique_ptr<guchar[]> frame_data_;
    int frame_width_;
    int frame_height_;
    int frame_channels_;

    // 缓存Cairo表面
    cairo_surface_t* cached_surface_;
    std::unique_ptr<guchar[]> cached_rgba_data_;
    int cached_width_;
    int cached_height_;

    // 回调函数
    std::function<void()> close_callback_;
    std::function<void(const WindowInfo&)> window_selection_callback_;
    std::function<void(const DeviceInfo&)> device_selection_callback_;

    /**
     * 初始化UI组件
     */
    void init_ui();

    /**
     * 创建左侧视频显示区域
     */
    void create_video_area();

    /**
     * 创建右侧源选择列表
     */
    void create_source_list();

    /**
     * 更新源列表内容
     */
    void update_source_list();

    /**
     * 获取可用窗口列表
     */
    void refresh_window_list();

    /**
     * 获取可用设备列表
     */
    void refresh_device_list();

    /**
     * 设置窗口样式
     */
    void setup_styling();

    // 静态回调函数
    static void on_draw_area(GtkDrawingArea* area, cairo_t* cr, int width, int height, gpointer user_data);
    static gboolean on_window_close(GtkWidget* widget, gpointer user_data);
    static void on_source_selection_changed(GtkListBox* list_box, GtkListBoxRow* row, gpointer user_data);
    static void on_refresh_button_clicked(GtkWidget* widget, gpointer user_data);
};

} // namespace gui
} // namespace duorou

#endif // DUOROU_GUI_ENHANCED_VIDEO_CAPTURE_WINDOW_H