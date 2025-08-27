#ifndef DUOROU_GUI_VIDEO_SOURCE_DIALOG_H
#define DUOROU_GUI_VIDEO_SOURCE_DIALOG_H

#include <gtk/gtk.h>
#include <functional>

namespace duorou {
namespace gui {

/**
 * 视频源选择对话框
 * 让用户选择录制桌面还是启动摄像头
 */
class VideoSourceDialog {
public:
    enum class VideoSource {
        DESKTOP_CAPTURE,
        CAMERA,
        CANCEL
    };

    VideoSourceDialog();
    ~VideoSourceDialog();

    // 禁用拷贝构造和赋值
    VideoSourceDialog(const VideoSourceDialog&) = delete;
    VideoSourceDialog& operator=(const VideoSourceDialog&) = delete;

    /**
     * 初始化对话框
     * @return 成功返回true，失败返回false
     */
    bool initialize();

    /**
     * 显示对话框
     * @param parent_window 父窗口
     * @param callback 用户选择后的回调函数
     */
    void show(GtkWidget* parent_window, std::function<void(VideoSource)> callback);

    /**
     * 隐藏对话框
     */
    void hide();

private:
    // GTK组件
    GtkWidget* dialog_;              // 对话框
    GtkWidget* content_box_;         // 内容容器
    GtkWidget* title_label_;         // 标题标签
    GtkWidget* desktop_button_;      // 桌面录制按钮
    GtkWidget* camera_button_;       // 摄像头按钮
    GtkWidget* cancel_button_;       // 取消按钮
    GtkWidget* button_box_;          // 按钮容器

    // 回调函数
    std::function<void(VideoSource)> selection_callback_;

    /**
     * 创建对话框内容
     */
    void create_content();

    /**
     * 设置对话框样式
     */
    void setup_styling();

    /**
     * 连接信号处理器
     */
    void connect_signals();

    /**
     * 处理用户选择
     * @param source 选择的视频源
     */
    void handle_selection(VideoSource source);

    // 静态回调函数
    static void on_desktop_button_clicked(GtkWidget* widget, gpointer user_data);
    static void on_camera_button_clicked(GtkWidget* widget, gpointer user_data);
    static void on_cancel_button_clicked(GtkWidget* widget, gpointer user_data);
    static gboolean on_dialog_delete_event(GtkWindow* window, gpointer user_data);
};

} // namespace gui
} // namespace duorou

#endif // DUOROU_GUI_VIDEO_SOURCE_DIALOG_H