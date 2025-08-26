#ifndef DUOROU_GUI_MAIN_WINDOW_H
#define DUOROU_GUI_MAIN_WINDOW_H

#include <gtk/gtk.h>
#include <memory>
#include <string>

namespace duorou {
namespace gui {

class ChatView;
class ImageView;
class SettingsDialog;

/**
 * 主窗口类 - 管理整个应用程序的主界面
 * 包含聊天界面、图像生成界面和设置面板的切换
 */
class MainWindow {
public:
    MainWindow();
    ~MainWindow();

    // 禁用拷贝构造和赋值
    MainWindow(const MainWindow&) = delete;
    MainWindow& operator=(const MainWindow&) = delete;

    /**
     * 初始化主窗口
     * @return 成功返回true，失败返回false
     */
    bool initialize();

    /**
     * 显示主窗口
     */
    void show();

    /**
     * 隐藏主窗口
     */
    void hide();

    /**
     * 获取GTK窗口指针
     * @return GTK窗口指针
     */
    GtkWidget* get_window() const { return window_; }

    /**
     * 设置窗口标题
     * @param title 窗口标题
     */
    void set_title(const std::string& title);

    /**
     * 切换到聊天界面
     */
    void switch_to_chat();

    /**
     * 切换到图像生成界面
     */
    void switch_to_image_generation();

    /**
     * 显示设置对话框
     */
    void show_settings();

    /**
     * 退出应用程序
     */
    void quit_application();

private:
    // GTK组件
    GtkWidget* window_;              // 主窗口
    GtkWidget* header_bar_;          // 标题栏
    GtkWidget* main_box_;            // 主容器
    GtkWidget* sidebar_;             // 侧边栏
    GtkWidget* content_stack_;       // 内容堆栈
    GtkWidget* status_bar_;          // 状态栏

    // 侧边栏按钮
    GtkWidget* chat_button_;         // 聊天按钮
    GtkWidget* image_button_;        // 图像生成按钮
    GtkWidget* settings_button_;     // 设置按钮

    // 子视图
    std::unique_ptr<ChatView> chat_view_;
    std::unique_ptr<ImageView> image_view_;
    std::unique_ptr<SettingsDialog> settings_dialog_;

    // 当前活动视图
    std::string current_view_;

    /**
     * 创建头部栏
     */
    void create_header_bar();

    /**
     * 创建侧边栏
     */
    void create_sidebar();

    /**
     * 创建内容区域
     */
    void create_content_area();

    /**
     * 创建状态栏
     */
    void create_status_bar();

    /**
     * 设置窗口样式
     */
    void setup_styling();

    /**
     * 连接信号处理器
     */
    void connect_signals();

    /**
     * 更新侧边栏按钮状态
     * @param active_button 当前激活的按钮
     */
    void update_sidebar_buttons(GtkWidget* active_button);

    // 静态回调函数
    static void on_chat_button_clicked(GtkWidget* widget, gpointer user_data);
    static void on_image_button_clicked(GtkWidget* widget, gpointer user_data);
    static void on_settings_button_clicked(GtkWidget* widget, gpointer user_data);
    static gboolean on_window_delete_event(GtkWindow* window, gpointer user_data);
    static void on_window_destroy(GtkWidget* widget, gpointer user_data);
};

} // namespace gui
} // namespace duorou

#endif // DUOROU_GUI_MAIN_WINDOW_H