#ifndef DUOROU_GUI_MAIN_WINDOW_H
#define DUOROU_GUI_MAIN_WINDOW_H

#include <gtk/gtk.h>
#include <memory>
#include <string>
#include <vector>

#ifdef __APPLE__
#include "../platform/macos_tray.h"
#endif

namespace duorou {
namespace gui {

class ChatView;
class ImageView;
class SettingsDialog;
class ChatSessionManager;
class SystemTray;

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

    /**
     * 设置系统托盘状态
     * @param status 状态描述（如："idle", "processing", "error"）
     */
    void set_tray_status(const std::string& status);

    /**
     * 从系统托盘恢复窗口显示
     */
    void restore_from_tray();

    /**
     * 创建新的聊天会话
     */
    void create_new_chat();

    /**
     * 切换到指定聊天会话
     * @param session_id 会话ID
     */
    void switch_to_chat_session(const std::string& session_id);

private:
    // GTK组件
    GtkWidget* window_;              // 主窗口
    GtkWidget* header_bar_;          // 标题栏
    GtkWidget* main_box_;            // 主容器
    GtkWidget* sidebar_;             // 侧边栏
    GtkWidget* content_stack_;       // 内容堆栈
    GtkWidget* status_bar_;          // 状态栏

    // 侧边栏按钮
    GtkWidget* new_chat_button_;     // 新建聊天按钮
    GtkWidget* image_button_;        // 图像生成按钮
    GtkWidget* settings_button_;     // 设置按钮
    GtkWidget* chat_history_box_;    // 聊天历史容器

    // UI组件
    std::unique_ptr<ChatView> chat_view_;
    std::unique_ptr<ImageView> image_view_;
    std::unique_ptr<SettingsDialog> settings_dialog_;
    std::unique_ptr<ChatSessionManager> session_manager_;
    std::unique_ptr<SystemTray> system_tray_;
    
#ifdef __APPLE__
    std::unique_ptr<MacOSTray> macos_tray_;
#endif

    // 当前视图状态
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
     * @param active_button 当前活动按钮
     */
    void update_sidebar_buttons(GtkWidget* active_button);

    /**
     * 更新聊天历史列表
     */
    void update_chat_history_list();

    /**
     * 会话变更回调
     * @param session_id 新的会话ID
     */
    void on_session_changed(const std::string& session_id);

    /**
     * 会话列表变更回调
     */
    void on_session_list_changed();

    // 静态回调函数
    static void on_new_chat_button_clicked(GtkWidget* widget, gpointer user_data);
    static void on_chat_history_item_clicked(GtkWidget* widget, gpointer user_data);
    static void on_image_button_clicked(GtkWidget* widget, gpointer user_data);
    static void on_settings_button_clicked(GtkWidget* widget, gpointer user_data);
    static gboolean on_window_delete_event(GtkWindow* window, gpointer user_data);
    static void on_window_destroy(GtkWidget* widget, gpointer user_data);
};

} // namespace gui
} // namespace duorou

#endif // DUOROU_GUI_MAIN_WINDOW_H