#ifndef DUOROU_GUI_SETTINGS_DIALOG_H
#define DUOROU_GUI_SETTINGS_DIALOG_H

#include <gtk/gtk.h>
#include <string>

namespace duorou {
namespace gui {

/**
 * 设置对话框类 - 管理应用程序配置
 */
class SettingsDialog {
public:
    SettingsDialog();
    ~SettingsDialog();

    // 禁用拷贝构造和赋值
    SettingsDialog(const SettingsDialog&) = delete;
    SettingsDialog& operator=(const SettingsDialog&) = delete;

    /**
     * 初始化设置对话框
     * @return 成功返回true，失败返回false
     */
    bool initialize();

    /**
     * 显示设置对话框
     * @param parent 父窗口
     */
    void show(GtkWidget* parent);

    /**
     * 隐藏设置对话框
     */
    void hide();

private:
    GtkWidget* dialog_;              // 对话框
    GtkWidget* notebook_;            // 标签页容器
    
    // 通用设置页面
    GtkWidget* general_page_;
    GtkWidget* theme_combo_;
    GtkWidget* language_combo_;
    GtkWidget* startup_check_;
    
    // 模型设置页面
    GtkWidget* model_page_;
    GtkWidget* llama_model_entry_;
    GtkWidget* sd_model_entry_;
    GtkWidget* model_path_entry_;
    
    // 性能设置页面
    GtkWidget* performance_page_;
    GtkWidget* threads_spin_;
    GtkWidget* gpu_check_;
    GtkWidget* memory_spin_;

    /**
     * 创建通用设置页面
     */
    void create_general_page();

    /**
     * 创建模型设置页面
     */
    void create_model_page();

    /**
     * 创建性能设置页面
     */
    void create_performance_page();

    /**
     * 连接信号处理器
     */
    void connect_signals();

    /**
     * 加载当前设置
     */
    void load_settings();

    /**
     * 保存设置
     */
    void save_settings();

    /**
     * 重置为默认设置
     */
    void reset_to_defaults();

    // 静态回调函数
    static void on_ok_button_clicked(GtkWidget* widget, gpointer user_data);
    static void on_cancel_button_clicked(GtkWidget* widget, gpointer user_data);
    static void on_apply_button_clicked(GtkWidget* widget, gpointer user_data);
    static void on_reset_button_clicked(GtkWidget* widget, gpointer user_data);
    static void on_dialog_response(GtkDialog* dialog, gint response_id, gpointer user_data);
};

} // namespace gui
} // namespace duorou

#endif // DUOROU_GUI_SETTINGS_DIALOG_H