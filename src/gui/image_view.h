#ifndef DUOROU_GUI_IMAGE_VIEW_H
#define DUOROU_GUI_IMAGE_VIEW_H

#include <gtk/gtk.h>
#include <string>

namespace duorou {
namespace gui {

/**
 * 图像视图类 - 处理图像生成模型的交互界面
 */
class ImageView {
public:
    ImageView();
    ~ImageView();

    // 禁用拷贝构造和赋值
    ImageView(const ImageView&) = delete;
    ImageView& operator=(const ImageView&) = delete;

    /**
     * 初始化图像视图
     * @return 成功返回true，失败返回false
     */
    bool initialize();

    /**
     * 获取主要的GTK组件
     * @return GTK组件指针
     */
    GtkWidget* get_widget() const { return main_widget_; }

    /**
     * 生成图像
     * @param prompt 图像生成提示词
     */
    void generate_image(const std::string& prompt);

    /**
     * 显示生成的图像
     * @param image_path 图像文件路径
     */
    void display_image(const std::string& image_path);

    /**
     * 清空当前图像
     */
    void clear_image();

private:
    GtkWidget* main_widget_;         // 主容器
    GtkWidget* prompt_box_;          // 提示词输入区域
    GtkWidget* prompt_entry_;        // 提示词输入框
    GtkWidget* generate_button_;     // 生成按钮
    GtkWidget* image_scrolled_;      // 图像滚动窗口
    GtkWidget* image_widget_;        // 图像显示组件
    GtkWidget* progress_bar_;        // 进度条
    GtkWidget* status_label_;        // 状态标签

    /**
     * 创建提示词输入区域
     */
    void create_prompt_area();

    /**
     * 创建图像显示区域
     */
    void create_image_area();

    /**
     * 创建状态区域
     */
    void create_status_area();

    /**
     * 连接信号处理器
     */
    void connect_signals();

    /**
     * 更新进度
     * @param progress 进度值 (0.0 - 1.0)
     */
    void update_progress(double progress);

    /**
     * 更新状态文本
     * @param status 状态文本
     */
    void update_status(const std::string& status);

    /**
     * 创建占位符图像
     */
    void create_placeholder_image();

    // 静态回调函数
    static void on_generate_button_clicked(GtkWidget* widget, gpointer user_data);
    static void on_prompt_entry_activate(GtkWidget* widget, gpointer user_data);
    static void on_clear_button_clicked(GtkWidget* widget, gpointer user_data);
};

} // namespace gui
} // namespace duorou

#endif // DUOROU_GUI_IMAGE_VIEW_H