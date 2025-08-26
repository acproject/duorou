#ifndef DUOROU_GUI_CHAT_VIEW_H
#define DUOROU_GUI_CHAT_VIEW_H

#include <gtk/gtk.h>
#include <string>
#include <vector>
#include <memory>
#include "video_display_window.h"

// 前向声明
namespace duorou {
namespace media {
    class VideoCapture;
    class AudioCapture;
}
}

namespace duorou {
namespace gui {

/**
 * 聊天视图类 - 处理文本生成模型的交互界面
 */
class ChatView {
public:
    ChatView();
    ~ChatView();

    // 禁用拷贝构造和赋值
    ChatView(const ChatView&) = delete;
    ChatView& operator=(const ChatView&) = delete;

    /**
     * 初始化聊天视图
     * @return 成功返回true，失败返回false
     */
    bool initialize();

    /**
     * 获取主要的GTK组件
     * @return GTK组件指针
     */
    GtkWidget* get_widget() const { return main_widget_; }

    /**
     * 发送消息
     * @param message 用户输入的消息
     */
    void send_message(const std::string& message);

    /**
     * 添加消息到聊天历史
     * @param message 消息内容
     * @param is_user 是否为用户消息
     */
    void add_message(const std::string& message, bool is_user);

    /**
     * 清空聊天历史
     */
    void clear_chat();

private:
    GtkWidget* main_widget_;         // 主容器
    GtkWidget* chat_scrolled_;       // 滚动窗口
    GtkWidget* chat_box_;            // 聊天消息容器
    GtkWidget* input_box_;           // 输入区域容器
    GtkWidget* input_entry_;         // 输入框
    GtkWidget* send_button_;         // 发送按钮
    GtkWidget* upload_image_button_; // 上传图片按钮
    GtkWidget* upload_file_button_;  // 上传文件按钮
    GtkWidget* video_record_button_; // 录制视频按钮
    
    // 存储选择的文件路径
    std::string selected_image_path_;
    std::string selected_file_path_;
    GtkWidget* model_selector_;      // 模型选择器
    GtkWidget* input_container_;     // 输入框容器
    
    bool welcome_cleared_;           // 标记是否已清除欢迎界面
    
    // 媒体捕获相关
    std::unique_ptr<media::VideoCapture> video_capture_;
    std::unique_ptr<media::AudioCapture> audio_capture_;
    std::unique_ptr<VideoDisplayWindow> video_display_window_;
    bool is_recording_;              // 录制状态标记
    
    // 录制按钮图标
    GtkWidget* video_off_image_;     // 录制关闭图标
    GtkWidget* video_on_image_;      // 录制开启图标

    /**
     * 创建聊天显示区域
     */
    void create_chat_area();

    /**
     * 创建欢迎界面
     */
    void create_welcome_screen();

    /**
     * 创建输入区域
     */
    void create_input_area();

    /**
     * 连接信号
     */
    void connect_signals();

    /**
     * 滚动到底部
     */
    void scroll_to_bottom();

    // 静态回调函数
    static void on_send_button_clicked(GtkWidget* widget, gpointer user_data);
    static void on_upload_image_button_clicked(GtkWidget* widget, gpointer user_data);
    static void on_upload_file_button_clicked(GtkWidget* widget, gpointer user_data);
    static void on_video_record_button_clicked(GtkWidget* widget, gpointer user_data);
    static void on_input_entry_activate(GtkWidget* widget, gpointer user_data);
    static void on_image_dialog_response(GtkDialog* dialog, gint response_id, gpointer user_data);
    static void on_file_dialog_response(GtkDialog* dialog, gint response_id, gpointer user_data);
    
    // 视频捕获方法
  
    void start_desktop_capture();
    void start_camera_capture();
    void stop_recording();
};

} // namespace gui
} // namespace duorou

#endif // DUOROU_GUI_CHAT_VIEW_H