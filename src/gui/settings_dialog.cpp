#include "settings_dialog.h"
#include "../core/logger.h"

#include <iostream>

namespace duorou {
namespace gui {

SettingsDialog::SettingsDialog()
    : dialog_(nullptr)
    , notebook_(nullptr)
    , general_page_(nullptr)
    , theme_combo_(nullptr)
    , language_combo_(nullptr)
    , startup_check_(nullptr)
    , model_page_(nullptr)
    , llama_model_entry_(nullptr)
    , sd_model_entry_(nullptr)
    , model_path_entry_(nullptr)
    , performance_page_(nullptr)
    , threads_spin_(nullptr)
    , gpu_check_(nullptr)
    , memory_spin_(nullptr)
{
}

SettingsDialog::~SettingsDialog() {
    // GTK4会自动清理子组件
}

bool SettingsDialog::initialize() {
    // 创建对话框
    dialog_ = gtk_dialog_new();
    if (!dialog_) {
        std::cerr << "Failed to create settings dialog" << std::endl;
        return false;
    }

    // 设置对话框属性
    gtk_window_set_title(GTK_WINDOW(dialog_), "Settings");
    gtk_window_set_default_size(GTK_WINDOW(dialog_), 600, 500);
    gtk_window_set_modal(GTK_WINDOW(dialog_), TRUE);
    gtk_window_set_resizable(GTK_WINDOW(dialog_), TRUE);

    // 获取对话框内容区域
    GtkWidget* content_area = gtk_dialog_get_content_area(GTK_DIALOG(dialog_));
    
    // 创建notebook用于分页
    notebook_ = gtk_notebook_new();
    gtk_widget_set_vexpand(notebook_, TRUE);
    gtk_widget_set_hexpand(notebook_, TRUE);
    
    // 创建各个设置页面
    create_general_page();
    create_model_page();
    create_performance_page();
    
    // 添加到内容区域
    gtk_box_append(GTK_BOX(content_area), notebook_);
    
    // 添加对话框按钮
    gtk_dialog_add_button(GTK_DIALOG(dialog_), "Cancel", GTK_RESPONSE_CANCEL);
    gtk_dialog_add_button(GTK_DIALOG(dialog_), "Apply", GTK_RESPONSE_APPLY);
    gtk_dialog_add_button(GTK_DIALOG(dialog_), "OK", GTK_RESPONSE_OK);
    
    // 连接信号
    connect_signals();
    
    // 加载当前设置
    load_settings();

    std::cout << "Settings dialog initialized successfully" << std::endl;
    return true;
}

void SettingsDialog::show(GtkWidget* parent) {
    if (dialog_) {
        if (parent) {
            gtk_window_set_transient_for(GTK_WINDOW(dialog_), GTK_WINDOW(parent));
        }
        gtk_widget_show(dialog_);
    }
}

void SettingsDialog::hide() {
    if (dialog_) {
        gtk_widget_hide(dialog_);
    }
}

void SettingsDialog::create_general_page() {
    general_page_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_widget_set_margin_start(general_page_, 20);
    gtk_widget_set_margin_end(general_page_, 20);
    gtk_widget_set_margin_top(general_page_, 20);
    gtk_widget_set_margin_bottom(general_page_, 20);
    
    // 应用程序设置组
    GtkWidget* app_group = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_widget_set_margin_bottom(app_group, 20);
    
    // 组标题
    GtkWidget* app_title = gtk_label_new("Application Settings");
    gtk_widget_set_halign(app_title, GTK_ALIGN_START);
    gtk_widget_add_css_class(app_title, "settings-group-title");
    gtk_widget_set_margin_bottom(app_title, 10);
    gtk_box_append(GTK_BOX(app_group), app_title);
    
    // 主题选择
    GtkWidget* theme_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    GtkWidget* theme_label = gtk_label_new("Theme:");
    gtk_widget_set_size_request(theme_label, 100, -1);
    gtk_widget_set_halign(theme_label, GTK_ALIGN_START);
    
    theme_combo_ = gtk_combo_box_text_new();
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(theme_combo_), "Light");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(theme_combo_), "Dark");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(theme_combo_), "Auto");
    gtk_combo_box_set_active(GTK_COMBO_BOX(theme_combo_), 0);
    
    gtk_box_append(GTK_BOX(theme_box), theme_label);
    gtk_box_append(GTK_BOX(theme_box), theme_combo_);
    gtk_widget_set_margin_start(theme_box, 15);
    gtk_widget_set_margin_bottom(theme_box, 5);
    gtk_box_append(GTK_BOX(app_group), theme_box);
    
    // 语言选择
    GtkWidget* language_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    GtkWidget* language_label = gtk_label_new("Language:");
    gtk_widget_set_size_request(language_label, 100, -1);
    gtk_widget_set_halign(language_label, GTK_ALIGN_START);
    
    language_combo_ = gtk_combo_box_text_new();
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(language_combo_), "English");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(language_combo_), "中文");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(language_combo_), "日本語");
    gtk_combo_box_set_active(GTK_COMBO_BOX(language_combo_), 0);
    
    gtk_box_append(GTK_BOX(language_box), language_label);
    gtk_box_append(GTK_BOX(language_box), language_combo_);
    gtk_widget_set_margin_start(language_box, 15);
    gtk_widget_set_margin_bottom(language_box, 5);
    gtk_box_append(GTK_BOX(app_group), language_box);
    
    // 启动时最小化到系统托盘
    startup_check_ = gtk_check_button_new_with_label("Minimize to system tray on startup");
    gtk_widget_set_margin_start(startup_check_, 15);
    gtk_widget_set_margin_bottom(startup_check_, 5);
    gtk_box_append(GTK_BOX(app_group), startup_check_);
    
    gtk_box_append(GTK_BOX(general_page_), app_group);
    
    // 添加到notebook
    gtk_notebook_append_page(GTK_NOTEBOOK(notebook_), general_page_, gtk_label_new("General"));
}

void SettingsDialog::create_model_page() {
    model_page_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_widget_set_margin_start(model_page_, 20);
    gtk_widget_set_margin_end(model_page_, 20);
    gtk_widget_set_margin_top(model_page_, 20);
    gtk_widget_set_margin_bottom(model_page_, 20);
    
    // 模型设置组
    GtkWidget* model_group = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_widget_set_margin_bottom(model_group, 20);
    
    // 组标题
    GtkWidget* model_title = gtk_label_new("Model Settings");
    gtk_widget_set_halign(model_title, GTK_ALIGN_START);
    gtk_widget_add_css_class(model_title, "settings-group-title");
    gtk_widget_set_margin_bottom(model_title, 10);
    gtk_box_append(GTK_BOX(model_group), model_title);
    
    // LLaMA模型路径
    GtkWidget* llama_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    GtkWidget* llama_label = gtk_label_new("LLaMA Model:");
    gtk_widget_set_size_request(llama_label, 120, -1);
    gtk_widget_set_halign(llama_label, GTK_ALIGN_START);
    
    llama_model_entry_ = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(llama_model_entry_), "Path to LLaMA model file...");
    gtk_widget_set_hexpand(llama_model_entry_, TRUE);
    
    GtkWidget* llama_browse = gtk_button_new_with_label("Browse");
    
    gtk_box_append(GTK_BOX(llama_box), llama_label);
    gtk_box_append(GTK_BOX(llama_box), llama_model_entry_);
    gtk_box_append(GTK_BOX(llama_box), llama_browse);
    gtk_widget_set_margin_start(llama_box, 15);
    gtk_widget_set_margin_bottom(llama_box, 5);
    gtk_box_append(GTK_BOX(model_group), llama_box);
    
    // Stable Diffusion模型路径
    GtkWidget* sd_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    GtkWidget* sd_label = gtk_label_new("SD Model:");
    gtk_widget_set_size_request(sd_label, 120, -1);
    gtk_widget_set_halign(sd_label, GTK_ALIGN_START);
    
    sd_model_entry_ = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(sd_model_entry_), "Path to Stable Diffusion model...");
    gtk_widget_set_hexpand(sd_model_entry_, TRUE);
    
    GtkWidget* sd_browse = gtk_button_new_with_label("Browse");
    
    gtk_box_append(GTK_BOX(sd_box), sd_label);
    gtk_box_append(GTK_BOX(sd_box), sd_model_entry_);
    gtk_box_append(GTK_BOX(sd_box), sd_browse);
    gtk_widget_set_margin_start(sd_box, 15);
    gtk_widget_set_margin_bottom(sd_box, 5);
    gtk_box_append(GTK_BOX(model_group), sd_box);
    
    // 模型存储路径
    GtkWidget* path_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    GtkWidget* path_label = gtk_label_new("Model Path:");
    gtk_widget_set_size_request(path_label, 120, -1);
    gtk_widget_set_halign(path_label, GTK_ALIGN_START);
    
    model_path_entry_ = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(model_path_entry_), "Directory for model storage...");
    gtk_widget_set_hexpand(model_path_entry_, TRUE);
    
    GtkWidget* path_browse = gtk_button_new_with_label("Browse");
    
    gtk_box_append(GTK_BOX(path_box), path_label);
    gtk_box_append(GTK_BOX(path_box), model_path_entry_);
    gtk_box_append(GTK_BOX(path_box), path_browse);
    gtk_widget_set_margin_start(path_box, 15);
    gtk_widget_set_margin_bottom(path_box, 5);
    gtk_box_append(GTK_BOX(model_group), path_box);
    
    gtk_box_append(GTK_BOX(model_page_), model_group);
    
    // 添加到notebook
    gtk_notebook_append_page(GTK_NOTEBOOK(notebook_), model_page_, gtk_label_new("Models"));
}

void SettingsDialog::create_performance_page() {
    performance_page_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_widget_set_margin_start(performance_page_, 20);
    gtk_widget_set_margin_end(performance_page_, 20);
    gtk_widget_set_margin_top(performance_page_, 20);
    gtk_widget_set_margin_bottom(performance_page_, 20);
    
    // 性能设置组
    GtkWidget* perf_group = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_widget_set_margin_bottom(perf_group, 20);
    
    // 组标题
    GtkWidget* perf_title = gtk_label_new("Performance Settings");
    gtk_widget_set_halign(perf_title, GTK_ALIGN_START);
    gtk_widget_add_css_class(perf_title, "settings-group-title");
    gtk_widget_set_margin_bottom(perf_title, 10);
    gtk_box_append(GTK_BOX(perf_group), perf_title);
    
    // 线程数设置
    GtkWidget* threads_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    GtkWidget* threads_label = gtk_label_new("CPU Threads:");
    gtk_widget_set_size_request(threads_label, 120, -1);
    gtk_widget_set_halign(threads_label, GTK_ALIGN_START);
    
    threads_spin_ = gtk_spin_button_new_with_range(1, 32, 1);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(threads_spin_), 4);
    
    gtk_box_append(GTK_BOX(threads_box), threads_label);
    gtk_box_append(GTK_BOX(threads_box), threads_spin_);
    gtk_widget_set_margin_start(threads_box, 15);
    gtk_widget_set_margin_bottom(threads_box, 5);
    gtk_box_append(GTK_BOX(perf_group), threads_box);
    
    // GPU加速
    gpu_check_ = gtk_check_button_new_with_label("Enable GPU acceleration (if available)");
    gtk_widget_set_margin_start(gpu_check_, 15);
    gtk_widget_set_margin_bottom(gpu_check_, 5);
    gtk_box_append(GTK_BOX(perf_group), gpu_check_);
    
    // 内存限制
    GtkWidget* memory_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    GtkWidget* memory_label = gtk_label_new("Memory Limit (GB):");
    gtk_widget_set_size_request(memory_label, 120, -1);
    gtk_widget_set_halign(memory_label, GTK_ALIGN_START);
    
    memory_spin_ = gtk_spin_button_new_with_range(1, 64, 1);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(memory_spin_), 8);
    
    gtk_box_append(GTK_BOX(memory_box), memory_label);
    gtk_box_append(GTK_BOX(memory_box), memory_spin_);
    gtk_widget_set_margin_start(memory_box, 15);
    gtk_widget_set_margin_bottom(memory_box, 5);
    gtk_box_append(GTK_BOX(perf_group), memory_box);
    
    gtk_box_append(GTK_BOX(performance_page_), perf_group);
    
    // 添加到notebook
    gtk_notebook_append_page(GTK_NOTEBOOK(notebook_), performance_page_, gtk_label_new("Performance"));
}

void SettingsDialog::connect_signals() {
    // 连接对话框响应信号
    g_signal_connect(dialog_, "response", G_CALLBACK(on_dialog_response), this);
}

void SettingsDialog::load_settings() {
    // TODO: 从配置文件加载设置
    std::cout << "Loading settings from configuration file..." << std::endl;
    
    // 设置默认值
    gtk_combo_box_set_active(GTK_COMBO_BOX(theme_combo_), 0);
    gtk_combo_box_set_active(GTK_COMBO_BOX(language_combo_), 0);
    gtk_check_button_set_active(GTK_CHECK_BUTTON(startup_check_), FALSE);
    gtk_check_button_set_active(GTK_CHECK_BUTTON(gpu_check_), FALSE);
}

void SettingsDialog::save_settings() {
    // TODO: 保存设置到配置文件
    std::cout << "Saving settings to configuration file..." << std::endl;
    
    // 获取当前设置值
    int theme_index = gtk_combo_box_get_active(GTK_COMBO_BOX(theme_combo_));
    int language_index = gtk_combo_box_get_active(GTK_COMBO_BOX(language_combo_));
    gboolean startup_minimize = gtk_check_button_get_active(GTK_CHECK_BUTTON(startup_check_));
    gboolean gpu_enabled = gtk_check_button_get_active(GTK_CHECK_BUTTON(gpu_check_));
    int threads = gtk_spin_button_get_value_as_int(GTK_SPIN_BUTTON(threads_spin_));
    int memory_limit = gtk_spin_button_get_value_as_int(GTK_SPIN_BUTTON(memory_spin_));
    
    const char* llama_path = gtk_entry_buffer_get_text(gtk_entry_get_buffer(GTK_ENTRY(llama_model_entry_)));
    const char* sd_path = gtk_entry_buffer_get_text(gtk_entry_get_buffer(GTK_ENTRY(sd_model_entry_)));
    const char* model_path = gtk_entry_buffer_get_text(gtk_entry_get_buffer(GTK_ENTRY(model_path_entry_)));
    
    // 这里应该保存到配置文件
    std::cout << "Theme: " << theme_index << ", Language: " << language_index << std::endl;
    std::cout << "Startup minimize: " << startup_minimize << ", GPU: " << gpu_enabled << std::endl;
    std::cout << "Threads: " << threads << ", Memory: " << memory_limit << "GB" << std::endl;
    std::cout << "LLaMA: " << (llama_path ? llama_path : "(empty)") << std::endl;
    std::cout << "SD: " << (sd_path ? sd_path : "(empty)") << std::endl;
    std::cout << "Model path: " << (model_path ? model_path : "(empty)") << std::endl;
}

void SettingsDialog::reset_to_defaults() {
    // TODO: 重置所有设置为默认值
    std::cout << "Resetting settings to default values..." << std::endl;
    
    // 重置UI控件为默认值
    gtk_combo_box_set_active(GTK_COMBO_BOX(theme_combo_), 0);
    gtk_combo_box_set_active(GTK_COMBO_BOX(language_combo_), 0);
    gtk_check_button_set_active(GTK_CHECK_BUTTON(startup_check_), FALSE);
    gtk_check_button_set_active(GTK_CHECK_BUTTON(gpu_check_), FALSE);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(threads_spin_), 4);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(memory_spin_), 8);
    
    gtk_entry_buffer_set_text(gtk_entry_get_buffer(GTK_ENTRY(llama_model_entry_)), "", 0);
    gtk_entry_buffer_set_text(gtk_entry_get_buffer(GTK_ENTRY(sd_model_entry_)), "", 0);
    gtk_entry_buffer_set_text(gtk_entry_get_buffer(GTK_ENTRY(model_path_entry_)), "", 0);
}

// 静态回调函数实现
void SettingsDialog::on_ok_button_clicked(GtkWidget* widget, gpointer user_data) {
    SettingsDialog* dialog = static_cast<SettingsDialog*>(user_data);
    dialog->save_settings();
    dialog->hide();
}

void SettingsDialog::on_cancel_button_clicked(GtkWidget* widget, gpointer user_data) {
    SettingsDialog* dialog = static_cast<SettingsDialog*>(user_data);
    dialog->hide();
}

void SettingsDialog::on_apply_button_clicked(GtkWidget* widget, gpointer user_data) {
    SettingsDialog* dialog = static_cast<SettingsDialog*>(user_data);
    dialog->save_settings();
}

void SettingsDialog::on_reset_button_clicked(GtkWidget* widget, gpointer user_data) {
    SettingsDialog* dialog = static_cast<SettingsDialog*>(user_data);
    dialog->reset_to_defaults();
}

// 对话框响应处理
void SettingsDialog::on_dialog_response(GtkDialog* dialog, gint response_id, gpointer user_data) {
    SettingsDialog* settings_dialog = static_cast<SettingsDialog*>(user_data);
    
    switch (response_id) {
        case GTK_RESPONSE_OK:
            settings_dialog->save_settings();
            settings_dialog->hide();
            break;
        case GTK_RESPONSE_CANCEL:
            settings_dialog->hide();
            break;
        case GTK_RESPONSE_APPLY:
            settings_dialog->save_settings();
            break;
        default:
            break;
    }
}

} // namespace gui
} // namespace duorou