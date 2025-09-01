#include "settings_dialog.h"
#include "../core/application.h"
#include "../core/config_manager.h"
#include "../core/logger.h"
#include "../core/model_manager.h"

#include <iostream>
#include <gtk/gtk.h>

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
    , llama_model_combo_(nullptr)
    , sd_model_entry_(nullptr)
    , sd_vae_entry_(nullptr)
    , sd_controlnet_entry_(nullptr)
    , sd_lora_entry_(nullptr)
    , model_path_entry_(nullptr)
    , ollama_path_entry_(nullptr)
    , performance_page_(nullptr)
    , threads_spin_(nullptr)
    , gpu_check_(nullptr)
    , memory_spin_(nullptr)
    , application_(nullptr)
{
}

SettingsDialog::SettingsDialog(core::Application* app)
    : dialog_(nullptr)
    , notebook_(nullptr)
    , general_page_(nullptr)
    , theme_combo_(nullptr)
    , language_combo_(nullptr)
    , startup_check_(nullptr)
    , model_page_(nullptr)
    , llama_model_combo_(nullptr)
    , sd_model_entry_(nullptr)
    , sd_vae_entry_(nullptr)
    , sd_controlnet_entry_(nullptr)
    , sd_lora_entry_(nullptr)
    , model_path_entry_(nullptr)
    , ollama_path_entry_(nullptr)
    , performance_page_(nullptr)
    , threads_spin_(nullptr)
    , gpu_check_(nullptr)
    , memory_spin_(nullptr)
    , application_(app)
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
        
        // 确保对话框在最顶层显示
        gtk_window_set_modal(GTK_WINDOW(dialog_), TRUE);
        
        // 显示对话框并确保获得焦点
        gtk_widget_show(dialog_);
        gtk_window_present(GTK_WINDOW(dialog_));
        
        // 强制获取焦点
        gtk_widget_grab_focus(dialog_);
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
    
    // LLaMA模型选择下拉菜单
    GtkWidget* llama_combo_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    GtkWidget* llama_combo_label = gtk_label_new("LLaMA Model:");
    gtk_widget_set_size_request(llama_combo_label, 120, -1);
    gtk_widget_set_halign(llama_combo_label, GTK_ALIGN_START);
    
    llama_model_combo_ = gtk_combo_box_text_new();
    gtk_widget_set_hexpand(llama_model_combo_, TRUE);
    
    gtk_box_append(GTK_BOX(llama_combo_box), llama_combo_label);
    gtk_box_append(GTK_BOX(llama_combo_box), llama_model_combo_);
    gtk_widget_set_margin_start(llama_combo_box, 15);
    gtk_widget_set_margin_bottom(llama_combo_box, 5);
    gtk_box_append(GTK_BOX(model_group), llama_combo_box);
    
    // Llama.cpp模型路径
    GtkWidget* llama_path_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    GtkWidget* llama_path_label = gtk_label_new("Llama.cpp Models Path:");
    gtk_widget_set_size_request(llama_path_label, 120, -1);
    gtk_widget_set_halign(llama_path_label, GTK_ALIGN_START);
    
    model_path_entry_ = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(model_path_entry_), "Directory for Llama.cpp model storage...");
    gtk_widget_set_hexpand(model_path_entry_, TRUE);
    
    GtkWidget* llama_path_browse = gtk_button_new_with_label("Browse");
    g_signal_connect(llama_path_browse, "clicked", G_CALLBACK(on_model_path_browse_clicked), this);
    
    gtk_box_append(GTK_BOX(llama_path_box), llama_path_label);
    gtk_box_append(GTK_BOX(llama_path_box), model_path_entry_);
    gtk_box_append(GTK_BOX(llama_path_box), llama_path_browse);
    gtk_widget_set_margin_start(llama_path_box, 15);
    gtk_widget_set_margin_bottom(llama_path_box, 5);
    gtk_box_append(GTK_BOX(model_group), llama_path_box);
    
    // Ollama模型路径
    GtkWidget* ollama_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    GtkWidget* ollama_label = gtk_label_new("Ollama Models Path:");
    gtk_widget_set_size_request(ollama_label, 120, -1);
    gtk_widget_set_halign(ollama_label, GTK_ALIGN_START);
    
    ollama_path_entry_ = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(ollama_path_entry_), "Directory for Ollama model storage...");
    gtk_widget_set_hexpand(ollama_path_entry_, TRUE);
    
    GtkWidget* ollama_browse = gtk_button_new_with_label("Browse");
    g_signal_connect(ollama_browse, "clicked", G_CALLBACK(on_ollama_path_browse_clicked), this);
    
    gtk_box_append(GTK_BOX(ollama_box), ollama_label);
    gtk_box_append(GTK_BOX(ollama_box), ollama_path_entry_);
    gtk_box_append(GTK_BOX(ollama_box), ollama_browse);
    gtk_widget_set_margin_start(ollama_box, 15);
    gtk_widget_set_margin_bottom(ollama_box, 5);
    gtk_box_append(GTK_BOX(model_group), ollama_box);
    
    // 移除Custom Path配置项，因为已有Model Path作为Llama.cpp模型存储位置
    
    // Stable Diffusion模型配置组
    GtkWidget* sd_group = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_widget_set_margin_top(sd_group, 10);
    gtk_widget_set_margin_bottom(sd_group, 10);
    
    // SD组标题
    GtkWidget* sd_title = gtk_label_new("Stable Diffusion Models");
    gtk_widget_set_halign(sd_title, GTK_ALIGN_START);
    gtk_widget_add_css_class(sd_title, "settings-subsection-title");
    gtk_widget_set_margin_bottom(sd_title, 5);
    gtk_box_append(GTK_BOX(sd_group), sd_title);
    
    // 主模型路径
    GtkWidget* sd_main_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    GtkWidget* sd_main_label = gtk_label_new("Main Model:");
    gtk_widget_set_size_request(sd_main_label, 120, -1);
    gtk_widget_set_halign(sd_main_label, GTK_ALIGN_START);
    
    sd_model_entry_ = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(sd_model_entry_), "Directory for main SD models...");
    gtk_widget_set_hexpand(sd_model_entry_, TRUE);
    
    GtkWidget* sd_main_browse = gtk_button_new_with_label("Browse");
    g_signal_connect(sd_main_browse, "clicked", G_CALLBACK(on_sd_browse_clicked), this);
    
    gtk_box_append(GTK_BOX(sd_main_box), sd_main_label);
    gtk_box_append(GTK_BOX(sd_main_box), sd_model_entry_);
    gtk_box_append(GTK_BOX(sd_main_box), sd_main_browse);
    gtk_widget_set_margin_start(sd_main_box, 15);
    gtk_widget_set_margin_bottom(sd_main_box, 5);
    gtk_box_append(GTK_BOX(sd_group), sd_main_box);
    
    // VAE模型路径
    GtkWidget* sd_vae_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    GtkWidget* sd_vae_label = gtk_label_new("VAE Model:");
    gtk_widget_set_size_request(sd_vae_label, 120, -1);
    gtk_widget_set_halign(sd_vae_label, GTK_ALIGN_START);
    
    sd_vae_entry_ = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(sd_vae_entry_), "Directory for VAE models (optional)...");
    gtk_widget_set_hexpand(sd_vae_entry_, TRUE);
    
    GtkWidget* sd_vae_browse = gtk_button_new_with_label("Browse");
    g_signal_connect(sd_vae_browse, "clicked", G_CALLBACK(on_sd_vae_browse_clicked), this);
    
    gtk_box_append(GTK_BOX(sd_vae_box), sd_vae_label);
    gtk_box_append(GTK_BOX(sd_vae_box), sd_vae_entry_);
    gtk_box_append(GTK_BOX(sd_vae_box), sd_vae_browse);
    gtk_widget_set_margin_start(sd_vae_box, 15);
    gtk_widget_set_margin_bottom(sd_vae_box, 5);
    gtk_box_append(GTK_BOX(sd_group), sd_vae_box);
    
    // ControlNet模型路径
    GtkWidget* sd_controlnet_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    GtkWidget* sd_controlnet_label = gtk_label_new("ControlNet:");
    gtk_widget_set_size_request(sd_controlnet_label, 120, -1);
    gtk_widget_set_halign(sd_controlnet_label, GTK_ALIGN_START);
    
    sd_controlnet_entry_ = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(sd_controlnet_entry_), "Directory for ControlNet models (optional)...");
    gtk_widget_set_hexpand(sd_controlnet_entry_, TRUE);
    
    GtkWidget* sd_controlnet_browse = gtk_button_new_with_label("Browse");
    g_signal_connect(sd_controlnet_browse, "clicked", G_CALLBACK(on_sd_controlnet_browse_clicked), this);
    
    gtk_box_append(GTK_BOX(sd_controlnet_box), sd_controlnet_label);
    gtk_box_append(GTK_BOX(sd_controlnet_box), sd_controlnet_entry_);
    gtk_box_append(GTK_BOX(sd_controlnet_box), sd_controlnet_browse);
    gtk_widget_set_margin_start(sd_controlnet_box, 15);
    gtk_widget_set_margin_bottom(sd_controlnet_box, 5);
    gtk_box_append(GTK_BOX(sd_group), sd_controlnet_box);
    
    // LoRA模型路径
    GtkWidget* sd_lora_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    GtkWidget* sd_lora_label = gtk_label_new("LoRA Models:");
    gtk_widget_set_size_request(sd_lora_label, 120, -1);
    gtk_widget_set_halign(sd_lora_label, GTK_ALIGN_START);
    
    sd_lora_entry_ = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(sd_lora_entry_), "Directory for LoRA models (optional)...");
    gtk_widget_set_hexpand(sd_lora_entry_, TRUE);
    
    GtkWidget* sd_lora_browse = gtk_button_new_with_label("Browse");
    g_signal_connect(sd_lora_browse, "clicked", G_CALLBACK(on_sd_lora_browse_clicked), this);
    
    gtk_box_append(GTK_BOX(sd_lora_box), sd_lora_label);
    gtk_box_append(GTK_BOX(sd_lora_box), sd_lora_entry_);
    gtk_box_append(GTK_BOX(sd_lora_box), sd_lora_browse);
    gtk_widget_set_margin_start(sd_lora_box, 15);
    gtk_widget_set_margin_bottom(sd_lora_box, 5);
    gtk_box_append(GTK_BOX(sd_group), sd_lora_box);
    
    gtk_box_append(GTK_BOX(model_group), sd_group);
    
    gtk_box_append(GTK_BOX(model_page_), model_group);
    
    // 添加到notebook
    gtk_notebook_append_page(GTK_NOTEBOOK(notebook_), model_page_, gtk_label_new("Models"));
    
    // 刷新模型列表
    refresh_model_list();
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
    if (!application_) {
        std::cout << "Warning: Application instance not available, using default values" << std::endl;
        // 设置默认值
        gtk_combo_box_set_active(GTK_COMBO_BOX(theme_combo_), 0);
        gtk_combo_box_set_active(GTK_COMBO_BOX(language_combo_), 0);
        gtk_check_button_set_active(GTK_CHECK_BUTTON(startup_check_), FALSE);
        gtk_check_button_set_active(GTK_CHECK_BUTTON(gpu_check_), FALSE);
        gtk_spin_button_set_value(GTK_SPIN_BUTTON(threads_spin_), 4);
        gtk_spin_button_set_value(GTK_SPIN_BUTTON(memory_spin_), 8);
        return;
    }
    
    auto* config_manager = application_->getConfigManager();
    if (!config_manager) {
        std::cout << "Warning: ConfigManager not available, using default values" << std::endl;
        // 设置默认值
        gtk_combo_box_set_active(GTK_COMBO_BOX(theme_combo_), 0);
        gtk_combo_box_set_active(GTK_COMBO_BOX(language_combo_), 0);
        gtk_check_button_set_active(GTK_CHECK_BUTTON(startup_check_), FALSE);
        gtk_check_button_set_active(GTK_CHECK_BUTTON(gpu_check_), FALSE);
        gtk_spin_button_set_value(GTK_SPIN_BUTTON(threads_spin_), 4);
        gtk_spin_button_set_value(GTK_SPIN_BUTTON(memory_spin_), 8);
        return;
    }
    
    std::cout << "Loading settings from configuration file..." << std::endl;
    
    // 从配置文件加载设置
    int theme_index = config_manager->getInt("ui.theme", 0);
    int language_index = config_manager->getInt("ui.language", 0);
    bool startup_minimize = config_manager->getBool("ui.startup_minimize", false);
    bool gpu_enabled = config_manager->getBool("performance.gpu_enabled", false);
    int threads = config_manager->getInt("performance.threads", 4);
    int memory_limit = config_manager->getInt("performance.memory_limit", 8);
    
    std::string llama_selected = config_manager->getString("models.llama_selected", "");
    std::string sd_path = config_manager->getString("models.sd_path", "");
    std::string sd_vae_path = config_manager->getString("models.sd_vae_path", "");
    std::string sd_controlnet_path = config_manager->getString("models.sd_controlnet_path", "");
    std::string sd_lora_path = config_manager->getString("models.sd_lora_path", "");
    std::string model_path = config_manager->getString("models.model_path", "");
    std::string ollama_path = config_manager->getString("models.ollama_path", "");
    
    // 设置UI控件的值
    gtk_combo_box_set_active(GTK_COMBO_BOX(theme_combo_), theme_index);
    gtk_combo_box_set_active(GTK_COMBO_BOX(language_combo_), language_index);
    gtk_check_button_set_active(GTK_CHECK_BUTTON(startup_check_), startup_minimize);
    gtk_check_button_set_active(GTK_CHECK_BUTTON(gpu_check_), gpu_enabled);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(threads_spin_), threads);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(memory_spin_), memory_limit);
    
    // 设置模型路径
    gtk_entry_buffer_set_text(gtk_entry_get_buffer(GTK_ENTRY(sd_model_entry_)), sd_path.c_str(), sd_path.length());
    gtk_entry_buffer_set_text(gtk_entry_get_buffer(GTK_ENTRY(sd_vae_entry_)), sd_vae_path.c_str(), sd_vae_path.length());
    gtk_entry_buffer_set_text(gtk_entry_get_buffer(GTK_ENTRY(sd_controlnet_entry_)), sd_controlnet_path.c_str(), sd_controlnet_path.length());
    gtk_entry_buffer_set_text(gtk_entry_get_buffer(GTK_ENTRY(sd_lora_entry_)), sd_lora_path.c_str(), sd_lora_path.length());
    gtk_entry_buffer_set_text(gtk_entry_get_buffer(GTK_ENTRY(model_path_entry_)), model_path.c_str(), model_path.length());
    gtk_entry_buffer_set_text(gtk_entry_get_buffer(GTK_ENTRY(ollama_path_entry_)), ollama_path.c_str(), ollama_path.length());
    
    // 如果有保存的模型选择，尝试设置到下拉菜单
    if (!llama_selected.empty() && llama_model_combo_) {
        // 查找匹配的模型选项
        GtkTreeModel* model = gtk_combo_box_get_model(GTK_COMBO_BOX(llama_model_combo_));
        GtkTreeIter iter;
        gboolean valid = gtk_tree_model_get_iter_first(model, &iter);
        int index = 0;
        
        while (valid) {
            gchar* text;
            gtk_tree_model_get(model, &iter, 0, &text, -1);
            if (text && llama_selected == text) {
                gtk_combo_box_set_active(GTK_COMBO_BOX(llama_model_combo_), index);
                g_free(text);
                break;
            }
            if (text) g_free(text);
            valid = gtk_tree_model_iter_next(model, &iter);
            index++;
        }
    }
    
    std::cout << "Settings loaded successfully" << std::endl;
}

void SettingsDialog::save_settings() {
    if (!application_) {
        std::cout << "Error: Application instance not available" << std::endl;
        return;
    }
    
    auto* config_manager = application_->getConfigManager();
    if (!config_manager) {
        std::cout << "Error: ConfigManager not available" << std::endl;
        return;
    }
    
    // 获取当前设置值
    int theme_index = gtk_combo_box_get_active(GTK_COMBO_BOX(theme_combo_));
    int language_index = gtk_combo_box_get_active(GTK_COMBO_BOX(language_combo_));
    gboolean startup_minimize = gtk_check_button_get_active(GTK_CHECK_BUTTON(startup_check_));
    gboolean gpu_enabled = gtk_check_button_get_active(GTK_CHECK_BUTTON(gpu_check_));
    int threads = gtk_spin_button_get_value_as_int(GTK_SPIN_BUTTON(threads_spin_));
    int memory_limit = gtk_spin_button_get_value_as_int(GTK_SPIN_BUTTON(memory_spin_));
    
    // 获取选中的LLaMA模型
    gchar* selected_model = nullptr;
    if (llama_model_combo_) {
        selected_model = gtk_combo_box_text_get_active_text(GTK_COMBO_BOX_TEXT(llama_model_combo_));
    }
    
    const char* sd_path = gtk_entry_buffer_get_text(gtk_entry_get_buffer(GTK_ENTRY(sd_model_entry_)));
    const char* sd_vae_path = gtk_entry_buffer_get_text(gtk_entry_get_buffer(GTK_ENTRY(sd_vae_entry_)));
    const char* sd_controlnet_path = gtk_entry_buffer_get_text(gtk_entry_get_buffer(GTK_ENTRY(sd_controlnet_entry_)));
    const char* sd_lora_path = gtk_entry_buffer_get_text(gtk_entry_get_buffer(GTK_ENTRY(sd_lora_entry_)));
    const char* model_path = gtk_entry_buffer_get_text(gtk_entry_get_buffer(GTK_ENTRY(model_path_entry_)));
    const char* ollama_path = gtk_entry_buffer_get_text(gtk_entry_get_buffer(GTK_ENTRY(ollama_path_entry_)));
    
    // 保存设置到配置文件
    config_manager->setInt("ui.theme", theme_index);
    config_manager->setInt("ui.language", language_index);
    config_manager->setBool("ui.startup_minimize", startup_minimize);
    config_manager->setBool("performance.gpu_enabled", gpu_enabled);
    config_manager->setInt("performance.threads", threads);
    config_manager->setInt("performance.memory_limit", memory_limit);
    
    if (selected_model) {
        config_manager->setString("models.llama_selected", selected_model);
    }
    
    if (sd_path && strlen(sd_path) > 0) {
        config_manager->setString("models.sd_path", sd_path);
    }
    
    if (sd_vae_path && strlen(sd_vae_path) > 0) {
        config_manager->setString("models.sd_vae_path", sd_vae_path);
    }
    
    if (sd_controlnet_path && strlen(sd_controlnet_path) > 0) {
        config_manager->setString("models.sd_controlnet_path", sd_controlnet_path);
    }
    
    if (sd_lora_path && strlen(sd_lora_path) > 0) {
        config_manager->setString("models.sd_lora_path", sd_lora_path);
    }
    
    if (model_path && strlen(model_path) > 0) {
        config_manager->setString("models.model_path", model_path);
    }
    
    if (ollama_path && strlen(ollama_path) > 0) {
        config_manager->setString("models.ollama_path", ollama_path);
    }
    
    // 保存配置到文件
    if (config_manager->saveConfig()) {
        std::cout << "Settings saved successfully" << std::endl;
    } else {
        std::cout << "Error: Failed to save settings" << std::endl;
    }
    
    if (selected_model) {
        g_free(selected_model);
    }
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
    
    // 重置模型选择
    if (llama_model_combo_) {
        gtk_combo_box_set_active(GTK_COMBO_BOX(llama_model_combo_), 0);
    }
    
    gtk_entry_buffer_set_text(gtk_entry_get_buffer(GTK_ENTRY(sd_model_entry_)), "", 0);
    gtk_entry_buffer_set_text(gtk_entry_get_buffer(GTK_ENTRY(sd_vae_entry_)), "", 0);
    gtk_entry_buffer_set_text(gtk_entry_get_buffer(GTK_ENTRY(sd_controlnet_entry_)), "", 0);
    gtk_entry_buffer_set_text(gtk_entry_get_buffer(GTK_ENTRY(sd_lora_entry_)), "", 0);
    gtk_entry_buffer_set_text(gtk_entry_get_buffer(GTK_ENTRY(model_path_entry_)), "", 0);
    gtk_entry_buffer_set_text(gtk_entry_get_buffer(GTK_ENTRY(ollama_path_entry_)), "", 0);
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
void SettingsDialog::set_application(core::Application* app) {
    application_ = app;
    refresh_model_list();
}

void SettingsDialog::refresh_model_list() {
    if (!llama_model_combo_ || !application_) {
        return;
    }
    
    // 清空现有选项
    gtk_combo_box_text_remove_all(GTK_COMBO_BOX_TEXT(llama_model_combo_));
    
    // 从ModelManager获取真实的可用模型列表
    auto model_manager = application_->getModelManager();
    if (model_manager) {
        std::vector<std::string> local_models = model_manager->getLocalModels();
        
        if (local_models.empty()) {
            // 如果没有本地模型，添加提示信息
            gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(llama_model_combo_), "No models found");
        } else {
            // 添加所有本地模型
            for (const auto& model : local_models) {
                gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(llama_model_combo_), model.c_str());
            }
        }
    } else {
        // 如果ModelManager不可用，显示错误信息
        gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(llama_model_combo_), "Model manager unavailable");
    }
    
    // 设置默认选择
    gtk_combo_box_set_active(GTK_COMBO_BOX(llama_model_combo_), 0);
}

// Llama模型现在通过下拉菜单选择，不再需要文件对话框

void SettingsDialog::on_sd_file_dialog_response(GtkNativeDialog* dialog, gint response, gpointer user_data) {
    SettingsDialog* settings = static_cast<SettingsDialog*>(user_data);
    if (response == GTK_RESPONSE_ACCEPT) {
        GtkFileChooser* chooser = GTK_FILE_CHOOSER(dialog);
        GFile* file = gtk_file_chooser_get_file(chooser);
        if (file) {
            char* filename = g_file_get_path(file);
            if (filename) {
                gtk_editable_set_text(GTK_EDITABLE(settings->sd_model_entry_), filename);
                g_free(filename);
            }
            g_object_unref(file);
        }
    }
    g_object_unref(dialog);
}

void SettingsDialog::on_sd_browse_clicked(GtkWidget* widget, gpointer user_data) {
    SettingsDialog* settings = static_cast<SettingsDialog*>(user_data);
    
    GtkFileChooserNative* dialog = gtk_file_chooser_native_new(
        "Select Main SD Models Directory",
        GTK_WINDOW(settings->dialog_),
        GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER,
        "Select",
        "Cancel");
    
    g_signal_connect(dialog, "response", G_CALLBACK(on_sd_file_dialog_response), settings);
    
    gtk_native_dialog_show(GTK_NATIVE_DIALOG(dialog));
}

void SettingsDialog::on_sd_vae_file_dialog_response(GtkNativeDialog* dialog, gint response, gpointer user_data) {
    SettingsDialog* settings = static_cast<SettingsDialog*>(user_data);
    if (response == GTK_RESPONSE_ACCEPT) {
        GtkFileChooser* chooser = GTK_FILE_CHOOSER(dialog);
        GFile* file = gtk_file_chooser_get_file(chooser);
        if (file) {
            char* filename = g_file_get_path(file);
            if (filename) {
                gtk_editable_set_text(GTK_EDITABLE(settings->sd_vae_entry_), filename);
                g_free(filename);
            }
            g_object_unref(file);
        }
    }
    g_object_unref(dialog);
}

void SettingsDialog::on_sd_vae_browse_clicked(GtkWidget* widget, gpointer user_data) {
    SettingsDialog* settings = static_cast<SettingsDialog*>(user_data);
    
    GtkFileChooserNative* dialog = gtk_file_chooser_native_new(
        "Select VAE Models Directory",
        GTK_WINDOW(settings->dialog_),
        GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER,
        "Select",
        "Cancel");
    
    g_signal_connect(dialog, "response", G_CALLBACK(on_sd_vae_file_dialog_response), settings);
    
    gtk_native_dialog_show(GTK_NATIVE_DIALOG(dialog));
}

void SettingsDialog::on_sd_controlnet_file_dialog_response(GtkNativeDialog* dialog, gint response, gpointer user_data) {
    SettingsDialog* settings = static_cast<SettingsDialog*>(user_data);
    if (response == GTK_RESPONSE_ACCEPT) {
        GtkFileChooser* chooser = GTK_FILE_CHOOSER(dialog);
        GFile* file = gtk_file_chooser_get_file(chooser);
        if (file) {
            char* filename = g_file_get_path(file);
            if (filename) {
                gtk_editable_set_text(GTK_EDITABLE(settings->sd_controlnet_entry_), filename);
                g_free(filename);
            }
            g_object_unref(file);
        }
    }
    g_object_unref(dialog);
}

void SettingsDialog::on_sd_controlnet_browse_clicked(GtkWidget* widget, gpointer user_data) {
    SettingsDialog* settings = static_cast<SettingsDialog*>(user_data);
    
    GtkFileChooserNative* dialog = gtk_file_chooser_native_new(
        "Select ControlNet Models Directory",
        GTK_WINDOW(settings->dialog_),
        GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER,
        "Select",
        "Cancel");
    
    g_signal_connect(dialog, "response", G_CALLBACK(on_sd_controlnet_file_dialog_response), settings);
    
    gtk_native_dialog_show(GTK_NATIVE_DIALOG(dialog));
}

void SettingsDialog::on_sd_lora_file_dialog_response(GtkNativeDialog* dialog, gint response, gpointer user_data) {
    SettingsDialog* settings = static_cast<SettingsDialog*>(user_data);
    if (response == GTK_RESPONSE_ACCEPT) {
        GtkFileChooser* chooser = GTK_FILE_CHOOSER(dialog);
        GFile* file = gtk_file_chooser_get_file(chooser);
        if (file) {
            char* filename = g_file_get_path(file);
            if (filename) {
                gtk_editable_set_text(GTK_EDITABLE(settings->sd_lora_entry_), filename);
                g_free(filename);
            }
            g_object_unref(file);
        }
    }
    g_object_unref(dialog);
}

void SettingsDialog::on_sd_lora_browse_clicked(GtkWidget* widget, gpointer user_data) {
    SettingsDialog* settings = static_cast<SettingsDialog*>(user_data);
    
    GtkFileChooserNative* dialog = gtk_file_chooser_native_new(
        "Select LoRA Models Directory",
        GTK_WINDOW(settings->dialog_),
        GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER,
        "Select",
        "Cancel");
    
    g_signal_connect(dialog, "response", G_CALLBACK(on_sd_lora_file_dialog_response), settings);
    
    gtk_native_dialog_show(GTK_NATIVE_DIALOG(dialog));
}

void SettingsDialog::on_model_path_dialog_response(GtkNativeDialog* dialog, gint response, gpointer user_data) {
    SettingsDialog* settings = static_cast<SettingsDialog*>(user_data);
    if (response == GTK_RESPONSE_ACCEPT) {
        GtkFileChooser* chooser = GTK_FILE_CHOOSER(dialog);
        GFile* file = gtk_file_chooser_get_file(chooser);
        if (file) {
            char* filename = g_file_get_path(file);
            if (filename) {
                gtk_editable_set_text(GTK_EDITABLE(settings->model_path_entry_), filename);
                g_free(filename);
            }
            g_object_unref(file);
        }
    }
    g_object_unref(dialog);
}

void SettingsDialog::on_model_path_browse_clicked(GtkWidget* widget, gpointer user_data) {
    SettingsDialog* settings = static_cast<SettingsDialog*>(user_data);
    
    GtkFileChooserNative* dialog = gtk_file_chooser_native_new(
        "Select Model Storage Directory",
        GTK_WINDOW(settings->dialog_),
        GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER,
        "Select",
        "Cancel");
    
    g_signal_connect(dialog, "response", G_CALLBACK(on_model_path_dialog_response), settings);
    
    gtk_native_dialog_show(GTK_NATIVE_DIALOG(dialog));
}

void SettingsDialog::on_ollama_path_dialog_response(GtkNativeDialog* dialog, gint response, gpointer user_data) {
    SettingsDialog* settings = static_cast<SettingsDialog*>(user_data);
    if (response == GTK_RESPONSE_ACCEPT) {
        GtkFileChooser* chooser = GTK_FILE_CHOOSER(dialog);
        GFile* file = gtk_file_chooser_get_file(chooser);
        if (file) {
            char* filename = g_file_get_path(file);
            if (filename) {
                gtk_editable_set_text(GTK_EDITABLE(settings->ollama_path_entry_), filename);
                g_free(filename);
            }
            g_object_unref(file);
        }
    }
    g_object_unref(dialog);
}

void SettingsDialog::on_ollama_path_browse_clicked(GtkWidget* widget, gpointer user_data) {
    SettingsDialog* settings = static_cast<SettingsDialog*>(user_data);
    
    GtkFileChooserNative* dialog = gtk_file_chooser_native_new(
        "Select Ollama Model Directory",
        GTK_WINDOW(settings->dialog_),
        GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER,
        "Select",
        "Cancel");
    
    g_signal_connect(dialog, "response", G_CALLBACK(on_ollama_path_dialog_response), settings);
    
    gtk_native_dialog_show(GTK_NATIVE_DIALOG(dialog));
}

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