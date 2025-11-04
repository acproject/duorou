#include "settings_dialog.h"
#include "../core/application.h"
#include "../core/config_manager.h"
#include "../core/logger.h"
#include "../core/model_manager.h"

#include <iostream>
#ifdef HAVE_GTK
#include <gtk/gtk.h>
#endif

namespace duorou {
namespace gui {

#ifdef HAVE_GTK

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
    // GTK4 automatically cleans up child components
}

bool SettingsDialog::initialize() {
    // Create dialog
    dialog_ = gtk_dialog_new();
    if (!dialog_) {
        std::cerr << "Failed to create settings dialog" << std::endl;
        return false;
    }

    // Set dialog properties
    gtk_window_set_title(GTK_WINDOW(dialog_), "Settings");
    gtk_window_set_default_size(GTK_WINDOW(dialog_), 600, 500);
    gtk_window_set_modal(GTK_WINDOW(dialog_), TRUE);
    gtk_window_set_resizable(GTK_WINDOW(dialog_), TRUE);

    // Get dialog content area
    GtkWidget* content_area = gtk_dialog_get_content_area(GTK_DIALOG(dialog_));
    
    // Create notebook for tabs
    notebook_ = gtk_notebook_new();
    gtk_widget_set_vexpand(notebook_, TRUE);
    gtk_widget_set_hexpand(notebook_, TRUE);
    
    // Create settings pages
    create_general_page();
    create_model_page();
    create_performance_page();
    
    // Add to content area
    gtk_box_append(GTK_BOX(content_area), notebook_);
    
    // Add dialog buttons
    gtk_dialog_add_button(GTK_DIALOG(dialog_), "Cancel", GTK_RESPONSE_CANCEL);
    gtk_dialog_add_button(GTK_DIALOG(dialog_), "Apply", GTK_RESPONSE_APPLY);
    gtk_dialog_add_button(GTK_DIALOG(dialog_), "OK", GTK_RESPONSE_OK);
    
    // Connect signals
    connect_signals();
    
    // Load current settings
    load_settings();

    std::cout << "Settings dialog initialized successfully" << std::endl;
    return true;
}

void SettingsDialog::show(GtkWidget* parent) {
    if (dialog_) {
        if (parent) {
            gtk_window_set_transient_for(GTK_WINDOW(dialog_), GTK_WINDOW(parent));
        }
        
        // Ensure dialog is shown on top
        gtk_window_set_modal(GTK_WINDOW(dialog_), TRUE);
        
        // Show dialog and ensure it gets focus
        gtk_widget_show(dialog_);
        gtk_window_present(GTK_WINDOW(dialog_));
        
        // Force focus grab
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
    
    // Application settings group
    GtkWidget* app_group = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_widget_set_margin_bottom(app_group, 20);
    
    // Group title
    GtkWidget* app_title = gtk_label_new("Application Settings");
    gtk_widget_set_halign(app_title, GTK_ALIGN_START);
    gtk_widget_add_css_class(app_title, "settings-group-title");
    gtk_widget_set_margin_bottom(app_title, 10);
    gtk_box_append(GTK_BOX(app_group), app_title);
    
    // Theme selection
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
    
    // Language selection
    GtkWidget* language_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    GtkWidget* language_label = gtk_label_new("Language:");
    gtk_widget_set_size_request(language_label, 100, -1);
    gtk_widget_set_halign(language_label, GTK_ALIGN_START);
    
    language_combo_ = gtk_combo_box_text_new();
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(language_combo_), "English");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(language_combo_), "中文");
    gtk_combo_box_set_active(GTK_COMBO_BOX(language_combo_), 0);
    
    gtk_box_append(GTK_BOX(language_box), language_label);
    gtk_box_append(GTK_BOX(language_box), language_combo_);
    gtk_widget_set_margin_start(language_box, 15);
    gtk_widget_set_margin_bottom(language_box, 5);
    gtk_box_append(GTK_BOX(app_group), language_box);
    
    // Minimize to system tray on startup
    startup_check_ = gtk_check_button_new_with_label("Minimize to system tray on startup");
    gtk_widget_set_margin_start(startup_check_, 15);
    gtk_widget_set_margin_bottom(startup_check_, 5);
    gtk_box_append(GTK_BOX(app_group), startup_check_);
    
    gtk_box_append(GTK_BOX(general_page_), app_group);
    
    // Add to notebook
    gtk_notebook_append_page(GTK_NOTEBOOK(notebook_), general_page_, gtk_label_new("General"));
}

void SettingsDialog::create_model_page() {
    model_page_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_widget_set_margin_start(model_page_, 20);
    gtk_widget_set_margin_end(model_page_, 20);
    gtk_widget_set_margin_top(model_page_, 20);
    gtk_widget_set_margin_bottom(model_page_, 20);
    
    // Model settings group
    GtkWidget* model_group = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_widget_set_margin_bottom(model_group, 20);
    
    // Group title
    GtkWidget* model_title = gtk_label_new("Model Settings");
    gtk_widget_set_halign(model_title, GTK_ALIGN_START);
    gtk_widget_add_css_class(model_title, "settings-group-title");
    gtk_widget_set_margin_bottom(model_title, 10);
    gtk_box_append(GTK_BOX(model_group), model_title);
    
    // LLaMA model selection dropdown
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
    
    // Add Force LLaMA backend checkbox
    force_llama_check_ = gtk_check_button_new_with_label("Force LLaMA backend (use llama.cpp for text generation)");
    gtk_widget_set_margin_start(force_llama_check_, 15);
    gtk_widget_set_margin_bottom(force_llama_check_, 5);
    gtk_box_append(GTK_BOX(model_group), force_llama_check_);
    
    // Llama.cpp model path
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
    
    // Ollama model path
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
    
    // Remove Custom Path configuration item, as Model Path already serves as Llama.cpp model storage location
    
    // Stable Diffusion model configuration group
    GtkWidget* sd_group = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_widget_set_margin_top(sd_group, 10);
    gtk_widget_set_margin_bottom(sd_group, 10);
    
    // SD group title
    GtkWidget* sd_title = gtk_label_new("Stable Diffusion Models");
    gtk_widget_set_halign(sd_title, GTK_ALIGN_START);
    gtk_widget_add_css_class(sd_title, "settings-subsection-title");
    gtk_widget_set_margin_bottom(sd_title, 5);
    gtk_box_append(GTK_BOX(sd_group), sd_title);
    
    // Main model path
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
    
    // VAE model path
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
    
    // ControlNet model path
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
    
    // LoRA model path
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
    
    // Add to notebook
    gtk_notebook_append_page(GTK_NOTEBOOK(notebook_), model_page_, gtk_label_new("Models"));
    
    // Refresh model list
    refresh_model_list();
}

void SettingsDialog::create_performance_page() {
    performance_page_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_widget_set_margin_start(performance_page_, 20);
    gtk_widget_set_margin_end(performance_page_, 20);
    gtk_widget_set_margin_top(performance_page_, 20);
    gtk_widget_set_margin_bottom(performance_page_, 20);
    
    // Performance settings group
    GtkWidget* perf_group = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_widget_set_margin_bottom(perf_group, 20);
    
    // Group title
    GtkWidget* perf_title = gtk_label_new("Performance Settings");
    gtk_widget_set_halign(perf_title, GTK_ALIGN_START);
    gtk_widget_add_css_class(perf_title, "settings-group-title");
    gtk_widget_set_margin_bottom(perf_title, 10);
    gtk_box_append(GTK_BOX(perf_group), perf_title);
    
    // Thread count setting
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
    
    // GPU acceleration
    gpu_check_ = gtk_check_button_new_with_label("Enable GPU acceleration (if available)");
    gtk_widget_set_margin_start(gpu_check_, 15);
    gtk_widget_set_margin_bottom(gpu_check_, 5);
    gtk_box_append(GTK_BOX(perf_group), gpu_check_);
    
    // Memory limit
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
    
    // Add to notebook
    gtk_notebook_append_page(GTK_NOTEBOOK(notebook_), performance_page_, gtk_label_new("Performance"));
}

void SettingsDialog::connect_signals() {
    // Connect dialog response signal
    g_signal_connect(dialog_, "response", G_CALLBACK(on_dialog_response), this);
}

void SettingsDialog::load_settings() {
    if (!application_) {
        std::cout << "Warning: Application instance not available, using default values" << std::endl;
        // Set default values
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
        // Set default values
        gtk_combo_box_set_active(GTK_COMBO_BOX(theme_combo_), 0);
        gtk_combo_box_set_active(GTK_COMBO_BOX(language_combo_), 0);
        gtk_check_button_set_active(GTK_CHECK_BUTTON(startup_check_), FALSE);
        gtk_check_button_set_active(GTK_CHECK_BUTTON(gpu_check_), FALSE);
        gtk_spin_button_set_value(GTK_SPIN_BUTTON(threads_spin_), 4);
        gtk_spin_button_set_value(GTK_SPIN_BUTTON(memory_spin_), 8);
        return;
    }
    
    std::cout << "Loading settings from configuration file..." << std::endl;
    
    // Load settings from configuration file
    int theme_index = config_manager->getInt("ui.theme", 0);
    int language_index = config_manager->getInt("ui.language", 0);
    bool startup_minimize = config_manager->getBool("ui.startup_minimize", false);
    bool gpu_enabled = config_manager->getBool("performance.gpu_enabled", false);
    int threads = config_manager->getInt("performance.threads", 4);
    int memory_limit = config_manager->getInt("performance.memory_limit", 8);
    bool force_llama = config_manager->getBool("model.force_llama", false);
    if (force_llama_check_) {
        gtk_check_button_set_active(GTK_CHECK_BUTTON(force_llama_check_), force_llama);
    }
    
    std::string llama_selected = config_manager->getString("models.llama_selected", "");
    std::string sd_path = config_manager->getString("models.sd_path", "");
    std::string sd_vae_path = config_manager->getString("models.sd_vae_path", "");
    std::string sd_controlnet_path = config_manager->getString("models.sd_controlnet_path", "");
    std::string sd_lora_path = config_manager->getString("models.sd_lora_path", "");
    std::string model_path = config_manager->getString("models.model_path", "");
    std::string ollama_path = config_manager->getString("models.ollama_path", "");
    
    // Set UI control values
    gtk_combo_box_set_active(GTK_COMBO_BOX(theme_combo_), theme_index);
    gtk_combo_box_set_active(GTK_COMBO_BOX(language_combo_), language_index);
    gtk_check_button_set_active(GTK_CHECK_BUTTON(startup_check_), startup_minimize);
    gtk_check_button_set_active(GTK_CHECK_BUTTON(gpu_check_), gpu_enabled);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(threads_spin_), threads);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(memory_spin_), memory_limit);
    
    // Set model paths
    gtk_entry_buffer_set_text(gtk_entry_get_buffer(GTK_ENTRY(sd_model_entry_)), sd_path.c_str(), sd_path.length());
    gtk_entry_buffer_set_text(gtk_entry_get_buffer(GTK_ENTRY(sd_vae_entry_)), sd_vae_path.c_str(), sd_vae_path.length());
    gtk_entry_buffer_set_text(gtk_entry_get_buffer(GTK_ENTRY(sd_controlnet_entry_)), sd_controlnet_path.c_str(), sd_controlnet_path.length());
    gtk_entry_buffer_set_text(gtk_entry_get_buffer(GTK_ENTRY(sd_lora_entry_)), sd_lora_path.c_str(), sd_lora_path.length());
    gtk_entry_buffer_set_text(gtk_entry_get_buffer(GTK_ENTRY(model_path_entry_)), model_path.c_str(), model_path.length());
    gtk_entry_buffer_set_text(gtk_entry_get_buffer(GTK_ENTRY(ollama_path_entry_)), ollama_path.c_str(), ollama_path.length());

    // 应用路径到 ModelManager 并刷新模型列表
    auto model_manager = application_->getModelManager();
    if (model_manager) {
        if (!ollama_path.empty()) {
            model_manager->setOllamaModelsPath(ollama_path);
        }
        if (!model_path.empty()) {
            model_manager->rescanModelDirectory(model_path);
        }
        refresh_model_list();
    }
    
    // If there's a saved model selection, try to set it in the dropdown
    if (!llama_selected.empty() && llama_model_combo_) {
        // Find matching model option
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
    
    // Get current setting values
    int theme_index = gtk_combo_box_get_active(GTK_COMBO_BOX(theme_combo_));
    int language_index = gtk_combo_box_get_active(GTK_COMBO_BOX(language_combo_));
    gboolean startup_minimize = gtk_check_button_get_active(GTK_CHECK_BUTTON(startup_check_));
    gboolean gpu_enabled = gtk_check_button_get_active(GTK_CHECK_BUTTON(gpu_check_));
    int threads = gtk_spin_button_get_value_as_int(GTK_SPIN_BUTTON(threads_spin_));
    int memory_limit = gtk_spin_button_get_value_as_int(GTK_SPIN_BUTTON(memory_spin_));
    
    // Get selected LLaMA model
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
    
    // Save settings to configuration file
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
    
    // Save configuration to file
    if (config_manager->saveConfig()) {
        std::cout << "Settings saved successfully" << std::endl;
    } else {
        std::cout << "Error: Failed to save settings" << std::endl;
    }

    if (selected_model) {
        g_free(selected_model);
    }

    // 保存后立即应用路径到 ModelManager 并刷新列表
    auto model_manager = application_->getModelManager();
    if (model_manager) {
        if (ollama_path && strlen(ollama_path) > 0) {
            model_manager->setOllamaModelsPath(ollama_path);
        }
        if (model_path && strlen(model_path) > 0) {
            model_manager->rescanModelDirectory(model_path);
        }
        refresh_model_list();
    }
}

void SettingsDialog::reset_to_defaults() {
    // TODO: Reset all settings to default values
    std::cout << "Resetting settings to default values..." << std::endl;
    
    // Reset UI controls to default values
    gtk_combo_box_set_active(GTK_COMBO_BOX(theme_combo_), 0);
    gtk_combo_box_set_active(GTK_COMBO_BOX(language_combo_), 0);
    gtk_check_button_set_active(GTK_CHECK_BUTTON(startup_check_), FALSE);
    gtk_check_button_set_active(GTK_CHECK_BUTTON(gpu_check_), FALSE);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(threads_spin_), 4);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(memory_spin_), 8);
    
    // Reset model selection
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

// Static callback function implementations
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

// Dialog response handling
void SettingsDialog::set_application(core::Application* app) {
    application_ = app;
    refresh_model_list();
}

void SettingsDialog::refresh_model_list() {
    if (!llama_model_combo_ || !application_) {
        return;
    }
    
    // Clear existing options
    gtk_combo_box_text_remove_all(GTK_COMBO_BOX_TEXT(llama_model_combo_));
    
    // Get real available model list from ModelManager
    auto model_manager = application_->getModelManager();
    if (model_manager) {
        std::vector<duorou::core::ModelManagerInfo> models = model_manager->getAllModels();
        
        // 筛选语言模型
        std::vector<std::string> language_model_names;
        for (const auto &info : models) {
            if (info.type == duorou::core::ModelType::LANGUAGE_MODEL) {
                // 优先展示 name，其次 id
                if (!info.name.empty()) {
                    language_model_names.push_back(info.name);
                } else if (!info.id.empty()) {
                    language_model_names.push_back(info.id);
                }
            }
        }

        if (language_model_names.empty()) {
            // If no local models, add hint information
            gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(llama_model_combo_), "No models found");
        } else {
            // Add all local models
            for (const auto& name : language_model_names) {
                gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(llama_model_combo_), name.c_str());
            }
        }
    } else {
        // If ModelManager is unavailable, show error message
        gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(llama_model_combo_), "Model manager unavailable");
    }
    
    // Set default selection
    gtk_combo_box_set_active(GTK_COMBO_BOX(llama_model_combo_), 0);
}

// Llama models are now selected through dropdown, no longer need file dialog

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
                // 立即触发重新扫描并刷新列表
                if (settings->application_) {
                    auto mm = settings->application_->getModelManager();
                    if (mm) {
                        mm->rescanModelDirectory(filename);
                        settings->refresh_model_list();
                    }
                }
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
                // 立即应用 Ollama 路径并刷新列表
                if (settings->application_) {
                    auto mm = settings->application_->getModelManager();
                    if (mm) {
                        mm->setOllamaModelsPath(filename);
                        settings->refresh_model_list();
                    }
                }
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

#else // HAVE_GTK

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

SettingsDialog::~SettingsDialog() {}

bool SettingsDialog::initialize() { return false; }
void SettingsDialog::show(GtkWidget* /*parent*/) {}
void SettingsDialog::hide() {}
void SettingsDialog::create_general_page() {}
void SettingsDialog::create_model_page() {}
void SettingsDialog::create_performance_page() {}
void SettingsDialog::connect_signals() {}
void SettingsDialog::load_settings() {}
void SettingsDialog::save_settings() {}
void SettingsDialog::reset_to_defaults() {}
void SettingsDialog::on_ok_button_clicked(GtkWidget* /*widget*/, gpointer /*user_data*/) {}
void SettingsDialog::on_cancel_button_clicked(GtkWidget* /*widget*/, gpointer /*user_data*/) {}
void SettingsDialog::on_apply_button_clicked(GtkWidget* /*widget*/, gpointer /*user_data*/) {}
void SettingsDialog::on_reset_button_clicked(GtkWidget* /*widget*/, gpointer /*user_data*/) {}
void SettingsDialog::set_application(core::Application* app) { application_ = app; }
void SettingsDialog::refresh_model_list() {}
void SettingsDialog::on_sd_file_dialog_response(GtkNativeDialog* /*dialog*/, gint /*response*/, gpointer /*user_data*/) {}
void SettingsDialog::on_sd_browse_clicked(GtkWidget* /*widget*/, gpointer /*user_data*/) {}
void SettingsDialog::on_sd_vae_file_dialog_response(GtkNativeDialog* /*dialog*/, gint /*response*/, gpointer /*user_data*/) {}
void SettingsDialog::on_sd_vae_browse_clicked(GtkWidget* /*widget*/, gpointer /*user_data*/) {}
void SettingsDialog::on_sd_controlnet_file_dialog_response(GtkNativeDialog* /*dialog*/, gint /*response*/, gpointer /*user_data*/) {}
void SettingsDialog::on_sd_controlnet_browse_clicked(GtkWidget* /*widget*/, gpointer /*user_data*/) {}
void SettingsDialog::on_sd_lora_file_dialog_response(GtkNativeDialog* /*dialog*/, gint /*response*/, gpointer /*user_data*/) {}
void SettingsDialog::on_sd_lora_browse_clicked(GtkWidget* /*widget*/, gpointer /*user_data*/) {}
void SettingsDialog::on_model_path_dialog_response(GtkNativeDialog* /*dialog*/, gint /*response*/, gpointer /*user_data*/) {}
void SettingsDialog::on_model_path_browse_clicked(GtkWidget* /*widget*/, gpointer /*user_data*/) {}
void SettingsDialog::on_ollama_path_dialog_response(GtkNativeDialog* /*dialog*/, gint /*response*/, gpointer /*user_data*/) {}
void SettingsDialog::on_ollama_path_browse_clicked(GtkWidget* /*widget*/, gpointer /*user_data*/) {}
void SettingsDialog::on_dialog_response(GtkDialog* /*dialog*/, gint /*response_id*/, gpointer /*user_data*/) {}

#endif // HAVE_GTK

} // namespace gui
} // namespace duorou