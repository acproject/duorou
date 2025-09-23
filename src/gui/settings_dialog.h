#ifndef DUOROU_GUI_SETTINGS_DIALOG_H
#define DUOROU_GUI_SETTINGS_DIALOG_H

#include <gtk/gtk.h>
#include <string>

namespace duorou {
namespace core {
class Application;
}
namespace gui {

/**
 * Settings dialog class - manages application configuration
 */
class SettingsDialog {
public:
    SettingsDialog();
    explicit SettingsDialog(core::Application* app);
    ~SettingsDialog();

    // Disable copy constructor and assignment
    SettingsDialog(const SettingsDialog&) = delete;
    SettingsDialog& operator=(const SettingsDialog&) = delete;

    /**
     * Initialize settings dialog
     * @return true on success, false on failure
     */
    bool initialize();

    /**
     * Show settings dialog
     * @param parent Parent window
     */
    void show(GtkWidget* parent);

    /**
     * Hide settings dialog
     */
    void hide();

private:
    GtkWidget* dialog_;              // Dialog
    GtkWidget* notebook_;            // Tab container
    
    // General settings page
    GtkWidget* general_page_;
    GtkWidget* theme_combo_;
    GtkWidget* language_combo_;
    GtkWidget* startup_check_;
    
    // Model settings page
    GtkWidget* model_page_;
    GtkWidget* llama_model_combo_;
    GtkWidget* sd_model_entry_;        // Main SD model
    GtkWidget* sd_vae_entry_;          // VAE model
    GtkWidget* sd_controlnet_entry_;   // ControlNet model
    GtkWidget* sd_lora_entry_;         // LoRA model directory
    GtkWidget* model_path_entry_;
    GtkWidget* ollama_path_entry_;
    
    // Performance settings page
    GtkWidget* performance_page_;
    GtkWidget* threads_spin_;
    GtkWidget* gpu_check_;
    GtkWidget* memory_spin_;

    /**
     * Create general settings page
     */
    void create_general_page();

    /**
     * Create model settings page
     */
    void create_model_page();

    /**
     * Create performance settings page
     */
    void create_performance_page();

    /**
     * Connect signal handlers
     */
    void connect_signals();

    /**
     * Load current settings
     */
    void load_settings();

    /**
     * Save settings
     */
    void save_settings();

    /**
     * Reset to default settings
     */
    void reset_to_defaults();

    /**
     * Set application instance reference
     * @param app Application instance pointer
     */
    void set_application(core::Application* app);

    /**
     * Refresh model list
     */
    void refresh_model_list();

    // Application instance reference
    core::Application* application_;

    // Static callback functions
    static void on_ok_button_clicked(GtkWidget* widget, gpointer user_data);
    static void on_cancel_button_clicked(GtkWidget* widget, gpointer user_data);
    static void on_apply_button_clicked(GtkWidget* widget, gpointer user_data);
    static void on_reset_button_clicked(GtkWidget* widget, gpointer user_data);
    static void on_sd_browse_clicked(GtkWidget* widget, gpointer user_data);
    static void on_sd_vae_browse_clicked(GtkWidget* widget, gpointer user_data);
    static void on_sd_controlnet_browse_clicked(GtkWidget* widget, gpointer user_data);
    static void on_sd_lora_browse_clicked(GtkWidget* widget, gpointer user_data);
    static void on_model_path_browse_clicked(GtkWidget* widget, gpointer user_data);
    static void on_ollama_path_browse_clicked(GtkWidget* widget, gpointer user_data);
    static void on_dialog_response(GtkDialog* dialog, gint response_id, gpointer user_data);
    
    // File selection dialog callback functions
    static void on_sd_file_dialog_response(GtkNativeDialog* dialog, gint response, gpointer user_data);
    static void on_sd_vae_file_dialog_response(GtkNativeDialog* dialog, gint response, gpointer user_data);
    static void on_sd_controlnet_file_dialog_response(GtkNativeDialog* dialog, gint response, gpointer user_data);
    static void on_sd_lora_file_dialog_response(GtkNativeDialog* dialog, gint response, gpointer user_data);
    static void on_model_path_dialog_response(GtkNativeDialog* dialog, gint response, gpointer user_data);
    static void on_ollama_path_dialog_response(GtkNativeDialog* dialog, gint response, gpointer user_data);
};

} // namespace gui
} // namespace duorou

#endif // DUOROU_GUI_SETTINGS_DIALOG_H