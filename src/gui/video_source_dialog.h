#ifndef DUOROU_GUI_VIDEO_SOURCE_DIALOG_H
#define DUOROU_GUI_VIDEO_SOURCE_DIALOG_H

#include <gtk/gtk.h>
#include <functional>

namespace duorou {
namespace gui {

/**
 * Video source selection dialog
 * Let users choose between desktop recording or camera
 */
class VideoSourceDialog {
public:
    enum class VideoSource {
        DESKTOP_CAPTURE,    // Desktop capture
        CAMERA,             // Camera
        CANCEL              // Cancel
    };

    VideoSourceDialog();
    ~VideoSourceDialog();

    // Disable copy constructor and assignment
    VideoSourceDialog(const VideoSourceDialog&) = delete;
    VideoSourceDialog& operator=(const VideoSourceDialog&) = delete;

    /**
     * Initialize dialog
     * @return true on success, false on failure
     */
    bool initialize();

    /**
     * Show dialog
     * @param parent_window Parent window
     * @param callback Callback function after user selection
     */
    void show(GtkWidget* parent_window, std::function<void(VideoSource)> callback);

    /**
     * Hide dialog
     */
    void hide();

private:
    // GTK components
    GtkWidget* dialog_;              // Dialog
    GtkWidget* content_box_;         // Content container
    GtkWidget* title_label_;         // Title label
    GtkWidget* desktop_button_;      // Desktop recording button
    GtkWidget* camera_button_;       // Camera button
    GtkWidget* cancel_button_;       // Cancel button
    GtkWidget* button_box_;          // Button container

    // Callback function
    std::function<void(VideoSource)> selection_callback_;

    /**
     * Create dialog content
     */
    void create_content();

    /**
     * Setup dialog styling
     */
    void setup_styling();

    /**
     * Connect signal handlers
     */
    void connect_signals();

    /**
     * Handle user selection
     * @param source Selected video source
     */
    void handle_selection(VideoSource source);

    // Static callback functions
    static void on_desktop_button_clicked(GtkWidget* widget, gpointer user_data);
    static void on_camera_button_clicked(GtkWidget* widget, gpointer user_data);
    static void on_cancel_button_clicked(GtkWidget* widget, gpointer user_data);
    static gboolean on_dialog_delete_event(GtkWindow* window, gpointer user_data);
};

} // namespace gui
} // namespace duorou

#endif // DUOROU_GUI_VIDEO_SOURCE_DIALOG_H