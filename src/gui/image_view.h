#ifndef DUOROU_GUI_IMAGE_VIEW_H
#define DUOROU_GUI_IMAGE_VIEW_H

#include <gtk/gtk.h>
#include <string>

namespace duorou {
namespace gui {

/**
 * Image view class - handles interactive interface for image generation models
 */
class ImageView {
public:
    ImageView();
    ~ImageView();

    // Disable copy constructor and assignment
    ImageView(const ImageView&) = delete;
    ImageView& operator=(const ImageView&) = delete;

    /**
     * Initialize image view
     * @return Returns true on success, false on failure
     */
    bool initialize();

    /**
     * Get main GTK component
     * @return GTK component pointer
     */
    GtkWidget* get_widget() const { return main_widget_; }

    /**
     * Generate image
     * @param prompt Image generation prompt
     */
    void generate_image(const std::string& prompt);

    /**
     * Display generated image
     * @param image_path Image file path
     */
    void display_image(const std::string& image_path);

    /**
     * Clear current image
     */
    void clear_image();

private:
    GtkWidget* main_widget_;         // Main container
    GtkWidget* prompt_box_;          // Prompt input area
    GtkWidget* prompt_entry_;        // Prompt input box
    GtkWidget* generate_button_;     // Generate button
    GtkWidget* image_scrolled_;      // Image scroll window
    GtkWidget* image_widget_;        // Image display component
    GtkWidget* progress_bar_;        // Progress bar
    GtkWidget* status_label_;        // Status label

    /**
     * Create prompt input area
     */
    void create_prompt_area();

    /**
     * Create image display area
     */
    void create_image_area();

    /**
     * Create status area
     */
    void create_status_area();

    /**
     * Connect signal handlers
     */
    void connect_signals();

    /**
     * Update progress
     * @param progress Progress value (0.0 - 1.0)
     */
    void update_progress(double progress);

    /**
     * Update status text
     * @param status Status text
     */
    void update_status(const std::string& status);

    /**
     * Create placeholder image
     */
    void create_placeholder_image();

    // Static callback functions
    static void on_generate_button_clicked(GtkWidget* widget, gpointer user_data);
    static void on_prompt_entry_activate(GtkWidget* widget, gpointer user_data);
    static void on_clear_button_clicked(GtkWidget* widget, gpointer user_data);
};

} // namespace gui
} // namespace duorou

#endif // DUOROU_GUI_IMAGE_VIEW_H