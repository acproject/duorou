#include "video_source_dialog.h"
#include "../core/logger.h"
#include <iostream>

namespace duorou {
namespace gui {

VideoSourceDialog::VideoSourceDialog()
    : dialog_(nullptr)
    , content_box_(nullptr)
    , title_label_(nullptr)
    , desktop_button_(nullptr)
    , camera_button_(nullptr)
    , cancel_button_(nullptr)
    , button_box_(nullptr)
{
}

VideoSourceDialog::~VideoSourceDialog() {
    if (dialog_) {
        gtk_window_destroy(GTK_WINDOW(dialog_));
        dialog_ = nullptr;
    }
}

bool VideoSourceDialog::initialize() {
    // Create dialog
    dialog_ = gtk_window_new();
    if (!dialog_) {
        std::cerr << "Failed to create video source dialog" << std::endl;
        return false;
    }

    // Set dialog properties
    gtk_window_set_title(GTK_WINDOW(dialog_), "Select Video Source");
    gtk_window_set_default_size(GTK_WINDOW(dialog_), 400, 200);
    gtk_window_set_resizable(GTK_WINDOW(dialog_), FALSE);
    gtk_window_set_modal(GTK_WINDOW(dialog_), TRUE);

    // Create content
    create_content();

    // Setup styling
    setup_styling();

    // Connect signals
    connect_signals();

    std::cout << "Video source dialog initialized successfully" << std::endl;
    return true;
}

void VideoSourceDialog::show(GtkWidget* parent_window, std::function<void(VideoSource)> callback) {
    if (!dialog_) {
        std::cerr << "Dialog not initialized" << std::endl;
        return;
    }

    // Save callback function
    selection_callback_ = callback;

    // Set parent window
    if (parent_window) {
        GtkWindow* parent = GTK_WINDOW(gtk_widget_get_root(parent_window));
        if (parent) {
            gtk_window_set_transient_for(GTK_WINDOW(dialog_), parent);
        }
    }

    // Show dialog
    gtk_widget_show(dialog_);
    gtk_window_present(GTK_WINDOW(dialog_));
}

void VideoSourceDialog::hide() {
    if (dialog_) {
        gtk_widget_hide(dialog_);
    }
}

void VideoSourceDialog::create_content() {
    // Create main container
    content_box_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 20);
    gtk_widget_set_margin_top(content_box_, 20);
    gtk_widget_set_margin_bottom(content_box_, 20);
    gtk_widget_set_margin_start(content_box_, 20);
    gtk_widget_set_margin_end(content_box_, 20);
    gtk_window_set_child(GTK_WINDOW(dialog_), content_box_);

    // Create title label
    title_label_ = gtk_label_new("Please select video source:");
    gtk_widget_add_css_class(title_label_, "title");
    gtk_box_append(GTK_BOX(content_box_), title_label_);

    // Create button container
    button_box_ = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    gtk_widget_set_halign(button_box_, GTK_ALIGN_CENTER);
    gtk_box_append(GTK_BOX(content_box_), button_box_);

    // Create desktop recording button
    desktop_button_ = gtk_button_new_with_label("Desktop Recording");
    gtk_widget_set_size_request(desktop_button_, 120, 50);
    gtk_widget_add_css_class(desktop_button_, "suggested-action");
    gtk_box_append(GTK_BOX(button_box_), desktop_button_);

    // Create camera button
    camera_button_ = gtk_button_new_with_label("Camera");
    gtk_widget_set_size_request(camera_button_, 120, 50);
    gtk_widget_add_css_class(camera_button_, "suggested-action");
    gtk_box_append(GTK_BOX(button_box_), camera_button_);

    // Create cancel button container
    GtkWidget* cancel_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    gtk_widget_set_halign(cancel_box, GTK_ALIGN_CENTER);
    gtk_box_append(GTK_BOX(content_box_), cancel_box);

    // Create cancel button
    cancel_button_ = gtk_button_new_with_label("Cancel");
    gtk_widget_set_size_request(cancel_button_, 80, 35);
    gtk_box_append(GTK_BOX(cancel_box), cancel_button_);
}

void VideoSourceDialog::setup_styling() {
    // Add CSS styles
    const char* css_data = 
        ".title { "
        "  font-size: 16px; "
        "  font-weight: bold; "
        "  margin-bottom: 10px; "
        "} "
        "button { "
        "  font-size: 14px; "
        "  padding: 8px 16px; "
        "  border-radius: 6px; "
        "} "
        "button.suggested-action { "
        "  background: linear-gradient(to bottom, #4a90e2, #357abd); "
        "  color: white; "
        "  border: 1px solid #2968a3; "
        "} "
        "button.suggested-action:hover { "
        "  background: linear-gradient(to bottom, #5ba0f2, #4a90e2); "
        "} ";

    GtkCssProvider* css_provider = gtk_css_provider_new();
    gtk_css_provider_load_from_data(css_provider, css_data, -1);
    
    GtkStyleContext* style_context = gtk_widget_get_style_context(dialog_);
    gtk_style_context_add_provider(style_context, 
                                   GTK_STYLE_PROVIDER(css_provider), 
                                   GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
    
    g_object_unref(css_provider);
}

void VideoSourceDialog::connect_signals() {
    // Connect button click signals
    g_signal_connect(desktop_button_, "clicked", 
                     G_CALLBACK(on_desktop_button_clicked), this);
    g_signal_connect(camera_button_, "clicked", 
                     G_CALLBACK(on_camera_button_clicked), this);
    g_signal_connect(cancel_button_, "clicked", 
                     G_CALLBACK(on_cancel_button_clicked), this);
    
    // Connect window close signal
    g_signal_connect(dialog_, "close-request", 
                     G_CALLBACK(on_dialog_delete_event), this);
}

void VideoSourceDialog::handle_selection(VideoSource source) {
    // Hide dialog
    hide();
    
    // Call callback function
    if (selection_callback_) {
        selection_callback_(source);
    }
}

// Static callback function implementations
void VideoSourceDialog::on_desktop_button_clicked(GtkWidget* widget, gpointer user_data) {
    VideoSourceDialog* dialog = static_cast<VideoSourceDialog*>(user_data);
    dialog->handle_selection(VideoSource::DESKTOP_CAPTURE);
}

void VideoSourceDialog::on_camera_button_clicked(GtkWidget* widget, gpointer user_data) {
    VideoSourceDialog* dialog = static_cast<VideoSourceDialog*>(user_data);
    dialog->handle_selection(VideoSource::CAMERA);
}

void VideoSourceDialog::on_cancel_button_clicked(GtkWidget* widget, gpointer user_data) {
    VideoSourceDialog* dialog = static_cast<VideoSourceDialog*>(user_data);
    dialog->handle_selection(VideoSource::CANCEL);
}

gboolean VideoSourceDialog::on_dialog_delete_event(GtkWindow* window, gpointer user_data) {
    VideoSourceDialog* dialog = static_cast<VideoSourceDialog*>(user_data);
    dialog->handle_selection(VideoSource::CANCEL);
    return TRUE; // Prevent default close behavior, handled by handle_selection
}

} // namespace gui
} // namespace duorou