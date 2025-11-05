#include "image_view.h"
#include "../core/logger.h"

#include <iostream>

namespace duorou {
namespace gui {

ImageView::ImageView()
    : main_widget_(nullptr)
    , prompt_box_(nullptr)
    , prompt_entry_(nullptr)
    , generate_button_(nullptr)
    , image_scrolled_(nullptr)
    , image_widget_(nullptr)
    , progress_bar_(nullptr)
    , status_label_(nullptr)
{
}

ImageView::~ImageView() {
    // GTK4 automatically cleans up child components
}

bool ImageView::initialize() {
    // Create main container
    main_widget_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    if (!main_widget_) {
        std::cerr << "Failed to create image view main container" << std::endl;
        return false;
    }

    gtk_widget_set_margin_start(main_widget_, 10);
    gtk_widget_set_margin_end(main_widget_, 10);
    gtk_widget_set_margin_top(main_widget_, 10);
    gtk_widget_set_margin_bottom(main_widget_, 10);

    // Create various areas
    create_prompt_area();
    create_image_area();
    create_status_area();
    
    // Connect signals
    connect_signals();

    std::cout << "Image view initialized successfully" << std::endl;
    return true;
}

void ImageView::generate_image(const std::string& prompt) {
    if (prompt.empty()) {
        update_status("Please enter a prompt");
        return;
    }

    // Show progress
    gtk_widget_show(progress_bar_);
    update_progress(0.0);
    update_status("Generating image...");
    
    // Disable generate button
    gtk_widget_set_sensitive(generate_button_, FALSE);
    
    // TODO: Call stable-diffusion model to generate image here
    // Create a placeholder image for now
    
    // Simulate progress updates
    update_progress(0.3);
    update_status("Processing prompt...");
    
    update_progress(0.6);
    update_status("Generating pixels...");
    
    update_progress(1.0);
    update_status("Image generated successfully (placeholder)");
    
    // Display placeholder image
    create_placeholder_image();
    
    // Hide progress bar and re-enable button
    gtk_widget_hide(progress_bar_);
    gtk_widget_set_sensitive(generate_button_, TRUE);
}

void ImageView::display_image(const std::string& image_path) {
    if (!image_widget_ || image_path.empty()) {
        return;
    }

    // Try to load image
    GError* error = nullptr;
    GdkPixbuf* pixbuf = gdk_pixbuf_new_from_file(image_path.c_str(), &error);
    
    if (error) {
        std::cerr << "Failed to load image: " << error->message << std::endl;
        g_error_free(error);
        update_status("Failed to load image: " + image_path);
        return;
    }
    
    if (pixbuf) {
        // Display image
        gtk_image_set_from_pixbuf(GTK_IMAGE(image_widget_), pixbuf);
        g_object_unref(pixbuf);
        update_status("Image loaded: " + image_path);
    }
}

void ImageView::clear_image() {
    if (image_widget_) {
        gtk_image_clear(GTK_IMAGE(image_widget_));
        update_status("Image cleared");
    }
}

void ImageView::create_prompt_area() {
    // Create prompt input area container
    prompt_box_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_widget_set_margin_bottom(prompt_box_, 10);
    
    // Create label
    GtkWidget* prompt_label = gtk_label_new("Image Generation Prompt:");
    gtk_widget_set_halign(prompt_label, GTK_ALIGN_START);
    gtk_widget_add_css_class(prompt_label, "prompt-label");
    
    // Create input box
    prompt_entry_ = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(prompt_entry_), "Describe the image you want to generate...");
    gtk_widget_set_hexpand(prompt_entry_, TRUE);
    
    // Create button container
    GtkWidget* button_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    gtk_widget_set_halign(button_box, GTK_ALIGN_CENTER);
    gtk_widget_set_margin_top(button_box, 5);
    
    // Create generate button
    generate_button_ = gtk_button_new_with_label("Generate Image");
    gtk_widget_set_size_request(generate_button_, 150, 40);
    gtk_widget_add_css_class(generate_button_, "generate-button");
    
    // Create clear button
    GtkWidget* clear_button = gtk_button_new_with_label("Clear");
    gtk_widget_set_size_request(clear_button, 100, 40);
    
    // Assemble components
    gtk_box_append(GTK_BOX(button_box), generate_button_);
    gtk_box_append(GTK_BOX(button_box), clear_button);
    
    gtk_box_append(GTK_BOX(prompt_box_), prompt_label);
    gtk_box_append(GTK_BOX(prompt_box_), prompt_entry_);
    gtk_box_append(GTK_BOX(prompt_box_), button_box);
    
    // Connect clear button signal
    g_signal_connect(clear_button, "clicked", G_CALLBACK(on_clear_button_clicked), this);
    
    // Add to main container
    gtk_box_append(GTK_BOX(main_widget_), prompt_box_);
}

void ImageView::create_image_area() {
    // Create image display component
    image_widget_ = gtk_image_new();
    gtk_widget_set_size_request(image_widget_, 512, 512);
    gtk_widget_set_halign(image_widget_, GTK_ALIGN_CENTER);
    gtk_widget_set_valign(image_widget_, GTK_ALIGN_CENTER);
    
    // Create scroll window
    image_scrolled_ = gtk_scrolled_window_new();
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(image_scrolled_), 
                                   GTK_POLICY_AUTOMATIC, GTK_POLICY_AUTOMATIC);
    gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(image_scrolled_), image_widget_);
    
    // Set scroll window size
    gtk_widget_set_vexpand(image_scrolled_, TRUE);
    gtk_widget_set_hexpand(image_scrolled_, TRUE);
    
    // Add border style
    gtk_widget_add_css_class(image_scrolled_, "image-display");
    
    // Add to main container
    gtk_box_append(GTK_BOX(main_widget_), image_scrolled_);
}

void ImageView::create_status_area() {
    // Create status area container
    GtkWidget* status_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_widget_set_margin_top(status_box, 10);
    
    // Create progress bar
    progress_bar_ = gtk_progress_bar_new();
    gtk_widget_hide(progress_bar_); // Initially hidden
    gtk_widget_set_margin_bottom(progress_bar_, 5);
    
    // Create status label
    status_label_ = gtk_label_new("Ready to generate images");
    gtk_widget_set_halign(status_label_, GTK_ALIGN_START);
    gtk_widget_add_css_class(status_label_, "status-label");
    
    // Assemble components
    gtk_box_append(GTK_BOX(status_box), progress_bar_);
    gtk_box_append(GTK_BOX(status_box), status_label_);
    
    // Add to main container
    gtk_box_append(GTK_BOX(main_widget_), status_box);
}

void ImageView::connect_signals() {
    // Connect generate button signal
    g_signal_connect(generate_button_, "clicked", G_CALLBACK(on_generate_button_clicked), this);
    
    // Connect Enter key to generate image
    g_signal_connect(prompt_entry_, "activate", G_CALLBACK(on_prompt_entry_activate), this);
}

void ImageView::update_progress(double progress) {
    if (progress_bar_) {
        gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(progress_bar_), progress);
    }
}

void ImageView::update_status(const std::string& status) {
    if (status_label_) {
        gtk_label_set_text(GTK_LABEL(status_label_), status.c_str());
    }
}

void ImageView::create_placeholder_image() {
    // Create a simple placeholder image
    GdkPixbuf* pixbuf = gdk_pixbuf_new(GDK_COLORSPACE_RGB, FALSE, 8, 512, 512);
    if (pixbuf) {
        // Fill with light gray
        gdk_pixbuf_fill(pixbuf, 0xC0C0C0FF);
        
        // Display image
        gtk_image_set_from_pixbuf(GTK_IMAGE(image_widget_), pixbuf);
        g_object_unref(pixbuf);
    }
}

// Static callback function implementations
void ImageView::on_generate_button_clicked(GtkWidget* widget, gpointer user_data) {
    ImageView* image_view = static_cast<ImageView*>(user_data);
    
    // Get prompt - create copy immediately to avoid pointer invalidation
    GtkEntryBuffer* buffer = gtk_entry_get_buffer(GTK_ENTRY(image_view->prompt_entry_));
    const char* prompt_text_ptr = gtk_entry_buffer_get_text(buffer);
    std::string prompt_copy = prompt_text_ptr ? std::string(prompt_text_ptr) : "";
    
    image_view->generate_image(prompt_copy);
}

void ImageView::on_prompt_entry_activate(GtkWidget* widget, gpointer user_data) {
    ImageView* image_view = static_cast<ImageView*>(user_data);
    
    // Get prompt - create copy immediately to avoid pointer invalidation
    GtkEntryBuffer* buffer = gtk_entry_get_buffer(GTK_ENTRY(widget));
    const char* prompt_text_ptr = gtk_entry_buffer_get_text(buffer);
    std::string prompt_copy = prompt_text_ptr ? std::string(prompt_text_ptr) : "";
    
    image_view->generate_image(prompt_copy);
}

void ImageView::on_clear_button_clicked(GtkWidget* widget, gpointer user_data) {
    ImageView* image_view = static_cast<ImageView*>(user_data);
    image_view->clear_image();
    
    // Clear input box
    gtk_entry_buffer_set_text(gtk_entry_get_buffer(GTK_ENTRY(image_view->prompt_entry_)), "", 0);
}

} // namespace gui
} // namespace duorou