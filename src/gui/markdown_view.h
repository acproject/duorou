// Lightweight Markdown rendering widget for GTK4
// - Converts Markdown to HTML (prefers markdown-it-cpp or MD4C if available)
// - Renders HTML via WebKitGTK if present; falls back to selectable GtkTextView
// - Provides copy/export (md/pdf) actions

#ifndef DUOROU_GUI_MARKDOWN_VIEW_H
#define DUOROU_GUI_MARKDOWN_VIEW_H

#if __has_include(<gtk/gtk.h>)
#include <gtk/gtk.h>
#else
// Minimal GTK stubs to reduce editor diagnostics when GTK headers are unavailable.
typedef void GtkWidget; typedef void GtkButton; typedef void GtkTextView; typedef void GtkCssProvider; typedef void GtkStyleContext; typedef void GtkWindow; typedef void GtkEntry;
typedef int gint; typedef void* gpointer;
#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif
#ifndef GTK_ORIENTATION_VERTICAL
#define GTK_ORIENTATION_VERTICAL 0
#endif
#ifndef GTK_ORIENTATION_HORIZONTAL
#define GTK_ORIENTATION_HORIZONTAL 1
#endif
#ifndef GTK_ALIGN_END
#define GTK_ALIGN_END 1
#endif
#ifndef GTK_WRAP_WORD_CHAR
#define GTK_WRAP_WORD_CHAR 0
#endif
#ifndef GTK_STYLE_PROVIDER_PRIORITY_APPLICATION
#define GTK_STYLE_PROVIDER_PRIORITY_APPLICATION 600
#endif
// Common GTK functions used in MarkdownView; define as no-ops/dummies for indexing.
#ifndef gtk_box_new
#define gtk_box_new(...) ((GtkWidget*)nullptr)
#endif
#ifndef gtk_box_append
#define gtk_box_append(...) ((void)0)
#endif
#ifndef gtk_widget_set_halign
#define gtk_widget_set_halign(...) ((void)0)
#endif
#ifndef gtk_widget_set_hexpand
#define gtk_widget_set_hexpand(...) ((void)0)
#endif
#ifndef gtk_widget_set_vexpand
#define gtk_widget_set_vexpand(...) ((void)0)
#endif
#ifndef gtk_text_view_new
#define gtk_text_view_new(...) ((GtkWidget*)nullptr)
#endif
#ifndef gtk_text_view_set_wrap_mode
#define gtk_text_view_set_wrap_mode(...) ((void)0)
#endif
#ifndef gtk_text_view_set_editable
#define gtk_text_view_set_editable(...) ((void)0)
#endif
#ifndef gtk_text_view_set_cursor_visible
#define gtk_text_view_set_cursor_visible(...) ((void)0)
#endif
#ifndef GtkTextBuffer
typedef void GtkTextBuffer;
#endif
#ifndef gtk_text_view_get_buffer
#define gtk_text_view_get_buffer(...) ((GtkTextBuffer*)nullptr)
#endif
#ifndef gtk_text_buffer_set_text
#define gtk_text_buffer_set_text(...) ((void)0)
#endif
#ifndef gtk_button_new_with_label
#define gtk_button_new_with_label(...) ((GtkWidget*)nullptr)
#endif
#ifndef gtk_style_context_add_provider
#define gtk_style_context_add_provider(...) ((void)0)
#endif
#ifndef gtk_widget_get_style_context
#define gtk_widget_get_style_context(...) ((GtkStyleContext*)nullptr)
#endif
#endif
#if __has_include(<string>)
#include <string>
#else
// Minimal std::string stub to reduce editor diagnostics when C++ STL headers are unavailable
namespace std { class string {}; }
#endif

namespace duorou {
namespace gui {

class MarkdownView {
public:
  // Construct view. If WebKitGTK is available at build, it will be used.
  MarkdownView();
  ~MarkdownView();

  // Top-level widget to be embedded in message bubble
  GtkWidget *widget() const { return container_; }

  // Set new markdown content and re-render
  void set_markdown(const std::string &markdown);

  // Get current raw markdown
  std::string get_markdown() const { return markdown_; }

  // Export helpers (return true on success)
  bool export_markdown_to_file(const std::string &file_path) const;
  bool export_pdf_to_file(const std::string &file_path);

  // Expose a small action bar (copy/save buttons) already created inside
  GtkWidget *actions_widget() const { return actions_box_; }

private:
  GtkWidget *container_ = nullptr;   // vertical box: [actions][content]
  GtkWidget *actions_box_ = nullptr; // right-aligned action buttons
  GtkWidget *content_ = nullptr;     // WebKitWebView or GtkTextView
  std::string markdown_;

  // Render markdown -> HTML
  std::string to_html(const std::string &md) const;

  // Build UI once
  void build_ui();

  // Connect actions for copy/save
  void setup_actions();

  // Internals for PDF export
  bool export_pdf_with_webkit(const std::string &file_path);
  bool export_pdf_with_cairo(const std::string &file_path);
};

} // namespace gui
} // namespace duorou

#endif // DUOROU_GUI_MARKDOWN_VIEW_H