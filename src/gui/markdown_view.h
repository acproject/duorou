// Lightweight Markdown rendering widget for GTK4
// - Converts Markdown to HTML (prefers markdown-it-cpp or MD4C if available)
// - Renders HTML via WebKitGTK if present; falls back to selectable GtkTextView
// - Provides copy/export (md/pdf) actions

#ifndef DUOROU_GUI_MARKDOWN_VIEW_H
#define DUOROU_GUI_MARKDOWN_VIEW_H

#include <gtk/gtk.h>
#include <string>

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