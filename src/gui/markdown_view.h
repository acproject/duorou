// Lightweight Markdown rendering widget for GTK4
// - Converts Markdown to HTML (prefers markdown-it-cpp or MD4C if available)
// - Renders HTML via WebKitGTK if present; falls back to selectable GtkTextView
// - Provides copy/export (md/pdf) actions

#ifndef DUOROU_GUI_MARKDOWN_VIEW_H
#define DUOROU_GUI_MARKDOWN_VIEW_H

#include <vector>

#if __has_include(<gtk/gtk.h>)
#include <gtk/gtk.h>
#else
// Minimal GTK stubs to reduce editor diagnostics when GTK headers are
// unavailable.
typedef void GtkWidget;
typedef void GtkButton;
typedef void GtkTextView;
typedef void GtkCssProvider;
typedef void GtkStyleContext;
typedef void GtkWindow;
typedef void GtkEntry;
typedef int gint;
typedef void *gpointer;
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
// Note: do not define function-like GTK stubs here to avoid runtime issues.
// Keep only type/constant placeholders for indexing when headers are missing.
typedef void GtkTextBuffer;
#endif
#ifdef __cplusplus
#if __has_include(<string>)
#include <string>
#else
// Forward declaration when <string> is missing but compiling as C++
namespace std {
class string;
}
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

  // Set target width for inner media (pictures) so height won't collapse
  void set_target_width(int px);

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
  // Content container (vertical box). If WebKit is available, a WebKitWebView
  // will be appended inside; otherwise, we build GTK widgets (labels/pictures)
  // directly under this box.
  GtkWidget *content_ = nullptr;
  // Optional inner view when using WebKit
  GtkWidget *content_view_ = nullptr;
  std::string markdown_;
  // 临时图片文件列表（用于 Fallback 远程图片下载的生命周期管理）
  std::vector<std::string> temp_files_;

  // Target width for media items inside content_
  int target_width_ = 0;

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
#else
// Non-C++ indexing mode: avoid namespace/class to reduce diagnostics.
// Provide opaque placeholder so C indexers don't error out on usage.
typedef void duorou_gui_markdown_view_opaque_t;
#endif

#endif // DUOROU_GUI_MARKDOWN_VIEW_H