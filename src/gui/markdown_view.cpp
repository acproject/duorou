// Implementation of MarkdownView for GTK4

#include "markdown_view.h"

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <gdk/gdk.h>
#include <cairo/cairo.h>
#include <cairo/cairo-pdf.h>

#if __has_include(<webkit2/webkit2.h>)
#include <webkit2/webkit2.h>
#define DUOROU_HAVE_WEBKIT2GTK 1
#else
#define DUOROU_HAVE_WEBKIT2GTK 0
#endif

// Optional MD4C for Markdown -> HTML conversion
#if __has_include(<md4c-html.h>)
#include <md4c-html.h>
#define DUOROU_HAVE_MD4C 1
#else
#define DUOROU_HAVE_MD4C 0
#endif

// Simple HTML escape for fallback
static std::string html_escape(const std::string &in) {
  std::string out;
  out.reserve(in.size() * 1.1);
  for (char c : in) {
    switch (c) {
    case '&': out += "&amp;"; break;
    case '<': out += "&lt;"; break;
    case '>': out += "&gt;"; break;
    case '"': out += "&quot;"; break;
    case '\'': out += "&#39;"; break;
    default: out.push_back(c); break;
    }
  }
  return out;
}

namespace duorou {
namespace gui {

MarkdownView::MarkdownView() { build_ui(); setup_actions(); }
MarkdownView::~MarkdownView() {}

void MarkdownView::build_ui() {
  container_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);

  // Actions row (copy / save md / save pdf)
  actions_box_ = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 4);
  gtk_widget_set_halign(actions_box_, GTK_ALIGN_END);
  gtk_widget_set_hexpand(actions_box_, TRUE);

  // Create content area
#if DUOROU_HAVE_WEBKIT2GTK
  content_ = webkit_web_view_new();
  gtk_widget_set_hexpand(content_, TRUE);
  gtk_widget_set_vexpand(content_, TRUE);
#else
  content_ = gtk_text_view_new();
  gtk_text_view_set_wrap_mode(GTK_TEXT_VIEW(content_), GTK_WRAP_WORD_CHAR);
  gtk_text_view_set_editable(GTK_TEXT_VIEW(content_), FALSE);
  gtk_text_view_set_cursor_visible(GTK_TEXT_VIEW(content_), TRUE);
  gtk_widget_set_hexpand(content_, TRUE);
  gtk_widget_set_vexpand(content_, TRUE);
#endif

  // Pack to container
  gtk_box_append(GTK_BOX(container_), actions_box_);
  gtk_box_append(GTK_BOX(container_), content_);
}

void MarkdownView::setup_actions() {
  // Buttons with text; icons can be added later
  GtkWidget *btn_copy = gtk_button_new_with_label("复制");
  GtkWidget *btn_save_md = gtk_button_new_with_label("保存MD");
  GtkWidget *btn_save_pdf = gtk_button_new_with_label("保存PDF");

  gtk_box_append(GTK_BOX(actions_box_), btn_copy);
  gtk_box_append(GTK_BOX(actions_box_), btn_save_md);
  gtk_box_append(GTK_BOX(actions_box_), btn_save_pdf);

  // Improve legibility in colored bubbles: force dark text on white buttons
  const char *btn_css =
      "button, button:hover, button:active, button:checked, button:focus {"
      "  color: #000000;"
      "  background-color: #61727cff;"
      "  border-radius: 8px;"
      "  padding: 4px 8px;"
      "}";
  GtkCssProvider *btn_provider = gtk_css_provider_new();
  gtk_css_provider_load_from_string(btn_provider, btn_css);
  gtk_style_context_add_provider(gtk_widget_get_style_context(btn_copy),
                                 GTK_STYLE_PROVIDER(btn_provider),
                                 GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
  gtk_style_context_add_provider(gtk_widget_get_style_context(btn_save_md),
                                 GTK_STYLE_PROVIDER(btn_provider),
                                 GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
  gtk_style_context_add_provider(gtk_widget_get_style_context(btn_save_pdf),
                                 GTK_STYLE_PROVIDER(btn_provider),
                                 GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
  g_object_unref(btn_provider);

  // Copy: if WebKit, user can select and Cmd/Ctrl+C; button copies all markdown
  g_signal_connect(
      btn_copy, "clicked",
      G_CALLBACK(+[](GtkButton *, gpointer user_data) {
        auto *self = static_cast<MarkdownView *>(user_data);
        if (!self)
          return;
        GdkDisplay *display = gdk_display_get_default();
        if (!display)
          return;
        GdkClipboard *clipboard = gdk_display_get_clipboard(display);
        if (!clipboard)
          return;
        std::string text = self->get_markdown();
        gdk_clipboard_set_text(clipboard, text.c_str());
      }),
      this);

  // Save MD (GTK4: async response)
  g_signal_connect(
      btn_save_md, "clicked",
      G_CALLBACK(+[](GtkButton *b, gpointer user_data) {
        auto *self = static_cast<MarkdownView *>(user_data);
        if (!self)
          return;
        GtkRoot *root = gtk_widget_get_root(GTK_WIDGET(b));
        GtkWidget *dialog = gtk_file_chooser_dialog_new(
            "保存为 Markdown", GTK_WINDOW(root), GTK_FILE_CHOOSER_ACTION_SAVE,
            "取消", GTK_RESPONSE_CANCEL, "保存", GTK_RESPONSE_ACCEPT, NULL);
        gtk_file_chooser_set_current_name(GTK_FILE_CHOOSER(dialog), "chat.md");
        g_object_set_data_full(G_OBJECT(dialog), "markdown_view", self, NULL);
        g_signal_connect(dialog, "response",
                         G_CALLBACK(+[](GtkDialog *dlg, gint response, gpointer) {
                           if (response == GTK_RESPONSE_ACCEPT) {
                             GFile *file = gtk_file_chooser_get_file(GTK_FILE_CHOOSER(dlg));
                             auto *mv = static_cast<MarkdownView *>(g_object_get_data(G_OBJECT(dlg), "markdown_view"));
                             if (mv && file) {
                               char *path = g_file_get_path(file);
                               if (path) {
                                 mv->export_markdown_to_file(path);
                                 g_free(path);
                               }
                               g_object_unref(file);
                             }
                           }
                           gtk_window_destroy(GTK_WINDOW(dlg));
                         }),
                         NULL);
        gtk_window_present(GTK_WINDOW(dialog));
      }),
      this);

  // Save PDF (GTK4: async response)
  g_signal_connect(
      btn_save_pdf, "clicked",
      G_CALLBACK(+[](GtkButton *b, gpointer user_data) {
        auto *self = static_cast<MarkdownView *>(user_data);
        if (!self)
          return;
        GtkRoot *root = gtk_widget_get_root(GTK_WIDGET(b));
        GtkWidget *dialog = gtk_file_chooser_dialog_new(
            "导出为 PDF", GTK_WINDOW(root), GTK_FILE_CHOOSER_ACTION_SAVE, "取消",
            GTK_RESPONSE_CANCEL, "保存", GTK_RESPONSE_ACCEPT, NULL);
        gtk_file_chooser_set_current_name(GTK_FILE_CHOOSER(dialog), "chat.pdf");
        g_object_set_data_full(G_OBJECT(dialog), "markdown_view", self, NULL);
        g_signal_connect(dialog, "response",
                         G_CALLBACK(+[](GtkDialog *dlg, gint response, gpointer) {
                           if (response == GTK_RESPONSE_ACCEPT) {
                             GFile *file = gtk_file_chooser_get_file(GTK_FILE_CHOOSER(dlg));
                             auto *mv = static_cast<MarkdownView *>(g_object_get_data(G_OBJECT(dlg), "markdown_view"));
                             if (mv && file) {
                               char *path = g_file_get_path(file);
                               if (path) {
                                 mv->export_pdf_to_file(path);
                                 g_free(path);
                               }
                               g_object_unref(file);
                             }
                           }
                           gtk_window_destroy(GTK_WINDOW(dlg));
                         }),
                         NULL);
        gtk_window_present(GTK_WINDOW(dialog));
      }),
      this);
}

void MarkdownView::set_markdown(const std::string &markdown) {
  markdown_ = markdown;
#if DUOROU_HAVE_WEBKIT2GTK
  std::string html = to_html(markdown_);
  // Minimal CSS for readability
  std::string full = std::string("<html><head><meta charset='utf-8'>") +
                     "<style>body{font-family:-apple-system,Segoe UI,Roboto,Ubuntu,Helvetica,Arial,sans-serif;line-height:1.5;padding:0;margin:0;}" \
                     "pre,code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:#f6f8fa;padding:2px 4px;border-radius:4px;}" \
                     "pre{padding:8px;overflow:auto;} blockquote{color:#6a737d;border-left:4px solid #dfe2e5;padding:0 1em;}" \
                     "table{border-collapse:collapse;} th,td{border:1px solid #dfe2e5;padding:6px 13px;}" \
                     "</style></head><body>" + html + "</body></html>";
  webkit_web_view_load_html(WEBKIT_WEB_VIEW(content_), full.c_str(), nullptr);
#else
  // Fallback: show raw markdown text in read-only text view (selectable)
  GtkTextBuffer *buf = gtk_text_view_get_buffer(GTK_TEXT_VIEW(content_));
  gtk_text_buffer_set_text(buf, markdown_.c_str(), (gint)markdown_.size());
#endif
}

std::string MarkdownView::to_html(const std::string &md) const {
#if DUOROU_HAVE_MD4C
  std::string html;
  html.reserve(md.size() * 2);
  auto cb = +[](const MD_CHAR *text, MD_SIZE size, void *userdata) {
    auto *out = static_cast<std::string *>(userdata);
    out->append(text, size);
  };
  // Flags: use defaults for broad compatibility
  md_html(md.c_str(), (MD_SIZE)md.size(), cb, &html, 0, 0);
  return html;
#else
  // Minimal fallback: wrap in <pre> with escaping
  return std::string("<pre>") + html_escape(md) + "</pre>";
#endif
}

bool MarkdownView::export_markdown_to_file(const std::string &file_path) const {
  std::ofstream ofs(file_path, std::ios::binary);
  if (!ofs.is_open())
    return false;
  ofs.write(markdown_.data(), (std::streamsize)markdown_.size());
  return ofs.good();
}

bool MarkdownView::export_pdf_to_file(const std::string &file_path) {
#if DUOROU_HAVE_WEBKIT2GTK
  return export_pdf_with_webkit(file_path);
#else
  return export_pdf_with_cairo(file_path);
#endif
}

bool MarkdownView::export_pdf_with_webkit(const std::string &file_path) {
#if DUOROU_HAVE_WEBKIT2GTK
  WebKitPrintOperation *op = webkit_print_operation_new(WEBKIT_WEB_VIEW(content_));
  GtkPrintSettings *settings = gtk_print_settings_new();
  gtk_print_settings_set(settings, GTK_PRINT_SETTINGS_OUTPUT_FILE_FORMAT, "pdf");
  gchar *uri = g_filename_to_uri(file_path.c_str(), nullptr, nullptr);
  if (uri) {
    gtk_print_settings_set(settings, GTK_PRINT_SETTINGS_OUTPUT_URI, uri);
    g_free(uri);
  }
  webkit_print_operation_set_print_settings(op, settings);
  // Print directly to file using settings, avoid extra dialog
  webkit_print_operation_print(op);
  g_object_unref(settings);
  g_object_unref(op);
  return true;
#else
  (void)file_path;
  return false;
#endif
}

bool MarkdownView::export_pdf_with_cairo(const std::string &file_path) {
  // Very simple PDF export: write raw markdown text; formatting minimal
  cairo_surface_t *surface = cairo_pdf_surface_create(file_path.c_str(), 595, 842); // A4 in points
  cairo_t *cr = cairo_create(surface);

  // Use Pango for layout if available
#if __has_include(<pango/pangocairo.h>)
  #include <pango/pangocairo.h>
  PangoLayout *layout = pango_cairo_create_layout(cr);
  pango_layout_set_width(layout, 555 * PANGO_SCALE);
  pango_layout_set_text(layout, markdown_.c_str(), (int)markdown_.size());
  PangoFontDescription *desc = pango_font_description_from_string("Monospace 10");
  pango_layout_set_font_description(layout, desc);
  pango_cairo_update_layout(cr, layout);
  cairo_move_to(cr, 20, 20);
  pango_cairo_show_layout(cr, layout);
  g_object_unref(layout);
  pango_font_description_free(desc);
#else
  // Fallback: very basic Cairo text
  cairo_select_font_face(cr, "Monaco", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);
  cairo_set_font_size(cr, 10);
  double x = 20, y = 30;
  std::istringstream iss(markdown_);
  std::string line;
  while (std::getline(iss, line)) {
    cairo_move_to(cr, x, y);
    cairo_show_text(cr, line.c_str());
    y += 14; // line height
  }
#endif

  cairo_show_page(cr);
  cairo_destroy(cr);
  cairo_surface_finish(surface);
  cairo_surface_destroy(surface);
  return true;
}

} // namespace gui
} // namespace duorou