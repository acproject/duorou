// Implementation of MarkdownView for GTK4

#include "markdown_view.h"

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <regex>
#include <cctype>
#include <cstring>
#include <cstdlib>
#if __has_include(<gdk/gdk.h>)
#include <gdk/gdk.h>
#define DUOROU_HAVE_GDK 1
#else
#define DUOROU_HAVE_GDK 0
// Minimal GDK/GLib stubs for indexers when headers are unavailable
typedef void GdkDisplay; typedef void GdkClipboard; typedef void GError; typedef void GtkWidget; typedef void GtkButton; typedef void GtkCssProvider; typedef void GtkDialog; typedef void GtkRoot; typedef void GFile;
#ifndef gdk_display_get_default
#define gdk_display_get_default(...) ((GdkDisplay*)nullptr)
#endif
#ifndef gdk_display_get_clipboard
#define gdk_display_get_clipboard(...) ((GdkClipboard*)nullptr)
#endif
#ifndef gdk_clipboard_set_text
#define gdk_clipboard_set_text(...) ((void)0)
#endif
#ifndef g_get_current_dir
#define g_get_current_dir(...) ((char*)nullptr)
#endif
#ifndef g_filename_to_uri
#define g_filename_to_uri(...) ((char*)nullptr)
#endif
#ifndef g_error_free
#define g_error_free(...) ((void)0)
#endif
#endif

// Ensure GTK is included when available, and DUOROU_HAVE_GTK reflects reality
#if __has_include(<gtk/gtk.h>)
#include <gtk/gtk.h>
#ifndef DUOROU_HAVE_GTK
#define DUOROU_HAVE_GTK 1
#endif
#else
#ifndef DUOROU_HAVE_GTK
#define DUOROU_HAVE_GTK 0
#endif
#endif
#if __has_include(<cairo/cairo.h>) && __has_include(<cairo/cairo-pdf.h>)
#include <cairo/cairo.h>
#include <cairo/cairo-pdf.h>
#define DUOROU_HAVE_CAIRO 1
#else
#define DUOROU_HAVE_CAIRO 0
#endif

// WebKitGTK headers vary across versions. Only enable when CMake found WebKitGTK.
#if defined(HAVE_WEBKIT2GTK)
  #if __has_include(<webkit/WebKit.h>)
    #include <webkit/WebKit.h>
  #elif __has_include(<webkit2/webkit2.h>)
    #include <webkit2/webkit2.h>
  #else
    // CMake reported WebKitGTK present but headers not found in compiler search path
    // This keeps compilation going while avoiding accidental inclusion of Apple WebKit.
    #pragma message("HAVE_WEBKIT2GTK defined but no WebKitGTK headers found")
  #endif
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

// CURL（用于下载远程图片，Fallback 渲染路径）
#if __has_include(<curl/curl.h>)
#include <curl/curl.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#define DUOROU_HAVE_CURL 1
#else
#define DUOROU_HAVE_CURL 0
#endif
#ifndef _WIN32
#include <unistd.h>
#endif

// --- GTK/GLib lightweight stubs when headers are missing ---
#if !__has_include(<gtk/gtk.h>)
#ifndef G_CALLBACK
#define G_CALLBACK(f) (f)
#endif
#ifndef GTK_WIDGET
#define GTK_WIDGET(x) (x)
#endif
#ifndef GTK_STYLE_PROVIDER
#define GTK_STYLE_PROVIDER(x) (x)
#endif
#ifndef gtk_css_provider_new
#define gtk_css_provider_new(...) ((GtkCssProvider*)nullptr)
#endif
#ifndef gtk_css_provider_load_from_string
#define gtk_css_provider_load_from_string(...) ((void)0)
#endif
#ifndef g_object_unref
#define g_object_unref(...) ((void)0)
#endif
#ifndef g_signal_connect
#define g_signal_connect(...) ((void)0)
#endif
#ifndef gtk_widget_get_root
#define gtk_widget_get_root(...) ((GtkRoot*)nullptr)
#endif
#ifndef gtk_file_chooser_dialog_new
#define gtk_file_chooser_dialog_new(...) ((GtkWidget*)nullptr)
#endif
#ifndef GTK_FILE_CHOOSER_ACTION_SAVE
#define GTK_FILE_CHOOSER_ACTION_SAVE 0
#endif
#ifndef GTK_RESPONSE_CANCEL
#define GTK_RESPONSE_CANCEL 0
#endif
#ifndef GTK_RESPONSE_ACCEPT
#define GTK_RESPONSE_ACCEPT 1
#endif
#ifndef gtk_file_chooser_set_current_name
#define gtk_file_chooser_set_current_name(...) ((void)0)
#endif
#ifndef GTK_FILE_CHOOSER
#define GTK_FILE_CHOOSER(x) (x)
#endif
#ifndef G_OBJECT
#define G_OBJECT(x) (x)
#endif
#ifndef gtk_window_destroy
#define gtk_window_destroy(...) ((void)0)
#endif
#ifndef gtk_window_present
#define gtk_window_present(...) ((void)0)
#endif
#ifndef gtk_file_chooser_get_file
#define gtk_file_chooser_get_file(...) ((GFile*)nullptr)
#endif
#ifndef g_file_get_path
#define g_file_get_path(...) ((char*)nullptr)
#endif
#ifndef g_free
#define g_free free
#endif
#ifndef g_object_get_data
#define g_object_get_data(...) ((void*)nullptr)
#endif
#ifndef g_object_set_data_full
#define g_object_set_data_full(...) ((void)0)
#endif
#endif // no gtk headers available

// Additional lightweight GTK/GIO stubs for fallback rendering
#if !__has_include(<gtk/gtk.h>)
#ifndef GtkLabel
typedef void GtkLabel;
#endif
#ifndef GtkPicture
typedef void GtkPicture;
#endif
#ifndef GtkStyleContext
typedef void GtkStyleContext;
#endif
#ifndef GtkWindow
typedef void GtkWindow;
#endif
#ifndef gint
typedef int gint;
#endif
#ifndef GTK_BOX
#define GTK_BOX(x) (x)
#endif
#ifndef GTK_WINDOW
#define GTK_WINDOW(x) (x)
#endif
#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif
#ifndef GTK_ALIGN_END
#define GTK_ALIGN_END 1
#endif
#ifndef GTK_ORIENTATION_VERTICAL
#define GTK_ORIENTATION_VERTICAL 0
#endif
#ifndef GTK_ORIENTATION_HORIZONTAL
#define GTK_ORIENTATION_HORIZONTAL 1
#endif
#ifndef GTK_STYLE_PROVIDER_PRIORITY_APPLICATION
#define GTK_STYLE_PROVIDER_PRIORITY_APPLICATION 600
#endif
#ifndef gtk_box_new
#define gtk_box_new(...) ((GtkWidget*)nullptr)
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
#ifndef gtk_button_new_with_label
#define gtk_button_new_with_label(...) ((GtkWidget*)nullptr)
#endif
#ifndef gtk_widget_get_style_context
#define gtk_widget_get_style_context(...) ((GtkStyleContext*)nullptr)
#endif
#ifndef gtk_style_context_add_provider
#define gtk_style_context_add_provider(...) ((void)0)
#endif
#ifndef gtk_box_append
#define gtk_box_append(...) ((void)0)
#endif
#ifndef gtk_label_new
#define gtk_label_new(...) ((GtkWidget*)nullptr)
#endif
#ifndef gtk_label_set_wrap
#define gtk_label_set_wrap(...) ((void)0)
#endif
#ifndef gtk_label_set_xalign
#define gtk_label_set_xalign(...) ((void)0)
#endif
#ifndef gtk_label_set_markup
#define gtk_label_set_markup(...) ((void)0)
#endif
#ifndef gtk_picture_new_for_filename
#define gtk_picture_new_for_filename(...) ((GtkWidget*)nullptr)
#endif
#ifndef gtk_picture_new_for_file
#define gtk_picture_new_for_file(...) ((GtkWidget*)nullptr)
#endif
#ifndef GTK_PICTURE
#define GTK_PICTURE(x) (x)
#endif
#ifndef gtk_picture_set_content_fit
#define gtk_picture_set_content_fit(...) ((void)0)
#endif
#ifndef gtk_picture_set_can_shrink
#define gtk_picture_set_can_shrink(...) ((void)0)
#endif
#ifndef GTK_CONTENT_FIT_CONTAIN
#define GTK_CONTENT_FIT_CONTAIN 0
#endif
#ifndef gtk_grid_new
#define gtk_grid_new(...) ((GtkWidget*)nullptr)
#endif
#ifndef gtk_grid_attach
#define gtk_grid_attach(...) ((void)0)
#endif
#ifndef GTK_GRID
#define GTK_GRID(x) (x)
#endif
#ifndef gtk_frame_new
#define gtk_frame_new(...) ((GtkWidget*)nullptr)
#endif
#ifndef gtk_frame_set_child
#define gtk_frame_set_child(...) ((void)0)
#endif
#ifndef GTK_FRAME
#define GTK_FRAME(x) (x)
#endif
#ifndef gtk_widget_set_margin_top
#define gtk_widget_set_margin_top(...) ((void)0)
#endif
#ifndef gtk_widget_set_margin_bottom
#define gtk_widget_set_margin_bottom(...) ((void)0)
#endif
#ifndef gtk_widget_set_visible
#define gtk_widget_set_visible(...) ((void)0)
#endif
#ifndef g_file_new_for_uri
#define g_file_new_for_uri(...) ((GFile*)nullptr)
#endif
#ifndef g_file_new_for_path
#define g_file_new_for_path(...) ((GFile*)nullptr)
#endif
#ifndef gtk_widget_get_first_child
#define gtk_widget_get_first_child(...) ((GtkWidget*)nullptr)
#endif
#ifndef gtk_widget_get_next_sibling
#define gtk_widget_get_next_sibling(...) ((GtkWidget*)nullptr)
#endif
#ifndef gtk_box_remove
#define gtk_box_remove(...) ((void)0)
#endif
#endif // no gtk headers

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

// --- Simple helpers for media detection ---
static std::string to_lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
  return s;
}

static std::string trim(const std::string &s) {
  size_t a = s.find_first_not_of("\t\r\n ");
  if (a == std::string::npos) return std::string();
  size_t b = s.find_last_not_of("\t\r\n ");
  return s.substr(a, b - a + 1);
}

static bool has_image_extension(const std::string &url_lower) {
  // Consider query fragments and anchors: only check up to '?' or '#'
  static const char *exts[] = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg", ".tiff"};
  size_t end = url_lower.size();
  size_t qpos = url_lower.find('?'); if (qpos != std::string::npos) end = std::min(end, qpos);
  size_t hpos = url_lower.find('#'); if (hpos != std::string::npos) end = std::min(end, hpos);
  for (auto *e : exts) {
    size_t elen = strlen(e);
    if (end >= elen && url_lower.rfind(e, end) == end - elen) {
      return true;
    }
  }
  return false;
}

static bool is_probable_url(const std::string &s) {
  return s.rfind("http://", 0) == 0 || s.rfind("https://", 0) == 0 || s.rfind("file://", 0) == 0 || s.rfind("/", 0) == 0;
}

// Convert lines that are pure image URLs into markdown image syntax, e.g. ![](url)
static std::string preprocess_markdown_for_media(const std::string &md) {
  std::istringstream iss(md);
  std::ostringstream oss;
  std::string line;
  bool changed = false;
  while (std::getline(iss, line)) {
    std::string t = trim(line);
    std::string tl = to_lower(t);
    if (!t.empty() && is_probable_url(t) && has_image_extension(tl)) {
      oss << "![](" << t << ")\n";
      changed = true;
    } else {
      oss << line << "\n";
    }
  }
  // Preserve trailing newline consistency
  std::string out = oss.str();
  if (!out.empty() && out.back() == '\n') {
    // ok
  }
  return changed ? out : md;
}

// Convert a single markdown line to simple Pango markup for fallback rendering.
// Supports headings (#, ##), bold (**bold**), italic (*italic*), inline code (`code`).
static std::string md_line_to_pango(const std::string &line) {
  std::string t = trim(line);
  if (t.empty()) return std::string();

  // Headings
  int hlevel = 0; size_t i = 0;
  while (i < t.size() && t[i] == '#') { ++hlevel; ++i; }
  if (hlevel > 0 && i < t.size() && t[i] == ' ') {
    std::string content = t.substr(i + 1);
    std::string esc = html_escape(content);
    // Use Pango-supported size keywords only
    const char *sizes[] = {"large","x-large","xx-large"};
    const char *size = sizes[std::min(hlevel, 3) - 1];
    return std::string("<span weight='bold' size='") + size + "'>" + esc + "</span>";
  }

  // Basic inline formatting: bold, italic, code
  std::string out; out.reserve(t.size() * 1.3);
  std::string plain;
  bool bold = false, italic = false, code = false;
  for (size_t p = 0; p < t.size(); ++p) {
    if (!code && p + 1 < t.size() && t[p] == '*' && t[p+1] == '*') {
      // flush plain
      if (!plain.empty()) { out += html_escape(plain); plain.clear(); }
      out += bold ? "</b>" : "<b>"; bold = !bold; ++p; continue;
    }
    if (!code && t[p] == '*') {
      if (!plain.empty()) { out += html_escape(plain); plain.clear(); }
      out += italic ? "</i>" : "<i>"; italic = !italic; continue;
    }
    if (t[p] == '`') {
      if (!plain.empty()) { out += html_escape(plain); plain.clear(); }
      out += code ? "</tt>" : "<tt>"; code = !code; continue;
    }
    // accumulate plain characters; will escape on flush
    plain.push_back(t[p]);
  }
  // Close any dangling tags
  if (!plain.empty()) { out += html_escape(plain); plain.clear(); }
  if (code) out += "</tt>"; if (bold) out += "</b>"; if (italic) out += "</i>";
  return out;
}

// Extract first URL-like token from a string (http/https/file/absolute path)
static std::string extract_first_url(const std::string &s) {
  // Look for http(s):// or file:// or an absolute path starting with '/'
  size_t p = std::string::npos;
  for (const char *pref : {"http://", "https://", "file://"}) {
    size_t pos = s.find(pref);
    if (pos != std::string::npos) {
      p = (p == std::string::npos) ? pos : std::min(p, pos);
    }
  }
  if (p != std::string::npos) {
    // take until whitespace or closing bracket
    size_t q = s.find_first_of("\t\n\r )]>", p);
    return s.substr(p, q == std::string::npos ? std::string::npos : (q - p));
  }
  // absolute path
  if (!s.empty() && s[0] == '/') {
    size_t q = s.find_first_of("\t\n\r )]>", 1);
    return s.substr(0, q == std::string::npos ? std::string::npos : q);
  }
  return std::string();
}

// 提取 Markdown 图片语法中的 URL，例如 ![](file:///path/img.png) 或 ![alt](./img.jpg)
static std::string extract_md_image_url(const std::string &s) {
  size_t bang = s.find('!');
  if (bang == std::string::npos) return std::string();
  size_t lb = s.find('[', bang);
  if (lb == std::string::npos) return std::string();
  size_t rb = s.find(']', lb);
  if (rb == std::string::npos) return std::string();
  size_t lp = s.find('(', rb);
  if (lp == std::string::npos) return std::string();
  // 支持行内多个图片语法的第一个匹配
  size_t rp = s.find(')', lp);
  if (rp == std::string::npos || rp <= lp + 1) return std::string();
  std::string url = trim(s.substr(lp + 1, rp - lp - 1));
  if (url.size() >= 2 && url.front() == '<' && url.back() == '>') {
    url = trim(url.substr(1, url.size() - 2));
  }
  // 简单过滤，避免把标题等误识别为URL
  std::string lower = to_lower(url);
  if (lower.rfind("file://", 0) == 0 || is_probable_url(lower) || has_image_extension(lower)) {
    return url;
  }
  return std::string();
}

// Detect typical markdown table separator line: "| --- | :---: | --- |"
static bool is_md_table_separator(const std::string &line) {
  std::string t = trim(line);
  if (t.empty()) return false;
  // allow leading/trailing '|'
  bool has_bar = t.find('|') != std::string::npos;
  if (!has_bar) return false;
  for (char ch : t) {
    if (!(ch == '|' || ch == '-' || ch == ':' || ch == ' ')) {
      return false;
    }
  }
  // must contain at least one "-"
  return t.find('-') != std::string::npos;
}

// Split a markdown table row into cells (trim spaces, ignore leading/trailing '|')
static std::vector<std::string> split_md_table_row(const std::string &line) {
  std::string t = trim(line);
  std::vector<std::string> cells;
  size_t start = 0, end;
  // ignore leading '|'
  if (!t.empty() && t.front() == '|') start = 1;
  while (start <= t.size()) {
    end = t.find('|', start);
    std::string cell = (end == std::string::npos) ? t.substr(start) : t.substr(start, end - start);
    // trim cell
    size_t a = 0; while (a < cell.size() && std::isspace((unsigned char)cell[a])) ++a;
    size_t b = cell.size(); while (b > a && std::isspace((unsigned char)cell[b-1])) --b;
    cells.push_back(cell.substr(a, b - a));
    if (end == std::string::npos) break;
    start = end + 1;
  }
  // drop trailing empty if the line ended with '|'
  if (!cells.empty() && cells.back().empty()) cells.pop_back();
  return cells;
}

// 使用 libcurl 将远程 URL 下载到临时文件（POSIX），供 Fallback 渲染
static bool download_url_to_temp(const std::string &url, std::string &out_path) {
#if DUOROU_HAVE_CURL
#ifndef _WIN32
  char tmpl[] = "/tmp/duorou-img-XXXXXX";
  int fd = mkstemp(tmpl);
  if (fd < 0) {
    return false;
  }
  FILE *fp = fdopen(fd, "wb");
  if (!fp) {
    close(fd);
    unlink(tmpl);
    return false;
  }
  CURL *curl = curl_easy_init();
  if (!curl) {
    fclose(fp);
    unlink(tmpl);
    return false;
  }
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "duorou/markdown-view");
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, +[](char *ptr, size_t size, size_t nmemb, void *userdata) -> size_t {
    FILE *out = static_cast<FILE*>(userdata);
    return fwrite(ptr, size, nmemb, out);
  });
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
  curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 5L);
  CURLcode res = curl_easy_perform(curl);
  curl_easy_cleanup(curl);
  fclose(fp);
  if (res != CURLE_OK) {
    unlink(tmpl);
    return false;
  }
  out_path = std::string(tmpl);
  return true;
#else
  (void)url; (void)out_path;
  return false;
#endif
#else
  (void)url; (void)out_path;
  return false;
#endif
}

namespace duorou {
namespace gui {

MarkdownView::MarkdownView() { build_ui(); setup_actions(); }
MarkdownView::~MarkdownView() {
#ifndef _WIN32
  for (const auto &p : temp_files_) {
    if (!p.empty()) unlink(p.c_str());
  }
#endif
}

void MarkdownView::build_ui() {
  container_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);

  // Actions row (copy / save md / save pdf)
  actions_box_ = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 4);
  gtk_widget_set_halign(actions_box_, GTK_ALIGN_END);
  gtk_widget_set_hexpand(actions_box_, TRUE);

  // Create content container (always a box); append inner view for WebKit when available
  content_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
  gtk_widget_set_hexpand(content_, TRUE);
  gtk_widget_set_vexpand(content_, TRUE);
  
#if DUOROU_HAVE_WEBKIT2GTK
  content_view_ = webkit_web_view_new();
  gtk_widget_set_hexpand(content_view_, TRUE);
  gtk_widget_set_vexpand(content_view_, TRUE);
  // 允许从 file:// 页面访问本地文件资源，确保 Markdown 中的本地图片正常加载
  {
    WebKitSettings *settings = webkit_web_view_get_settings(WEBKIT_WEB_VIEW(content_view_));
    if (settings) {
      // 打开文件访问与跨文件访问（不同目录的 file://）
      webkit_settings_set_allow_file_access_from_file_urls(settings, TRUE);
      webkit_settings_set_allow_universal_access_from_file_urls(settings, TRUE);
      // 保持常见功能开启（某些 Markdown 插件或高亮脚本可能需要）
      webkit_settings_set_enable_javascript(settings, TRUE);
    }
  }
  gtk_box_append(GTK_BOX(content_), content_view_);
#else
  content_view_ = nullptr; // non-WebKit: we will add labels/pictures directly under content_
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
  // Preprocess: auto-convert bare image links to markdown image syntax
  std::string preprocessed = preprocess_markdown_for_media(markdown_);
  std::string html = to_html(preprocessed);
  // Minimal CSS for readability
  std::string full = std::string("<html><head><meta charset='utf-8'>") +
                     "<style>body{font-family:-apple-system,Segoe UI,Roboto,Ubuntu,Helvetica,Arial,sans-serif;line-height:1.5;padding:0;margin:0;}" \
                     "img{max-width:100%;width:100%;height:auto;border-radius:8px;}" \
                     "a{word-break:break-all;}" \
                     "pre,code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:#f6f8fa;padding:2px 4px;border-radius:4px;}" \
                     "pre{padding:8px;overflow:auto;} blockquote{color:#6a737d;border-left:4px solid #dfe2e5;padding:0 1em;}" \
                     "table{border-collapse:collapse;} th,td{border:1px solid #dfe2e5;padding:6px 13px;}" \
                     "</style></head><body>" + html + "</body></html>";
  // 提供本地基础URI，便于加载相对路径和 file:// 资源
  char *cwd = g_get_current_dir();
  GError *uri_err = nullptr;
  char *base_uri = g_filename_to_uri(cwd, nullptr, &uri_err);
  // 如果转换失败，使用一个稳健的回退：尝试使用 HOME 目录作为 file:// 基础
  const char *fallback_home = g_get_home_dir();
  std::string base_for_load;
  if (base_uri && *base_uri) {
    base_for_load.assign(base_uri);
  } else if (fallback_home && *fallback_home) {
    base_for_load = std::string("file://") + fallback_home;
  }
  webkit_web_view_load_html(WEBKIT_WEB_VIEW(content_view_), full.c_str(), base_for_load.empty() ? nullptr : base_for_load.c_str());
  if (base_uri) g_free(base_uri);
  if (cwd) g_free(cwd);
  if (uri_err) g_error_free(uri_err);
#else
  // Fallback: build simple GTK widgets that handle images and text.
  // Clear previous children
  GtkWidget *child = gtk_widget_get_first_child(content_);
  while (child) {
    GtkWidget *next = gtk_widget_get_next_sibling(child);
    gtk_box_remove(GTK_BOX(content_), child);
    child = next;
  }
  // 清理上一次渲染留下的临时图片文件
#ifndef _WIN32
  for (const auto &p : temp_files_) {
    if (!p.empty()) unlink(p.c_str());
  }
#endif
  temp_files_.clear();

  auto append_text = [&](const std::string &text) {
    if (text.empty()) return;
    // Render with simple Pango markup to improve readability
    std::string markup = md_line_to_pango(text);
    GtkWidget *lbl = gtk_label_new(NULL);
    if (!markup.empty()) {
      gtk_label_set_markup((GtkLabel*)lbl, markup.c_str());
    } else {
      // Fallback to plain text when markup conversion yields empty
      gtk_label_set_markup((GtkLabel*)lbl, html_escape(text).c_str());
    }
    gtk_label_set_wrap((GtkLabel*)lbl, TRUE);
    gtk_label_set_xalign((GtkLabel*)lbl, 0.0f);
    gtk_widget_set_hexpand(lbl, TRUE);
    gtk_box_append(GTK_BOX(content_), lbl);
  };

  auto append_picture = [&](const std::string &url) {
    GtkWidget *pic = nullptr;
    auto try_local_path = [&](const std::string &path) {
      GtkWidget *w = nullptr;
      GFile *gf = g_file_new_for_path(path.c_str());
      if (gf) {
        w = gtk_picture_new_for_file(gf);
        g_object_unref(gf);
      }
      if (!w) w = gtk_picture_new_for_filename(path.c_str());
      return w;
    };
    if (url.rfind("http://", 0) == 0 || url.rfind("https://", 0) == 0) {
      // In pure GTK fallback, http(s) fetching may be unavailable.
      // 先尝试通过 GFile 打开；失败则用 libcurl 下载到临时文件
      std::string lower = to_lower(url);
      GFile *gf = nullptr;
      if (has_image_extension(lower)) {
        gf = g_file_new_for_uri(url.c_str());
        if (gf) {
          pic = gtk_picture_new_for_file(gf);
          g_object_unref(gf);
        }
      }
      if (!pic) {
        std::string tmp;
        if (download_url_to_temp(url, tmp)) {
          pic = gtk_picture_new_for_filename(tmp.c_str());
          if (pic) {
            temp_files_.push_back(tmp);
          } else {
#ifndef _WIN32
            unlink(tmp.c_str());
#endif
          }
        }
      }
    } else if (url.rfind("file://", 0) == 0) {
      GFile *gf = g_file_new_for_uri(url.c_str());
      if (gf) {
        // 首选基于 GFile 的加载
        pic = gtk_picture_new_for_file(gf);
        if (!pic) {
          // 退化：将 file:// URI 转换为本地路径再尝试
          char *path = g_file_get_path(gf);
          if (path) {
            GtkWidget *alt = gtk_picture_new_for_filename(path);
            if (alt) pic = alt;
            g_free(path);
          }
        }
        g_object_unref(gf);
      }
    } else {
      // Local paths: support absolute, relative and '~' expansion
      std::string path = url;
      if (!url.empty() && url[0] == '~') {
        const char *home = getenv("HOME");
        if (home && url.size() > 1 && url[1] == '/') {
          path = std::string(home) + url.substr(1);
        }
      }
      if (!path.empty() && path[0] == '/') {
        pic = try_local_path(path);
      } else {
        char *cwd = g_get_current_dir();
        if (cwd) {
          std::string abs = std::string(cwd) + "/" + path;
          pic = try_local_path(abs);
          g_free(cwd);
        } else {
          pic = try_local_path(path);
        }
      }
    }
    if (!pic) {
      // If we cannot create picture, just show the URL text
      append_text(url);
      return;
    }
    // Ensure picture keeps a non-zero height and respects aspect ratio
    gtk_picture_set_content_fit(GTK_PICTURE(pic), GTK_CONTENT_FIT_CONTAIN);
    gtk_picture_set_can_shrink(GTK_PICTURE(pic), FALSE);
    gtk_widget_set_hexpand(pic, TRUE);
    gtk_widget_set_vexpand(pic, TRUE);
    // Mark this widget as a markdown picture for later width sync
    g_object_set_data(G_OBJECT(pic), "markdown_picture", (gpointer)1);
    // Apply target width immediately to avoid height collapsing
    if (target_width_ > 0) {
      gtk_widget_set_size_request(pic, target_width_, -1);
    }
    gtk_widget_set_margin_top(pic, 4);
    gtk_widget_set_margin_bottom(pic, 4);
    gtk_box_append(GTK_BOX(content_), pic);
  };

  // Parse markdown by lines; render images, tables and text blocks
  std::vector<std::string> lines;
  {
    std::istringstream iss(markdown_);
    std::string l; while (std::getline(iss, l)) lines.push_back(l);
  }
  for (size_t idx = 0; idx < lines.size(); ++idx) {
    std::string line = lines[idx];
    std::string t = trim(line);
    if (t.empty()) continue;
    std::string tl = to_lower(t);

    // Case 0: custom media hint line like "<__media__>: ..."
    if (t.rfind("<__media__>", 0) == 0 || t.rfind("<__media__>:", 0) == 0) {
      std::string url = extract_first_url(t);
      if (!url.empty()) {
        append_picture(url);
        continue;
      }
      // no url found, show as text
      append_text(line);
      continue;
    }

    // Case 1.5: 仅当整行就是一个 markdown 图片语法时，作为纯图片行处理
    {
      static const std::regex re_md_pure_line("^\\s*!\\[[^\\]]*\\]\\(([^)]+)\\)\\s*$");
      std::smatch m;
      if (std::regex_match(line, m, re_md_pure_line) && m.size() >= 2) {
        std::string img_url = trim(m[1].str());
        append_picture(img_url);
        continue;
      }
    }

    // Case 1: whole-line is an image URL (with common extensions)
    if (is_probable_url(t) && has_image_extension(tl)) {
      append_picture(t);
      continue;
    }

    // Case 2: markdown table block: header, separator, then rows
    if (t.find('|') != std::string::npos && idx + 1 < lines.size() && is_md_table_separator(lines[idx + 1])) {
      // Collect table rows
      std::vector<std::vector<std::string>> rows;
      // Header
      rows.push_back(split_md_table_row(lines[idx]));
      // Skip separator
      idx += 1;
      // Data rows
      size_t rstart = idx + 1;
      for (; rstart < lines.size(); ++rstart) {
        std::string lt = trim(lines[rstart]);
        if (lt.empty() || lt.find('|') == std::string::npos) break;
        rows.push_back(split_md_table_row(lines[rstart]));
      }
      // Render grid
      GtkWidget *grid = gtk_grid_new();
      gtk_widget_set_hexpand(grid, TRUE);
      gtk_widget_set_margin_top(grid, 4);
      gtk_widget_set_margin_bottom(grid, 4);
      size_t nrows = rows.size();
      size_t ncols = 0; for (auto &rw : rows) ncols = std::max(ncols, rw.size());
      for (size_t r = 0; r < nrows; ++r) {
        for (size_t c = 0; c < ncols; ++c) {
          std::string cell = (c < rows[r].size()) ? rows[r][c] : std::string();
          std::string markup = r == 0 ? (std::string("<b>") + html_escape(cell) + "</b>") : html_escape(cell);
          GtkWidget *frame = gtk_frame_new(NULL);
          GtkWidget *lbl = gtk_label_new(NULL);
          gtk_label_set_markup((GtkLabel*)lbl, markup.c_str());
          gtk_label_set_wrap((GtkLabel*)lbl, TRUE);
          gtk_label_set_xalign((GtkLabel*)lbl, 0.0f);
          gtk_widget_set_margin_top(lbl, 2);
          gtk_widget_set_margin_bottom(lbl, 2);
          gtk_frame_set_child(GTK_FRAME(frame), lbl);
          gtk_widget_set_hexpand(frame, TRUE);
          gtk_widget_set_margin_top(frame, 1);
          gtk_widget_set_margin_bottom(frame, 1);
          gtk_grid_attach(GTK_GRID(grid), frame, (int)c, (int)r, 1, 1);
        }
      }
      gtk_box_append(GTK_BOX(content_), grid);
      // Advance index
      idx = (rstart == 0 ? idx : rstart - 1);
      continue;
    }

    // Case 3: inline images mixed with text: ![alt](url)
    size_t pos = 0;
    std::string before;
    bool emitted_any = false;
    while (true) {
      size_t bang = line.find("![", pos);
      if (bang == std::string::npos) break;
      before += line.substr(pos, bang - pos);
      size_t rb = line.find(']', bang + 2);
      if (rb == std::string::npos) break;
      size_t lp = line.find('(', rb + 1);
      if (lp == std::string::npos) break;
      size_t rp = line.find(')', lp + 1);
      if (rp == std::string::npos) break;
      std::string url = trim(line.substr(lp + 1, rp - (lp + 1)));
      if (!url.empty()) {
        if (!before.empty()) { append_text(before); before.clear(); emitted_any = true; }
        append_picture(url);
        emitted_any = true;
      } else {
        before += line.substr(bang, rp - bang + 1);
      }
      pos = rp + 1;
    }
    before += line.substr(pos);
    if (!before.empty()) { append_text(before); emitted_any = true; }
    if (!emitted_any) { append_text(line); }
  }
#endif
}

void MarkdownView::set_target_width(int px) {
  target_width_ = px;
  if (!content_) return;
  // Apply to existing picture children
  GtkWidget *child = gtk_widget_get_first_child(content_);
  while (child) {
    GtkWidget *next = gtk_widget_get_next_sibling(child);
    gpointer tag = g_object_get_data(G_OBJECT(child), "markdown_picture");
    if (tag && target_width_ > 0) {
      gtk_widget_set_size_request(child, target_width_, -1);
      // Also ensure vertical expansion so height doesn't collapse under layout changes
      gtk_widget_set_vexpand(child, TRUE);
    }
    child = next;
  }
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
  // Minimal fallback HTML: convert pure image links per-line; otherwise escape as paragraphs
  std::istringstream iss(md);
  std::ostringstream out;
  std::string line;
  while (std::getline(iss, line)) {
    std::string t = trim(line);
    std::string tl = to_lower(t);
    // Case 1: whole-line is an image URL
    if (!t.empty() && is_probable_url(t) && has_image_extension(tl)) {
      out << "<img src=\"" << t << "\" style=\"max-width:100%;height:auto;border-radius:8px;\">\n";
      continue;
    }
    if (t.empty()) continue;

    // Case 2: parse inline markdown image syntax: ![alt](url)
    std::string rendered;
    size_t pos = 0;
    bool has_img = false;
    while (true) {
      size_t bang = line.find("![", pos);
      if (bang == std::string::npos) {
        // append remaining as text
        rendered.append(html_escape(line.substr(pos)));
        break;
      }
      // append text before image
      rendered.append(html_escape(line.substr(pos, bang - pos)));
      size_t rb = line.find(']', bang + 2);
      if (rb == std::string::npos) { // malformed, treat as text
        rendered.append(html_escape(line.substr(bang)));
        break;
      }
      size_t lp = line.find('(', rb + 1);
      if (lp == std::string::npos) { // malformed
        rendered.append(html_escape(line.substr(bang)));
        break;
      }
      size_t rp = line.find(')', lp + 1);
      if (rp == std::string::npos) { // malformed
        rendered.append(html_escape(line.substr(bang)));
        break;
      }
      std::string alt = line.substr(bang + 2, rb - (bang + 2));
      std::string url = trim(line.substr(lp + 1, rp - (lp + 1)));
      std::string url_l = to_lower(url);
      // Emit image only if URL seems valid
      if (is_probable_url(url) || has_image_extension(url_l)) {
        rendered.append(std::string("<img src=\"") + html_escape(url) + "\" alt=\"" + html_escape(alt) + "\" style=\"max-width:100%;height:auto;border-radius:8px;\">");
        has_img = true;
      } else {
        // Not a valid URL, fall back to text
        rendered.append(html_escape(line.substr(bang, rp - bang + 1)));
      }
      pos = rp + 1;
    }
    // Wrap line: if we inserted images, allow them without extra <p>; include any surrounding text in <p>
    if (has_img) {
      out << rendered << "\n";
    } else {
      out << "<p>" << rendered << "</p>\n";
    }
  }
  return out.str().empty() ? std::string("<pre>") + html_escape(md) + "</pre>" : out.str();
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
  // Print the inner WebKit view, not the outer GTK box
  WebKitPrintOperation *op = webkit_print_operation_new(WEBKIT_WEB_VIEW(content_view_));
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
  // If Cairo headers are unavailable, indicate failure gracefully
#if DUOROU_HAVE_CAIRO
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
#else
  (void)file_path;
  return false;
#endif
}

} // namespace gui
} // namespace duorou
#if !__has_include(<gtk/gtk.h>)
#ifndef gtk_widget_get_first_child
#define gtk_widget_get_first_child(...) ((GtkWidget*)nullptr)
#endif
#ifndef gtk_widget_get_next_sibling
#define gtk_widget_get_next_sibling(...) ((GtkWidget*)nullptr)
#endif
#ifndef gtk_box_remove
#define gtk_box_remove(...) ((void)0)
#endif
#endif // no gtk headers