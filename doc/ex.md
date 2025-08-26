
下面是一套可直接编译运行的最小示例工程，使用 **GTK4** + **GStreamer** 实现跨平台（Windows / macOS）摄像头预览与音视频录制。

---

## 目录结构

```
gtk4-gst-recorder/
├─ CMakeLists.txt
├─ README.md
├─ mac/
│  └─ Info.plist.in
└─ src/
   └─ main.c
```

---

## CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.16)
project(gtk4_gst_recorder C)

set(CMAKE_C_STANDARD 11)

# 让 pkg-config 帮我们找到 GTK4 和 GStreamer
find_package(PkgConfig REQUIRED)

pkg_check_modules(GTK4 REQUIRED IMPORTED_TARGET gtk4)
pkg_check_modules(GSTREAMER REQUIRED IMPORTED_TARGET gstreamer-1.0 gstreamer-video-1.0 gstreamer-audio-1.0)

add_executable(gtk4_gst_recorder
    src/main.c
)

# macOS: 生成 bundle，并配置摄像头/麦克风权限描述
if(APPLE)
    set_target_properties(gtk4_gst_recorder PROPERTIES MACOSX_BUNDLE TRUE)
    set(MACOSX_BUNDLE_BUNDLE_NAME "GTK4GstRecorder")
    set(MACOSX_BUNDLE_GUI_IDENTIFIER "com.example.gtk4gstrecorder")
    set(MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/mac/Info.plist.in)
endif()

# 包含和链接
target_link_libraries(gtk4_gst_recorder
    PkgConfig::GTK4
    PkgConfig::GSTREAMER
)

# Windows: 如果使用 MSYS2/MINGW，通常需要把运行时 dll 放到可执行文件目录。
# 这里不做自动复制，README 里说明如何运行。

# 更友好的编译警告
if(MSVC)
    target_compile_options(gtk4_gst_recorder PRIVATE /W3)
else()
    target_compile_options(gtk4_gst_recorder PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()
```

---

## mac/Info.plist.in

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleDevelopmentRegion</key>
	<string>en</string>
	<key>CFBundleExecutable</key>
	<string>gtk4_gst_recorder</string>
	<key>CFBundleIconFile</key>
	<string></string>
	<key>CFBundleIdentifier</key>
	<string>com.example.gtk4gstrecorder</string>
	<key>CFBundleInfoDictionaryVersion</key>
	<string>6.0</string>
	<key>CFBundleName</key>
	<string>GTK4GstRecorder</string>
	<key>CFBundlePackageType</key>
	<string>APPL</string>
	<key>CFBundleShortVersionString</key>
	<string>1.0</string>
	<key>LSMinimumSystemVersion</key>
	<string>10.15</string>

	<!-- 摄像头/麦克风权限描述（必需） -->
	<key>NSCameraUsageDescription</key>
	<string>App needs camera access to record video.</string>
	<key>NSMicrophoneUsageDescription</key>
	<string>App needs microphone access to record audio.</string>
</dict>
</plist>
```

---

## src/main.c

```c
#include <gtk/gtk.h>
#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/audio/audio.h>

/*
 * 本示例：
 *  - 使用 gtksink 进行摄像头预览（GTK4 原生控件）
 *  - 同时从摄像头+麦克风采集，编码后写入 MP4 文件（H264/AAC）
 *  - 自动探测可用的编码器，失败则给出错误提示
 */

typedef struct {
    GtkWindow   *window;
    GtkButton   *btn_start;
    GtkButton   *btn_stop;
    GtkLabel    *status;
    GtkWidget   *video_widget; // 来自 gtksink 的 widget

    GstElement  *pipeline;
    GstElement  *video_src;
    GstElement  *audio_src;
    GstElement  *video_convert;
    GstElement  *audio_convert;
    GstElement  *audio_resample;
    GstElement  *video_enc;
    GstElement  *audio_enc;
    GstElement  *video_queue_preview;
    GstElement  *video_queue_rec;
    GstElement  *audio_queue_rec;
    GstElement  *tee;
    GstElement  *preview_sink;    // gtksink
    GstElement  *mux;
    GstElement  *file_sink;

    gchar       *outfile;
    gboolean     recording;
} App;

static void set_status(App *app, const char *msg) {
    gtk_label_set_text(app->status, msg);
}

static GstElement* try_make(const char *factory) {
    GstElement *e = gst_element_factory_make(factory, NULL);
    return e; // 可能为 NULL
}

static GstElement* choose_h264_encoder(void) {
    // 按优先级尝试：x264enc -> vtenc_h264(macOS VideoToolbox) -> openh264enc
    const char *candidates[] = { "x264enc", "vtenc_h264", "openh264enc" };
    for (size_t i = 0; i < G_N_ELEMENTS(candidates); ++i) {
        GstElement *e = try_make(candidates[i]);
        if (e) return e;
    }
    return NULL;
}

static GstElement* choose_aac_encoder(void) {
    // 常见：avenc_aac(需要 gst-libav) / fdkaacenc / voaacenc / faac
    const char *candidates[] = { "avenc_aac", "fdkaacenc", "voaacenc", "faac" };
    for (size_t i = 0; i < G_N_ELEMENTS(candidates); ++i) {
        GstElement *e = try_make(candidates[i]);
        if (e) return e;
    }
    return NULL;
}

static gboolean bus_msg_cb(GstBus *bus, GstMessage *msg, gpointer user_data) {
    (void)bus;
    App *app = (App*)user_data;
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_ERROR: {
            GError *err = NULL; gchar *dbg = NULL;
            gst_message_parse_error(msg, &err, &dbg);
            g_printerr("GStreamer ERROR: %s\n", err->message);
            if (dbg) g_printerr("Debug: %s\n", dbg);
            set_status(app, "Error: check console");
            g_clear_error(&err); g_free(dbg);
            // 出错就停掉
            gst_element_set_state(app->pipeline, GST_STATE_NULL);
            app->recording = FALSE;
            gtk_widget_set_sensitive(GTK_WIDGET(app->btn_start), TRUE);
            gtk_widget_set_sensitive(GTK_WIDGET(app->btn_stop), FALSE);
            break;
        }
        case GST_MESSAGE_EOS:
            // 录制自然结束（我们发送 EOS 后收到）
            gst_element_set_state(app->pipeline, GST_STATE_NULL);
            app->recording = FALSE;
            set_status(app, "Stopped (EOS)");
            gtk_widget_set_sensitive(GTK_WIDGET(app->btn_start), TRUE);
            gtk_widget_set_sensitive(GTK_WIDGET(app->btn_stop), FALSE);
            break;
        default:
            break;
    }
    return TRUE;
}

static gboolean build_pipeline(App *app) {
    // 选择平台源
#ifdef _WIN32
    const char *video_src_factory = "dshowvideosrc"; // 或者 ksvideosrc
    const char *audio_src_factory = "wasapisrc";
#elif defined(__APPLE__)
    const char *video_src_factory = "avfvideosrc";
    const char *audio_src_factory = "avfaudiosrc";
#else
    const char *video_src_factory = "v4l2src";
    const char *audio_src_factory = "pulsesrc"; // Linux 示例
#endif

    app->pipeline = gst_pipeline_new("recorder");
    if (!app->pipeline) return FALSE;

    // 源与基础处理
    app->video_src = try_make(video_src_factory);
    app->audio_src = try_make(audio_src_factory);
    app->video_convert = try_make("videoconvert");
    app->audio_convert = try_make("audioconvert");
    app->audio_resample = try_make("audioresample");
    app->video_queue_preview = try_make("queue");
    app->video_queue_rec = try_make("queue");
    app->audio_queue_rec = try_make("queue");
    app->tee = try_make("tee");
    app->preview_sink = try_make("gtksink");

    if (!app->video_src || !app->audio_src || !app->video_convert || !app->audio_convert ||
        !app->audio_resample || !app->video_queue_preview || !app->video_queue_rec ||
        !app->audio_queue_rec || !app->tee || !app->preview_sink) {
        g_printerr("Failed to create basic elements.\n");
        return FALSE;
    }

    // 获取 gtksink 里的 widget 嵌入 GTK 窗口
    g_object_get(app->preview_sink, "widget", &app->video_widget, NULL);
    if (!app->video_widget) {
        g_printerr("gtksink has no widget.\n");
        return FALSE;
    }

    // 选择编码器与封装
    app->video_enc = choose_h264_encoder();
    app->audio_enc = choose_aac_encoder();
    app->mux = try_make("mp4mux");
    app->file_sink = try_make("filesink");

    if (!app->video_enc || !app->audio_enc || !app->mux || !app->file_sink) {
        g_printerr("H264/AAC encoder or mp4mux not available.\n");
        g_printerr("Please install gst-plugins-{base,good,bad,ugly,libav} as needed.\n");
        return FALSE;
    }

    // 设置一些合理的参数
#ifdef _WIN32
    // dshow 默认即可；如需指定设备可设置 "device-name"
#elif defined(__APPLE__)
    // avfvideosrc/avfaudiosrc 默认选择系统默认设备
#endif

    g_object_set(app->file_sink, "location", app->outfile, NULL);

    // 把所有元素加入管道
    gst_bin_add_many(GST_BIN(app->pipeline),
        app->video_src, app->video_convert, app->tee,
        app->video_queue_preview, app->preview_sink,
        app->video_queue_rec, app->video_enc,
        app->audio_src, app->audio_convert, app->audio_resample,
        app->audio_queue_rec, app->audio_enc,
        app->mux, app->file_sink,
        NULL);

    // 链接：视频源 → 转换 → tee
    if (!gst_element_link_many(app->video_src, app->video_convert, app->tee, NULL)) {
        g_printerr("Link video_src→videoconvert→tee failed.\n");
        return FALSE;
    }

    // tee 分两路：预览 与 录制
    GstPad *tee_src_preview = gst_element_get_request_pad(app->tee, "src_%u");
    GstPad *tee_src_record  = gst_element_get_request_pad(app->tee, "src_%u");
    GstPad *qprev_sink = gst_element_get_static_pad(app->video_queue_preview, "sink");
    GstPad *qrec_sink  = gst_element_get_static_pad(app->video_queue_rec, "sink");

    if (gst_pad_link(tee_src_preview, qprev_sink) != GST_PAD_LINK_OK ||
        gst_pad_link(tee_src_record,  qrec_sink)  != GST_PAD_LINK_OK) {
        g_printerr("Link tee branches failed.\n");
        return FALSE;
    }

    gst_object_unref(qprev_sink);
    gst_object_unref(qrec_sink);

    // 预览支路：queue → gtksink
    if (!gst_element_link_many(app->video_queue_preview, app->preview_sink, NULL)) {
        g_printerr("Link preview branch failed.\n");
        return FALSE;
    }

    // 录制视频支路：queue → h264enc → mp4mux
    if (!gst_element_link_many(app->video_queue_rec, app->video_enc, app->mux, NULL)) {
        g_printerr("Link video record branch failed.\n");
        return FALSE;
    }

    // 音频链路：audiosrc → convert → resample → queue → aacenc → mp4mux
    if (!gst_element_link_many(app->audio_src, app->audio_convert, app->audio_resample,
                               app->audio_queue_rec, app->audio_enc, app->mux, NULL)) {
        g_printerr("Link audio branch failed.\n");
        return FALSE;
    }

    // mp4mux → filesink（注意：通常 mp4 需要 EOS 才能正确写尾部）
    if (!gst_element_link(app->mux, app->file_sink)) {
        g_printerr("Link mux→filesink failed.\n");
        return FALSE;
    }

    // 监听 bus
    GstBus *bus = gst_element_get_bus(app->pipeline);
    gst_bus_add_watch(bus, bus_msg_cb, app);
    gst_object_unref(bus);

    return TRUE;
}

static void on_start(GtkButton *b, gpointer user_data) {
    (void)b;
    App *app = (App*)user_data;
    if (app->recording) return;

    // 让用户选择输出文件
    GtkFileDialog *dlg = gtk_file_dialog_new();
    gtk_file_dialog_set_initial_name(dlg, "output.mp4");

    gtk_file_dialog_save(dlg, GTK_WINDOW(app->window), NULL, (GAsyncReadyCallback)+[](GObject* obj, GAsyncResult* res, gpointer data){
        App *app = (App*)data;
        g_autoptr(GFile) file = gtk_file_dialog_save_finish(GTK_FILE_DIALOG(obj), res, NULL);
        if (!file) return; // 用户取消
        g_clear_pointer(&app->outfile, g_free);
        app->outfile = g_file_get_path(file);

        // 构建或更新 pipeline
        if (app->pipeline) {
            gst_element_set_state(app->pipeline, GST_STATE_NULL);
            gst_object_unref(app->pipeline);
            app->pipeline = NULL;
        }
        if (!build_pipeline(app)) {
            set_status(app, "Failed to build pipeline");
            return;
        }

        // 把预览 widget 加进窗口（首次/重建）
        if (!gtk_widget_get_parent(app->video_widget)) {
            // 用一个 Box 包住状态+按钮+视频
            GtkWidget *root = gtk_window_get_child(app->window);
            GtkWidget *box = GTK_IS_BOX(root) ? root : NULL;
            if (box) {
                // 最后一个 child 是占位的 scrolledwindow? 我们直接把 video_widget 设置为 box 的第三个
                gtk_box_append(GTK_BOX(box), app->video_widget);
            }
        }

        // 开始播放
        GstStateChangeReturn r = gst_element_set_state(app->pipeline, GST_STATE_PLAYING);
        if (r == GST_STATE_CHANGE_FAILURE) {
            set_status(app, "Failed to start (PLAYING)");
            return;
        }
        app->recording = TRUE;
        set_status(app, "Recording…");
        gtk_widget_set_sensitive(GTK_WIDGET(app->btn_start), FALSE);
        gtk_widget_set_sensitive(GTK_WIDGET(app->btn_stop), TRUE);
    }, app);
}

static void on_stop(GtkButton *b, gpointer user_data) {
    (void)b;
    App *app = (App*)user_data;
    if (!app->recording || !app->pipeline) return;

    // 发送 EOS，等待 bus 收到 EOS 再置 NULL
    gst_element_send_event(app->pipeline, gst_event_new_eos());
    set_status(app, "Stopping… (finalizing MP4)");
}

static void activate(GtkApplication *app_g, gpointer user_data) {
    App *app = (App*)user_data;

    app->window = GTK_WINDOW(gtk_application_window_new(app_g));
    gtk_window_set_title(app->window, "GTK4 + GStreamer Recorder");
    gtk_window_set_default_size(app->window, 960, 600);

    GtkWidget *box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
    gtk_window_set_child(app->window, box);

    GtkWidget *toolbar = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    gtk_box_append(GTK_BOX(box), toolbar);

    app->btn_start = GTK_BUTTON(gtk_button_new_with_label("Start"));
    app->btn_stop  = GTK_BUTTON(gtk_button_new_with_label("Stop"));
    app->status    = GTK_LABEL(gtk_label_new("Idle"));

    gtk_widget_set_sensitive(GTK_WIDGET(app->btn_stop), FALSE);

    gtk_box_append(GTK_BOX(toolbar), GTK_WIDGET(app->btn_start));
    gtk_box_append(GTK_BOX(toolbar), GTK_WIDGET(app->btn_stop));
    gtk_box_append(GTK_BOX(toolbar), GTK_WIDGET(app->status));

    g_signal_connect(app->btn_start, "clicked", G_CALLBACK(on_start), app);
    g_signal_connect(app->btn_stop,  "clicked", G_CALLBACK(on_stop),  app);

    // 先放一个占位控件，真正的 video_widget 会在 build_pipeline 后拿到
    GtkWidget *placeholder = gtk_label_new("Video preview will appear here after Start.");
    gtk_box_append(GTK_BOX(box), placeholder);

    gtk_window_present(app->window);
}

int main(int argc, char *argv[]) {
    // 初始化 GTK 与 GStreamer
    gst_init(&argc, &argv);

    App app = {0};
    GtkApplication *gtk_app = gtk_application_new("com.example.gtk4gstrecorder", G_APPLICATION_FLAGS_NONE);
    g_signal_connect(gtk_app, "activate", G_CALLBACK(activate), &app);

    int status = g_application_run(G_APPLICATION(gtk_app), argc, argv);

    if (app.pipeline) {
        gst_element_set_state(app.pipeline, GST_STATE_NULL);
        gst_object_unref(app.pipeline);
    }
    g_clear_pointer(&app.outfile, g_free);

    g_object_unref(gtk_app);
    return status;
}
```

---

## README.md（构建与运行）

````md
# GTK4 + GStreamer 跨平台录制示例

## 依赖
- GTK 4
- GStreamer 1.20+（建议完整安装 base/good/bad/ugly 以及 libav 插件，保证有 H264/AAC）

### macOS
- 建议安装：
  - `brew install gtk4`
  - GStreamer 官方 pkg（包含 dev & runtime）：https://gstreamer.freedesktop.org/download/
- 编译：
  ```bash
  mkdir build && cd build
  cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
  ninja
  open gtk4_gst_recorder.app
````

* 首次运行会弹出摄像头/麦克风权限对话框。

### Windows（MSYS2/MINGW64 推荐）

* 安装：

  ```bash
  pacman -S mingw-w64-x86_64-gtk4 \
            mingw-w64-x86_64-gstreamer \
            mingw-w64-x86_64-gst-plugins-base \
            mingw-w64-x86_64-gst-plugins-good \
            mingw-w64-x86_64-gst-plugins-bad \
            mingw-w64-x86_64-gst-plugins-ugly \
            mingw-w64-x86_64-gst-libav \
            mingw-w64-x86_64-cmake \
            mingw-w64-x86_64-ninja
  ```
* 构建：

  ```bash
  cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
  ninja
  ./gtk4_gst_recorder.exe
  ```

## 使用说明

1. 启动程序，点击 `Start`，选择保存为 `output.mp4` 的路径。
2. 程序会显示摄像头预览，同时录制摄像头+麦克风到 MP4。
3. 点击 `Stop`，发送 EOS，待文件封装完成后自动停止。

## 常见问题

* **提示找不到编码器/封装器**：

  * 请安装 `gst-plugins-ugly` 与 `gst-libav`，以获得 `x264enc` / `avenc_aac` 等。
  * 若仍不可用，可修改 `choose_h264_encoder/choose_aac_encoder` 或改用 `matroskamux + vp8enc + vorbisenc`。
* **没有预览**：检查是否成功创建 `gtksink`，或是否把其 `widget` 放入窗口。
* **停止后文件损坏**：确保使用 `Stop` 发送 EOS；直接强行退出可能导致 MP4 末尾未写入。

```

---

### 备注
- 这是“最小可用”的跨平台示例，实际项目中可加入设备枚举、分辨率/帧率选择、音量表、码率控制、文件名时间戳等。

```
