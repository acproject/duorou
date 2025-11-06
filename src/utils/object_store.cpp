#include "object_store.h"

#include <glib.h>
#include <gio/gio.h>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <iomanip>

namespace duorou {
namespace utils {

static std::string join_path(const std::string &a, const std::string &b) {
  if (a.empty()) return b;
  if (a.back() == G_DIR_SEPARATOR) return a + b;
  return a + G_DIR_SEPARATOR_S + b;
}

std::string ObjectStore::objects_dir() {
  const char *home = g_get_home_dir();
  std::string base = home ? std::string(home) : std::string(".");
  std::string dir = join_path(base, ".duorou");
  std::string objects = join_path(dir, "objects");
  g_mkdir_with_parents(objects.c_str(), 0700);
  return objects;
}

static std::string file_extension(const std::string &path) {
  auto pos = path.find_last_of('.');
  if (pos == std::string::npos) return "";
  return path.substr(pos); // includes the dot
}

static std::string sha256_file(const std::string &path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs.is_open()) return "";

  GChecksum *checksum = g_checksum_new(G_CHECKSUM_SHA256);
  if (!checksum) return "";

  char buf[8192];
  while (ifs.good()) {
    ifs.read(buf, sizeof(buf));
    std::streamsize got = ifs.gcount();
    if (got > 0) {
      g_checksum_update(checksum, reinterpret_cast<const guchar *>(buf), (gssize)got);
    }
  }

  const gchar *hex = g_checksum_get_string(checksum);
  std::string result = hex ? std::string(hex) : std::string();
  g_checksum_free(checksum);
  return result;
}

std::string ObjectStore::store_file(const std::string &src_path) {
  if (src_path.empty()) return "";
  std::string id = sha256_file(src_path);
  if (id.empty()) {
    // Fallback: random UUID string
    char *uuid = g_uuid_string_random();
    id = uuid ? std::string(uuid) : std::string("unknown");
    if (uuid) g_free(uuid);
  }
  std::string ext = file_extension(src_path);
  std::string dest_dir = objects_dir();
  std::string dest_path = join_path(dest_dir, id + ext);

  // If file already exists (same hash), skip copy
  if (!g_file_test(dest_path.c_str(), G_FILE_TEST_EXISTS)) {
    GFile *src = g_file_new_for_path(src_path.c_str());
    GFile *dst = g_file_new_for_path(dest_path.c_str());
    GError *err = nullptr;
    g_file_copy(src, dst, G_FILE_COPY_NONE, nullptr, nullptr, nullptr, &err);
    if (err) {
      g_error_free(err);
    }
    if (src) g_object_unref(src);
    if (dst) g_object_unref(dst);
  }

  return dest_path;
}

std::string ObjectStore::to_file_uri(const std::string &path) {
  if (path.empty()) return "";
  GError *err = nullptr;
  char *uri = g_filename_to_uri(path.c_str(), nullptr, &err);
  if (uri) {
    std::string out(uri);
    g_free(uri);
    return out;
  }
  if (err) g_error_free(err);
  return std::string("file://") + path;
}

} // namespace utils
} // namespace duorou