// Simple file-based object store for attachments
// Stores files under ~/.duorou/objects and returns canonical local paths
// Note: focuses on images/documents selected via GUI; no networking/service.

#ifndef DUOROU_UTILS_OBJECT_STORE_H
#define DUOROU_UTILS_OBJECT_STORE_H

#ifdef __cplusplus
#include <string>

namespace duorou {
namespace utils {

class ObjectStore {
public:
  // Ensure the objects directory exists and return its absolute path
  static std::string objects_dir();

  // Store a file into the objects directory; returns stored absolute path
  // Uses SHA256(content) as object id and preserves extension
  static std::string store_file(const std::string &src_path);

  // Convert local absolute path to file:// URI
  static std::string to_file_uri(const std::string &path);
};

} // namespace utils
} // namespace duorou
#else
// When parsed by tools without C++ standard library or Gtk headers (e.g. C indexers),
// keep the header minimal to avoid false-positive diagnostics.
#endif

#endif // DUOROU_UTILS_OBJECT_STORE_H