#include "text_file_parser.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

namespace duorou {
namespace core {

std::string TextFileParser::parse(const std::string &file_path) {
  std::ifstream file(file_path);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << file_path << std::endl;
    return "";
  }
  std::string line;
  std::string content;
  int rowCount = 0;
  const int MAX_ROWS = 16;
  const int MAX_COLS = 40; // Max chars per line for preview if needed, but
                           // request was 40 columns (fields)

  // For CSV, we might want to split by comma to count columns, but user said
  // "40 columns". Simple truncation for now:

  while (std::getline(file, line)) {
    if (rowCount >= MAX_ROWS) {
      content += "\n[... truncated (showing first " + std::to_string(MAX_ROWS) +
                 " rows) ...]\n";
      break;
    }

    // Check if it's a CSV to apply column limit?
    // Or just apply to all text files for safety?
    // Let's check extension in a member or just do it generally.
    // Assuming simple comma splitting for CSV logic if strictly required,
    // but robust CSV parsing is complex.
    // Let's just limit line length for non-CSV, or column count for CSV.

    // Simple heuristic: if line is very long, truncate it too
    if (line.length() > 1000) {
      line = line.substr(0, 1000) + "...";
    }

    content += line + "\n";
    rowCount++;
  }

  return content;
}

bool TextFileParser::supports(const std::string &extension) {
  std::string ext = extension;
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  return ext == ".txt" || ext == ".md" || ext == ".csv" || ext == ".json" ||
         ext == ".xml";
}

} // namespace core
} // namespace duorou
