#include "pdf_parser.h"
#include "../utils/object_store.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <regex>
#include <sstream>
#include <vector>

// Include TextExtraction header
// Assuming the include path is set correctly by CMake
#include "TextExtraction.h"

namespace {

std::string generate_pdf_ocr_images_markdown(const std::string &file_path) {
#ifdef _WIN32
  (void)file_path;
  return "";
#else
  namespace fs = std::filesystem;

  std::string objects_dir = duorou::utils::ObjectStore::objects_dir();
  if (objects_dir.empty()) {
    return "";
  }

  fs::path out_dir(objects_dir);
  fs::path pdf_path(file_path);
  std::string stem = pdf_path.stem().string();

  auto now = std::chrono::system_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch())
                .count();

  std::string prefix = stem + "_page_" + std::to_string(ms);
  fs::path prefix_path = out_dir / prefix;

  std::string cmd = "pdftoppm -png \"" + file_path + "\" \"" +
                    prefix_path.string() + "\" > /dev/null 2>&1";
  int ret = std::system(cmd.c_str());
  if (ret != 0) {
    std::cerr << "pdftoppm failed for file: " << file_path << " code=" << ret
              << std::endl;
    return "";
  }

  std::vector<fs::path> images;
  for (const auto &entry : fs::directory_iterator(out_dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const fs::path &p = entry.path();
    if (p.extension() != ".png") {
      continue;
    }
    std::string name = p.stem().string();
    std::string expected_prefix = prefix + "-";
    if (name.rfind(expected_prefix, 0) == 0) {
      images.push_back(p);
    }
  }

  if (images.empty()) {
    return "";
  }

  std::sort(images.begin(), images.end());

  const size_t max_pages = 8;
  std::ostringstream oss;
  oss << "[PDF OCR] Generated page images for multimodal model:\n";

  size_t page_index = 0;
  for (const auto &img : images) {
    if (page_index >= max_pages) {
      break;
    }
    std::string uri = duorou::utils::ObjectStore::to_file_uri(img.string());
    ++page_index;
    oss << "![" << "page " << page_index << "](" << uri << ")\n";
  }

  return oss.str();
#endif
}

} // namespace

namespace duorou {
namespace core {

std::string PdfParser::parse(const std::string &file_path) {
  std::cout << "Starting PDF parsing: " << file_path << std::endl;
  TextExtraction extractor;
  PDFHummus::EStatusCode status = extractor.ExtractText(file_path);

  if (status != PDFHummus::eSuccess) {
    std::cerr << "Failed to extract text from PDF: " << file_path << std::endl;
    return "[PDF parsing failed: unable to extract text content from file. "
           "The file may be encrypted, corrupted, or an unsupported format.]";
  }

  std::stringstream ss;
  extractor.GetResultsAsText(0, TextComposer::eSpacingBoth, ss);

  std::string raw_text = ss.str();
  std::cout << "Raw PDF text extracted, length: " << raw_text.length()
            << std::endl;

  if (raw_text.empty()) {
    std::cerr << "Warning: Extracted PDF text is empty. The file might be "
                 "scanned (images only) or protected."
              << std::endl;

    std::string ocr_md = generate_pdf_ocr_images_markdown(file_path);
    if (!ocr_md.empty()) {
      return ocr_md;
    }

    return "[PDF parsing failed: No text content found. The file might be a "
           "scanned image.]";
  }

  std::regex multiple_newlines("[\r\n]{3,}");
  std::string cleaned = std::regex_replace(raw_text, multiple_newlines, "\n\n");

  size_t first = cleaned.find_first_not_of(" \t\r\n");
  if (std::string::npos == first) {
    std::cerr << "Warning: Cleaned PDF text is whitespace only, trying OCR for "
                 "images."
              << std::endl;

    std::string ocr_md = generate_pdf_ocr_images_markdown(file_path);
    if (!ocr_md.empty()) {
      return ocr_md;
    }

    return "[PDF parsing warning: extracted content contains only whitespace "
           "or non-text elements.]";
  }
  size_t last = cleaned.find_last_not_of(" \t\r\n");
  cleaned = cleaned.substr(first, (last - first + 1));

  std::cout << "Cleaned PDF text length: " << cleaned.length() << std::endl;
  return cleaned;
}

bool PdfParser::supports(const std::string &extension) {
  std::string ext = extension;
  for (auto &c : ext)
    c = tolower(c);
  return ext == ".pdf";
}

} // namespace core
} // namespace duorou
