#include "pdf_parser.h"
#include <iostream>
#include <regex>
#include <sstream>

// Include TextExtraction header
// Assuming the include path is set correctly by CMake
#include "TextExtraction.h"

namespace duorou {
namespace core {

std::string PdfParser::parse(const std::string &file_path) {
  std::cout << "Starting PDF parsing: " << file_path << std::endl;
  TextExtraction extractor;
  PDFHummus::EStatusCode status = extractor.ExtractText(file_path);

  if (status != PDFHummus::eSuccess) {
    std::cerr << "Failed to extract text from PDF: " << file_path << std::endl;
    return "";
  }

  std::stringstream ss;
  // bidiFlag=0 (assuming LTR default or no special handling),
  // spacingFlag=kSpacingBoth
  extractor.GetResultsAsText(0, TextComposer::eSpacingBoth, ss);

  std::string raw_text = ss.str();
  std::cout << "Raw PDF text extracted, length: " << raw_text.length()
            << std::endl;

  if (raw_text.empty()) {
    std::cerr << "Warning: Extracted PDF text is empty. The file might be "
                 "scanned (images only) or protected."
              << std::endl;
    return "[PDF parsing failed: No text content found. The file might be a "
           "scanned image.]";
  }

  // Clean up text: replace multiple newlines with a single newline to save
  // context space and remove excessive whitespace.
  // 1. Replace multiple newlines with double newline (paragraph break)
  std::regex multiple_newlines("[\r\n]{3,}");
  std::string cleaned = std::regex_replace(raw_text, multiple_newlines, "\n\n");

  // 2. Trim leading/trailing whitespace (simple implementation)
  size_t first = cleaned.find_first_not_of(" \t\r\n");
  if (std::string::npos == first) {
    return "";
  }
  size_t last = cleaned.find_last_not_of(" \t\r\n");
  cleaned = cleaned.substr(first, (last - first + 1));

  std::cout << "Cleaned PDF text length: " << cleaned.length() << std::endl;
  return cleaned;
}

bool PdfParser::supports(const std::string &extension) {
  // Simple check for .pdf (case insensitive usually, but here strict for now)
  // In a real app, use case-insensitive comparison
  std::string ext = extension;
  // Convert to lower case
  for (auto &c : ext)
    c = tolower(c);
  return ext == ".pdf";
}

} // namespace core
} // namespace duorou
