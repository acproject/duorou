#include "file_parser.h"
#include "pdf_parser.h"
#include "text_file_parser.h"
#include <algorithm>
#include <filesystem>

namespace duorou {
namespace core {

std::unique_ptr<FileParser> FileParserFactory::get_parser(const std::string& file_path) {
    std::filesystem::path path(file_path);
    std::string extension = path.extension().string();
    
    // Normalize extension
    std::string lower_ext = extension;
    std::transform(lower_ext.begin(), lower_ext.end(), lower_ext.begin(), ::tolower);

    if (lower_ext == ".pdf") {
        return std::make_unique<PdfParser>();
    } else if (lower_ext == ".txt" || lower_ext == ".md" || lower_ext == ".csv" || lower_ext == ".json") {
        return std::make_unique<TextFileParser>();
    }
    
    return nullptr;
}

} // namespace core
} // namespace duorou
