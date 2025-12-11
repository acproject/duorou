#include "text_file_parser.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

namespace duorou {
namespace core {

std::string TextFileParser::parse(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

bool TextFileParser::supports(const std::string& extension) {
    std::string ext = extension;
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".txt" || ext == ".md" || ext == ".csv" || ext == ".json" || ext == ".xml";
}

} // namespace core
} // namespace duorou
