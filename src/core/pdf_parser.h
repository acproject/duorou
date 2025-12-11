#ifndef DUOROU_CORE_PDF_PARSER_H
#define DUOROU_CORE_PDF_PARSER_H

#include "file_parser.h"

namespace duorou {
namespace core {

class PdfParser : public FileParser {
public:
    std::string parse(const std::string& file_path) override;
    bool supports(const std::string& extension) override;
};

} // namespace core
} // namespace duorou

#endif // DUOROU_CORE_PDF_PARSER_H
