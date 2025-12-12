#ifndef DUOROU_CORE_EXCEL_PARSER_H
#define DUOROU_CORE_EXCEL_PARSER_H

#include "file_parser.h"

namespace duorou {
namespace core {

class ExcelParser : public FileParser {
public:
    std::string parse(const std::string& file_path) override;
    bool supports(const std::string& extension) override;
};

} // namespace core
} // namespace duorou

#endif // DUOROU_CORE_EXCEL_PARSER_H
