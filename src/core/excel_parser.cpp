#include "excel_parser.h"
#include <OpenXLSX.hpp>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>

namespace duorou {
namespace core {

using namespace OpenXLSX;

std::string ExcelParser::parse(const std::string &file_path) {
  std::stringstream ss;
  XLDocument doc;

  try {
    doc.open(file_path);
    auto workbook = doc.workbook();
    auto sheetNames = workbook.worksheetNames();

    for (const auto &sheetName : sheetNames) {
      ss << "### Sheet: " << sheetName << "\n\n";
      auto wks = workbook.worksheet(sheetName);

      // Note: OpenXLSX row iteration
      bool firstRow = true;
      int columnCount = 0;

      // Limit rows to avoid huge output
      const int MAX_ROWS = 16;
      const int MAX_COLS = 40;
      int rowCount = 0;

      for (auto &row : wks.rows()) {
        if (rowCount >= MAX_ROWS) {
          ss << "\n[... truncated (showing first " + std::to_string(MAX_ROWS) +
                    " rows) ...]\n";
          break;
        }

        std::stringstream rowSS;
        rowSS << "|";

        int currentCols = 0;
        // cell iteration
        // OpenXLSX cells() returns a range of cells in the row
        for (auto &cell : row.cells()) {
          if (currentCols >= MAX_COLS) {
            rowSS << " ... |";
            break;
          }
          rowSS << " " << cell.value() << " |";
          currentCols++;
        }

        // If empty row, skip or just print empty
        if (currentCols == 0)
          continue;

        ss << rowSS.str() << "\n";

        if (firstRow) {
          columnCount = currentCols;
          ss << "|";
          for (int i = 0; i < columnCount; ++i)
            ss << " --- |";
          ss << "\n";
          firstRow = false;
        }
        rowCount++;
      }
      ss << "\n";
    }
  } catch (const std::exception &e) {
    std::cerr << "Excel parsing error: " << e.what() << std::endl;
    return "[Error parsing Excel file: " + std::string(e.what()) + "]";
  }

  return ss.str();
}

bool ExcelParser::supports(const std::string &extension) {
  std::string ext = extension;
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  return ext == ".xlsx";
}

} // namespace core
} // namespace duorou
