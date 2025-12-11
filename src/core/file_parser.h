#ifndef DUOROU_CORE_FILE_PARSER_H
#define DUOROU_CORE_FILE_PARSER_H

#include <string>
#include <vector>
#include <memory>

namespace duorou {
namespace core {

/**
 * Interface for file parsers.
 * Responsible for extracting text content from files.
 */
class FileParser {
public:
    virtual ~FileParser() = default;

    /**
     * Parse the file and extract text content.
     * @param file_path The path to the file.
     * @return The extracted text content.
     * @throws std::runtime_error if parsing fails.
     */
    virtual std::string parse(const std::string& file_path) = 0;

    /**
     * Check if the parser supports the given file extension.
     * @param extension The file extension (including dot, e.g., ".pdf").
     * @return True if supported, false otherwise.
     */
    virtual bool supports(const std::string& extension) = 0;
};

/**
 * Factory class to get the appropriate parser.
 */
class FileParserFactory {
public:
    static std::unique_ptr<FileParser> get_parser(const std::string& file_path);
};

} // namespace core
} // namespace duorou

#endif // DUOROU_CORE_FILE_PARSER_H
