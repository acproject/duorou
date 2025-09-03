#ifndef DUOROU_EXTENSIONS_OLLAMA_MODELFILE_PARSER_H
#define DUOROU_EXTENSIONS_OLLAMA_MODELFILE_PARSER_H

#include <string>
#include <unordered_map>
#include <vector>

namespace duorou {
namespace extensions {
namespace ollama {

// Modelfile指令类型
enum class ModelfileInstruction {
    FROM,
    PARAMETER,
    TEMPLATE,
    SYSTEM,
    ADAPTER,
    LICENSE,
    MESSAGE,
    UNKNOWN
};

// Modelfile参数结构
struct ModelfileParameter {
    std::string name;
    std::string value;
    std::string type; // "string", "number", "boolean"
};

// Modelfile消息结构
struct ModelfileMessage {
    std::string role; // "system", "user", "assistant"
    std::string content;
};

// 解析后的Modelfile结构
struct ParsedModelfile {
    std::string from_model;  // FROM指令指定的基础模型
    std::string system_prompt;  // SYSTEM指令内容
    std::string template_content;  // TEMPLATE指令内容
    std::string license_content;  // LICENSE指令内容
    std::vector<std::string> adapters;  // ADAPTER指令列表
    std::vector<ModelfileParameter> parameters;  // PARAMETER指令列表
    std::vector<ModelfileMessage> messages;  // MESSAGE指令列表
    
    // 辅助方法
    bool hasParameter(const std::string& name) const;
    std::string getParameterValue(const std::string& name, const std::string& default_value = "") const;
    void setParameter(const std::string& name, const std::string& value, const std::string& type = "string");
};

// Ollama Modelfile解析器
class ModelfileParser {
public:
    ModelfileParser();
    ~ModelfileParser();

    // 解析Modelfile内容
    bool parseFromString(const std::string& modelfile_content, ParsedModelfile& result);
    bool parseFromFile(const std::string& modelfile_path, ParsedModelfile& result);
    
    // 生成Modelfile内容
    std::string generateModelfile(const ParsedModelfile& modelfile);
    
    // 验证Modelfile
    bool validateModelfile(const ParsedModelfile& modelfile, std::vector<std::string>& errors);
    
    // 设置选项
    void setVerbose(bool verbose) { verbose_ = verbose; }
    void setStrictMode(bool strict) { strict_mode_ = strict; }

private:
    // 解析单行指令
    bool parseLine(const std::string& line, ParsedModelfile& result);
    
    // 解析具体指令
    bool parseFromInstruction(const std::string& args, ParsedModelfile& result);
    bool parseParameterInstruction(const std::string& args, ParsedModelfile& result);
    bool parseTemplateInstruction(const std::string& args, ParsedModelfile& result);
    bool parseSystemInstruction(const std::string& args, ParsedModelfile& result);
    bool parseAdapterInstruction(const std::string& args, ParsedModelfile& result);
    bool parseLicenseInstruction(const std::string& args, ParsedModelfile& result);
    bool parseMessageInstruction(const std::string& args, ParsedModelfile& result);
    
    // 辅助方法
    ModelfileInstruction getInstructionType(const std::string& instruction);
    std::string trim(const std::string& str);
    std::string unquote(const std::string& str);
    std::vector<std::string> splitArgs(const std::string& args);
    std::string readFileContent(const std::string& file_path);
    void log(const std::string& level, const std::string& message);
    
    // 多行内容处理
    std::string parseMultilineContent(const std::vector<std::string>& lines, size_t& current_line);
    bool isMultilineStart(const std::string& content);
    bool isMultilineEnd(const std::string& content);

private:
    bool verbose_;
    bool strict_mode_;  // 严格模式，遇到未知指令时报错
    std::unordered_map<std::string, ModelfileInstruction> instruction_map_;
};

} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // DUOROU_EXTENSIONS_OLLAMA_MODELFILE_PARSER_H