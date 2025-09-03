#include "modelfile_parser.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <cctype>

namespace duorou {
namespace extensions {
namespace ollama {

// ParsedModelfile 实现
bool ParsedModelfile::hasParameter(const std::string& name) const {
    for (const auto& param : parameters) {
        if (param.name == name) {
            return true;
        }
    }
    return false;
}

std::string ParsedModelfile::getParameterValue(const std::string& name, const std::string& default_value) const {
    for (const auto& param : parameters) {
        if (param.name == name) {
            return param.value;
        }
    }
    return default_value;
}

void ParsedModelfile::setParameter(const std::string& name, const std::string& value, const std::string& type) {
    // 查找是否已存在该参数
    for (auto& param : parameters) {
        if (param.name == name) {
            param.value = value;
            param.type = type;
            return;
        }
    }
    
    // 添加新参数
    parameters.push_back({name, value, type});
}

// ModelfileParser 实现
ModelfileParser::ModelfileParser() : verbose_(false), strict_mode_(false) {
    // 初始化指令映射
    instruction_map_["FROM"] = ModelfileInstruction::FROM;
    instruction_map_["PARAMETER"] = ModelfileInstruction::PARAMETER;
    instruction_map_["TEMPLATE"] = ModelfileInstruction::TEMPLATE;
    instruction_map_["SYSTEM"] = ModelfileInstruction::SYSTEM;
    instruction_map_["ADAPTER"] = ModelfileInstruction::ADAPTER;
    instruction_map_["LICENSE"] = ModelfileInstruction::LICENSE;
    instruction_map_["MESSAGE"] = ModelfileInstruction::MESSAGE;
}

ModelfileParser::~ModelfileParser() = default;

bool ModelfileParser::parseFromString(const std::string& modelfile_content, ParsedModelfile& result) {
    std::istringstream stream(modelfile_content);
    std::string line;
    
    // 清空结果
    result = ParsedModelfile{};
    
    while (std::getline(stream, line)) {
        line = trim(line);
        
        // 跳过空行和注释
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        if (!parseLine(line, result)) {
            if (strict_mode_) {
                return false;
            }
            // 非严格模式下继续解析
        }
    }
    
    return true;
}

bool ModelfileParser::parseFromFile(const std::string& modelfile_path, ParsedModelfile& result) {
    std::string content = readFileContent(modelfile_path);
    if (content.empty()) {
        if (verbose_) {
            log("ERROR", "Failed to read Modelfile: " + modelfile_path);
        }
        return false;
    }
    
    return parseFromString(content, result);
}

std::string ModelfileParser::generateModelfile(const ParsedModelfile& modelfile) {
    std::ostringstream oss;
    
    // FROM指令
    if (!modelfile.from_model.empty()) {
        oss << "FROM " << modelfile.from_model << "\n";
    }
    
    // PARAMETER指令
    for (const auto& param : modelfile.parameters) {
        oss << "PARAMETER " << param.name << " " << param.value << "\n";
    }
    
    // TEMPLATE指令
    if (!modelfile.template_content.empty()) {
        oss << "TEMPLATE \"\"\"\n" << modelfile.template_content << "\n\"\"\"\n";
    }
    
    // SYSTEM指令
    if (!modelfile.system_prompt.empty()) {
        oss << "SYSTEM \"\"\"\n" << modelfile.system_prompt << "\n\"\"\"\n";
    }
    
    // ADAPTER指令
    for (const auto& adapter : modelfile.adapters) {
        oss << "ADAPTER " << adapter << "\n";
    }
    
    // LICENSE指令
    if (!modelfile.license_content.empty()) {
        oss << "LICENSE \"\"\"\n" << modelfile.license_content << "\n\"\"\"\n";
    }
    
    // MESSAGE指令
    for (const auto& message : modelfile.messages) {
        oss << "MESSAGE " << message.role << " \"\"\"\n" << message.content << "\n\"\"\"\n";
    }
    
    return oss.str();
}

bool ModelfileParser::validateModelfile(const ParsedModelfile& modelfile, std::vector<std::string>& errors) {
    errors.clear();
    
    // 检查必需的FROM指令
    if (modelfile.from_model.empty()) {
        errors.push_back("Missing required FROM instruction");
    }
    
    // 验证参数
    for (const auto& param : modelfile.parameters) {
        if (param.name.empty()) {
            errors.push_back("Parameter name cannot be empty");
        }
    }
    
    // 验证消息
    for (const auto& message : modelfile.messages) {
        if (message.role != "system" && message.role != "user" && message.role != "assistant") {
            errors.push_back("Invalid message role: " + message.role);
        }
        if (message.content.empty()) {
            errors.push_back("Message content cannot be empty for role: " + message.role);
        }
    }
    
    return errors.empty();
}

bool ModelfileParser::parseLine(const std::string& line, ParsedModelfile& result) {
    // 查找第一个空格，分离指令和参数
    size_t space_pos = line.find(' ');
    if (space_pos == std::string::npos) {
        if (verbose_) {
            log("WARNING", "Invalid line format: " + line);
        }
        return false;
    }
    
    std::string instruction = line.substr(0, space_pos);
    std::string args = trim(line.substr(space_pos + 1));
    
    // 转换为大写
    std::transform(instruction.begin(), instruction.end(), instruction.begin(), ::toupper);
    
    ModelfileInstruction inst_type = getInstructionType(instruction);
    
    switch (inst_type) {
        case ModelfileInstruction::FROM:
            return parseFromInstruction(args, result);
        case ModelfileInstruction::PARAMETER:
            return parseParameterInstruction(args, result);
        case ModelfileInstruction::TEMPLATE:
            return parseTemplateInstruction(args, result);
        case ModelfileInstruction::SYSTEM:
            return parseSystemInstruction(args, result);
        case ModelfileInstruction::ADAPTER:
            return parseAdapterInstruction(args, result);
        case ModelfileInstruction::LICENSE:
            return parseLicenseInstruction(args, result);
        case ModelfileInstruction::MESSAGE:
            return parseMessageInstruction(args, result);
        default:
            if (verbose_) {
                log("WARNING", "Unknown instruction: " + instruction);
            }
            return !strict_mode_;
    }
}

bool ModelfileParser::parseFromInstruction(const std::string& args, ParsedModelfile& result) {
    result.from_model = unquote(trim(args));
    return !result.from_model.empty();
}

bool ModelfileParser::parseParameterInstruction(const std::string& args, ParsedModelfile& result) {
    std::vector<std::string> parts = splitArgs(args);
    if (parts.size() < 2) {
        if (verbose_) {
            log("ERROR", "PARAMETER instruction requires name and value");
        }
        return false;
    }
    
    std::string name = parts[0];
    std::string value = parts[1];
    
    // 检测值的类型
    std::string type = "string";
    if (value == "true" || value == "false") {
        type = "boolean";
    } else {
        // 尝试解析为数字
        char* end;
        std::strtod(value.c_str(), &end);
        if (*end == '\0') {
            type = "number";
        }
    }
    
    result.setParameter(name, unquote(value), type);
    return true;
}

bool ModelfileParser::parseTemplateInstruction(const std::string& args, ParsedModelfile& result) {
    result.template_content = unquote(trim(args));
    return true;
}

bool ModelfileParser::parseSystemInstruction(const std::string& args, ParsedModelfile& result) {
    result.system_prompt = unquote(trim(args));
    return true;
}

bool ModelfileParser::parseAdapterInstruction(const std::string& args, ParsedModelfile& result) {
    std::string adapter = unquote(trim(args));
    if (!adapter.empty()) {
        result.adapters.push_back(adapter);
        return true;
    }
    return false;
}

bool ModelfileParser::parseLicenseInstruction(const std::string& args, ParsedModelfile& result) {
    result.license_content = unquote(trim(args));
    return true;
}

bool ModelfileParser::parseMessageInstruction(const std::string& args, ParsedModelfile& result) {
    std::vector<std::string> parts = splitArgs(args);
    if (parts.size() < 2) {
        if (verbose_) {
            log("ERROR", "MESSAGE instruction requires role and content");
        }
        return false;
    }
    
    ModelfileMessage message;
    message.role = parts[0];
    message.content = unquote(parts[1]);
    
    result.messages.push_back(message);
    return true;
}

ModelfileInstruction ModelfileParser::getInstructionType(const std::string& instruction) {
    auto it = instruction_map_.find(instruction);
    return (it != instruction_map_.end()) ? it->second : ModelfileInstruction::UNKNOWN;
}

std::string ModelfileParser::trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return "";
    }
    
    size_t end = str.find_last_not_of(" \t\r\n");
    return str.substr(start, end - start + 1);
}

std::string ModelfileParser::unquote(const std::string& str) {
    std::string trimmed = trim(str);
    
    // 处理三引号字符串
    if (trimmed.length() >= 6 && 
        trimmed.substr(0, 3) == "\"\"\"" && 
        trimmed.substr(trimmed.length() - 3) == "\"\"\"") {
        return trimmed.substr(3, trimmed.length() - 6);
    }
    
    // 处理单引号字符串
    if (trimmed.length() >= 2 && 
        trimmed[0] == '"' && 
        trimmed[trimmed.length() - 1] == '"') {
        return trimmed.substr(1, trimmed.length() - 2);
    }
    
    return trimmed;
}

std::vector<std::string> ModelfileParser::splitArgs(const std::string& args) {
    std::vector<std::string> result;
    std::istringstream stream(args);
    std::string token;
    
    while (stream >> token) {
        result.push_back(token);
    }
    
    return result;
}

std::string ModelfileParser::readFileContent(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        return "";
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

void ModelfileParser::log(const std::string& level, const std::string& message) {
    if (verbose_) {
        std::cout << "[" << level << "] ModelfileParser: " << message << std::endl;
    }
}

std::string ModelfileParser::parseMultilineContent(const std::vector<std::string>& lines, size_t& current_line) {
    // 多行内容解析的实现
    // 这里可以处理三引号包围的多行内容
    return "";
}

bool ModelfileParser::isMultilineStart(const std::string& content) {
    std::string trimmed = trim(content);
    return trimmed.find("\"\"\"") != std::string::npos;
}

bool ModelfileParser::isMultilineEnd(const std::string& content) {
    std::string trimmed = trim(content);
    return trimmed.find("\"\"\"") != std::string::npos;
}

} // namespace ollama
} // namespace extensions
} // namespace duorou