#include "modelfile_parser.h"
#include "../../third_party/llama.cpp/vendor/nlohmann/json.hpp"
#include "logger.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>

namespace duorou {
namespace core {

// Supported media type constants
static const std::vector<std::string> SUPPORTED_MEDIA_TYPES = {
    "application/vnd.ollama.image.model",
    "application/vnd.ollama.image.template",
    "application/vnd.ollama.image.system",
    "application/vnd.ollama.image.params",
    "application/vnd.ollama.image.adapter",
    "application/vnd.docker.image.rootfs.diff.tar.gzip"};

ModelfileParser::ModelfileParser(
    std::shared_ptr<ModelPathManager> model_path_manager)
    : model_path_manager_(model_path_manager) {}

bool ModelfileParser::parseFromManifest(const ModelManifest &manifest,
                                        ModelfileConfig &config) {
  if (!model_path_manager_) {
    return false;
  }

  // Iterate through all layers and parse different types of configurations
  for (const auto &layer : manifest.layers) {
    if (layer.media_type == "application/vnd.ollama.image.template") {
      parseTemplateLayer(layer.digest, config);
    } else if (layer.media_type == "application/vnd.ollama.image.system") {
      parseSystemLayer(layer.digest, config);
    } else if (layer.media_type == "application/vnd.ollama.image.params") {
      parseParametersLayer(layer.digest, config);
    } else if (layer.media_type == "application/vnd.ollama.image.adapter") {
      parseAdapterLayer(layer.digest, config);
    } else if (layer.media_type == "application/vnd.ollama.image.model" ||
               layer.media_type ==
                   "application/vnd.docker.image.rootfs.diff.tar.gzip") {
      // Base model path
      config.base_model = model_path_manager_->getBlobFilePath(layer.digest);
    }
  }

  return true;
}

bool ModelfileParser::parseFromJson(const std::string &json_str,
                                    ModelfileConfig &config) {
  try {
    nlohmann::json json_data = nlohmann::json::parse(json_str, nullptr, /*allow_exceptions=*/false);
    if (json_data.is_discarded()) {
      return false;
    }

    // Parse base model
    if (json_data.contains("base_model")) {
      config.base_model = json_data["base_model"].get<std::string>();
    }

    // Parse LoRA adapters
    if (json_data.contains("adapters") && json_data["adapters"].is_array()) {
      for (const auto &adapter_json : json_data["adapters"]) {
        LoRAAdapter adapter;
        if (adapter_json.contains("name")) {
          adapter.name = adapter_json["name"].get<std::string>();
        }
        if (adapter_json.contains("path")) {
          adapter.path = adapter_json["path"].get<std::string>();
        }
        if (adapter_json.contains("scale")) {
          adapter.scale = adapter_json["scale"].get<float>();
        }
        if (adapter_json.contains("digest")) {
          adapter.digest = adapter_json["digest"].get<std::string>();
        }
        if (adapter_json.contains("size")) {
          adapter.size = adapter_json["size"].get<size_t>();
        }
        config.lora_adapters.push_back(adapter);
      }
    }

    // Parse parameters
    if (json_data.contains("parameters") &&
        json_data["parameters"].is_object()) {
      for (auto &[key, value] : json_data["parameters"].items()) {
        config.parameters[key] = value.get<std::string>();
      }
    }

    // Parse system prompt and template
    if (json_data.contains("system_prompt")) {
      config.system_prompt = json_data["system_prompt"].get<std::string>();
    }
    if (json_data.contains("template_format")) {
      config.template_format = json_data["template_format"].get<std::string>();
    }

    return true;
  } catch (const std::exception &e) {
    return false;
  }
}

bool ModelfileParser::parseFromFile(const std::string &file_path,
                                    ModelfileConfig &config) {
  std::ifstream file(file_path);
  if (!file.is_open()) {
    return false;
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string content = buffer.str();

  // Try to parse as JSON
  if (content.front() == '{' && content.back() == '}') {
    return parseFromJson(content, config);
  }

  // Parse as Modelfile instruction format
  return parseModelfileInstructions(content, config);
}

bool ModelfileParser::validateLoRAAdapter(const LoRAAdapter &adapter) {
  if (adapter.path.empty()) {
    return false;
  }

  // Check if file exists
  if (!std::filesystem::exists(adapter.path)) {
    return false;
  }

  // Check file extension (should be .gguf)
  std::filesystem::path path(adapter.path);
  if (path.extension() != ".gguf") {
    return false;
  }

  // Check scaling factor range
  if (adapter.scale <= 0.0f || adapter.scale > 10.0f) {
    return false;
  }

  // Check file size (LoRA files are usually small)
  std::error_code ec;
  auto file_size = std::filesystem::file_size(adapter.path, ec);
  if (ec) {
    return false;
  }

  // LoRA files are usually between a few MB to several hundred MB
  const size_t MIN_LORA_SIZE = 1024 * 1024;               // 1MB
  const size_t MAX_LORA_SIZE = 2ULL * 1024 * 1024 * 1024; // 2GB
  if (file_size < MIN_LORA_SIZE || file_size > MAX_LORA_SIZE) {
    return false;
  }

  // Validate GGUF file header
  if (!validateGGUFHeader(adapter.path)) {
    return false;
  }

  return true;
}

std::vector<std::string> ModelfileParser::getSupportedMediaTypes() {
  return SUPPORTED_MEDIA_TYPES;
}

bool ModelfileParser::parseTemplateLayer(const std::string &layer_digest,
                                         ModelfileConfig &config) {
  std::string content = readBlobContent(layer_digest);
  if (content.empty()) {
    return false;
  }

  config.template_format = content;
  return true;
}

bool ModelfileParser::parseSystemLayer(const std::string &layer_digest,
                                       ModelfileConfig &config) {
  std::string content = readBlobContent(layer_digest);
  if (content.empty()) {
    return false;
  }

  config.system_prompt = content;
  return true;
}

bool ModelfileParser::parseParametersLayer(const std::string &layer_digest,
                                           ModelfileConfig &config) {
  std::string content = readBlobContent(layer_digest);
  if (content.empty()) {
    return false;
  }

  // Parse parameters (may be JSON format or key-value pair format)
  try {
    nlohmann::json params = nlohmann::json::parse(content, nullptr, /*allow_exceptions=*/false);
    if (params.is_object()) {
      for (auto &[key, value] : params.items()) {
        config.parameters[key] = value.get<std::string>();
      }
      return true;
    }
  } catch (const std::exception &) {
    // If not JSON, try to parse as key-value pairs
  }

  // Parse key-value pair format
  std::istringstream iss(content);
  std::string line;
  while (std::getline(iss, line)) {
    size_t pos = line.find('=');
    if (pos != std::string::npos) {
      std::string key = line.substr(0, pos);
      std::string value = line.substr(pos + 1);
      // Remove leading and trailing spaces
      key.erase(0, key.find_first_not_of(" \t"));
      key.erase(key.find_last_not_of(" \t") + 1);
      value.erase(0, value.find_first_not_of(" \t"));
      value.erase(value.find_last_not_of(" \t") + 1);
      config.parameters[key] = value;
    }
  }

  return true;
}

bool ModelfileParser::parseAdapterLayer(const std::string &layer_digest,
                                        ModelfileConfig &config) {
  std::string content = readBlobContent(layer_digest);
  if (content.empty()) {
    return false;
  }

  // Parse adapter information (may be JSON format or Modelfile instruction
  // format)
  try {
    nlohmann::json adapter_json = nlohmann::json::parse(content, nullptr, /*allow_exceptions=*/false);
    if (adapter_json.is_object()) {
      LoRAAdapter adapter;
      if (adapter_json.contains("name")) {
        adapter.name = adapter_json["name"].get<std::string>();
      }
      if (adapter_json.contains("path")) {
        adapter.path = adapter_json["path"].get<std::string>();
      }
      if (adapter_json.contains("scale")) {
        adapter.scale = adapter_json["scale"].get<float>();
      }
      adapter.digest = layer_digest;
      config.lora_adapters.push_back(adapter);
      return true;
    }
  } catch (const std::exception &) {
    // If not JSON, try to parse as Modelfile instructions
  }

  // Parse Modelfile instruction format
  return parseModelfileInstructions(content, config);
}

std::string ModelfileParser::readBlobContent(const std::string &digest) {
  if (!model_path_manager_) {
    return "";
  }

  std::string blob_path = model_path_manager_->getBlobFilePath(digest);
  if (!std::filesystem::exists(blob_path)) {
    return "";
  }

  std::ifstream file(blob_path, std::ios::binary);
  if (!file.is_open()) {
    return "";
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

bool ModelfileParser::parseModelfileInstructions(const std::string &content,
                                                 ModelfileConfig &config) {
  std::istringstream iss(content);
  std::string line;

  while (std::getline(iss, line)) {
    // Remove leading and trailing spaces
    line.erase(0, line.find_first_not_of(" \t"));
    line.erase(line.find_last_not_of(" \t") + 1);

    // Skip empty lines and comments
    if (line.empty() || line[0] == '#') {
      continue;
    }

    // Convert to uppercase for instruction matching
    std::string upper_line = line;
    std::transform(upper_line.begin(), upper_line.end(), upper_line.begin(),
                   ::toupper);

    if (upper_line.substr(0, 5) == "FROM ") {
      parseFromInstruction(line, config);
    } else if (upper_line.substr(0, 8) == "ADAPTER ") {
      parseAdapterInstruction(line, config);
    } else if (upper_line.substr(0, 10) == "PARAMETER ") {
      parseParameterInstruction(line, config);
    } else if (upper_line.substr(0, 9) == "TEMPLATE ") {
      parseTemplateInstruction(line, config);
    } else if (upper_line.substr(0, 7) == "SYSTEM ") {
      parseSystemInstruction(line, config);
    }
  }

  return true;
}

bool ModelfileParser::parseFromInstruction(const std::string &line,
                                           ModelfileConfig &config) {
  std::regex from_regex(R"(^FROM\s+(.+)$)", std::regex_constants::icase);
  std::smatch match;

  if (std::regex_match(line, match, from_regex)) {
    config.base_model = match[1].str();
    return true;
  }

  return false;
}

bool ModelfileParser::parseAdapterInstruction(const std::string &line,
                                              ModelfileConfig &config) {
  std::regex adapter_regex(R"(^ADAPTER\s+(\S+)(?:\s+(.+))?$)",
                           std::regex_constants::icase);
  std::smatch match;

  if (std::regex_match(line, match, adapter_regex)) {
    LoRAAdapter adapter;
    adapter.path = match[1].str();

    // Parse optional parameters (such as scaling factor)
    if (match.size() > 2 && !match[2].str().empty()) {
      std::string params = match[2].str();
      std::regex scale_regex(R"(scale=([0-9]*\.?[0-9]+))",
                             std::regex_constants::icase);
      std::smatch scale_match;
      if (std::regex_search(params, scale_match, scale_regex)) {
        adapter.scale = std::stof(scale_match[1].str());
      }

      std::regex name_regex(R"(name=([^\s]+))", std::regex_constants::icase);
      std::smatch name_match;
      if (std::regex_search(params, name_match, name_regex)) {
        adapter.name = name_match[1].str();
      }
    }

    // If no name is specified, use the filename
    if (adapter.name.empty()) {
      std::filesystem::path path(adapter.path);
      adapter.name = path.stem().string();
    }

    config.lora_adapters.push_back(adapter);
    return true;
  }

  return false;
}

bool ModelfileParser::parseParameterInstruction(const std::string &line,
                                                ModelfileConfig &config) {
  std::regex param_regex(R"(^PARAMETER\s+(\S+)\s+(.+)$)",
                         std::regex_constants::icase);
  std::smatch match;

  if (std::regex_match(line, match, param_regex)) {
    std::string key = match[1].str();
    std::string value = match[2].str();

    // Remove quotes
    if ((value.front() == '"' && value.back() == '"') ||
        (value.front() == '\'' && value.back() == '\'')) {
      value = value.substr(1, value.length() - 2);
    }

    config.parameters[key] = value;
    return true;
  }

  return false;
}

bool ModelfileParser::parseTemplateInstruction(const std::string &line,
                                               ModelfileConfig &config) {
  std::regex template_regex(R"(^TEMPLATE\s+(.+)$)",
                            std::regex_constants::icase);
  std::smatch match;

  if (std::regex_match(line, match, template_regex)) {
    std::string template_str = match[1].str();

    // Remove quotes
    if ((template_str.front() == '"' && template_str.back() == '"') ||
        (template_str.front() == '\'' && template_str.back() == '\'')) {
      template_str = template_str.substr(1, template_str.length() - 2);
    }

    config.template_format = template_str;
    return true;
  }

  return false;
}

bool ModelfileParser::parseSystemInstruction(const std::string &line,
                                             ModelfileConfig &config) {
  std::regex system_regex(R"(^SYSTEM\s+(.+)$)", std::regex_constants::icase);
  std::smatch match;

  if (std::regex_match(line, match, system_regex)) {
    std::string system_str = match[1].str();

    // Remove quotes
    if ((system_str.front() == '"' && system_str.back() == '"') ||
        (system_str.front() == '\'' && system_str.back() == '\'')) {
      system_str = system_str.substr(1, system_str.length() - 2);
    }

    config.system_prompt = system_str;
    return true;
  }

  return true;
}

bool ModelfileParser::validateGGUFHeader(const std::string &file_path) {
  std::ifstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    return false;
  }

  // GGUF file header magic number is "GGUF"
  char magic[4];
  file.read(magic, 4);
  if (file.gcount() != 4) {
    return false;
  }

  // Check magic number
  if (magic[0] != 'G' || magic[1] != 'G' || magic[2] != 'U' ||
      magic[3] != 'F') {
    return false;
  }

  // Read version number
  uint32_t version;
  file.read(reinterpret_cast<char *>(&version), sizeof(version));
  if (file.gcount() != sizeof(version)) {
    return false;
  }

  // Supported GGUF version (usually 3 or higher)
  if (version < 3) {
    return false;
  }

  // Read tensor count
  uint64_t tensor_count;
  file.read(reinterpret_cast<char *>(&tensor_count), sizeof(tensor_count));
  if (file.gcount() != sizeof(tensor_count)) {
    return false;
  }

  // LoRA files should have a reasonable number of tensors (usually tens to
  // hundreds)
  if (tensor_count == 0 || tensor_count > 10000) {
    return false;
  }

  // Read metadata key-value pair count
  uint64_t metadata_kv_count;
  file.read(reinterpret_cast<char *>(&metadata_kv_count),
            sizeof(metadata_kv_count));
  if (file.gcount() != sizeof(metadata_kv_count)) {
    return false;
  }

  // Metadata count should be reasonable
  if (metadata_kv_count > 1000) {
    return false;
  }

  return true;
}

} // namespace core
} // namespace duorou