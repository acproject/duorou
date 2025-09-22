#include "model.h"
#include "byte_pair_encoding.h"
#include "sentence_piece.h"
#include "tokenizer_factory.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>

namespace duorou {
namespace model {

// BaseModel implementation
BaseModel::BaseModel()
    : loaded_(false), modelName_("BaseModel"), modelVersion_("1.0") {}

BaseModel::~BaseModel() { unload(); }

bool BaseModel::load(const std::string &modelPath) {
  if (loaded_) {
    unload();
  }

  modelPath_ = modelPath;

  // Detect GGUF model file: either modelPath itself is a .gguf file, or a .gguf
  // inside the directory
  auto findGGUF = [](const std::string &path) -> std::string {
    namespace fs = std::filesystem;
    try {
      fs::path p(path);
      if (fs::exists(p) && fs::is_regular_file(p) && p.extension() == ".gguf") {
        return p.string();
      }
      if (fs::exists(p) && fs::is_directory(p)) {
        for (const auto &entry : fs::directory_iterator(p)) {
          if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
            return entry.path().string();
          }
        }
      }
    } catch (...) {
      // ignore filesystem errors and treat as not found
    }
    return std::string();
  };

  const std::string ggufFile = findGGUF(modelPath);
  if (!ggufFile.empty()) {
    // Load tokenizer/vocabulary directly from GGUF metadata
    try {
      duorou::extensions::ollama::GGUFParser parser(/*verbose=*/true);
      if (!parser.parseFile(ggufFile)) {
        return false;
      }

      // Build vocabulary from GGUF
      std::vector<std::string> tokens;
      if (const auto *kvTokens = parser.getMetadata("tokenizer.ggml.tokens")) {
        tokens = kvTokens->asStringArray();
      }

      std::vector<int32_t> types;
      if (const auto *kvTypes =
              parser.getMetadata("tokenizer.ggml.token_type")) {
        types = kvTypes->asInt32Array();
      }
      if (types.empty() && !tokens.empty()) {
        types.assign(tokens.size(), duorou::model::TOKEN_TYPE_NORMAL);
      }

      std::vector<std::string> merges;
      if (const auto *kvMerges = parser.getMetadata("tokenizer.ggml.merges")) {
        merges = kvMerges->asStringArray();
      }

      if (!tokens.empty()) {
        vocabulary_ = std::make_shared<Vocabulary>();
        vocabulary_->initialize(tokens, types, /*scores*/ {}, merges);

        // BOS/EOS configuration
        std::vector<int32_t> bos_ids;
        std::vector<int32_t> eos_ids;
        bool add_bos = false;
        bool add_eos = false;
        if (const auto *kvBOS =
                parser.getMetadata("tokenizer.ggml.bos_token_id")) {
          bos_ids.push_back(kvBOS->asInt32());
        }
        if (const auto *kvEOS =
                parser.getMetadata("tokenizer.ggml.eos_token_id")) {
          eos_ids.push_back(kvEOS->asInt32());
        }
        if (const auto *kvAddBOS =
                parser.getMetadata("tokenizer.ggml.add_bos_token")) {
          add_bos = kvAddBOS->asBool();
        }
        if (const auto *kvAddEOS =
                parser.getMetadata("tokenizer.ggml.add_eos_token")) {
          add_eos = kvAddEOS->asBool();
        }
        if (!bos_ids.empty()) {
          vocabulary_->setBOS(bos_ids, add_bos);
        }
        if (!eos_ids.empty()) {
          vocabulary_->setEOS(eos_ids, add_eos);
        }

        // Create tokenizer from GGUF metadata
        TokenizerFactoryOptions opts; // allow env/override
        tokenizer_ = createTextProcessorFromGGUF(parser, vocabulary_, opts);
        config_.vocab_size = vocabulary_->size();

        // Infer architecture if available
        std::string arch = parser.getArchitecture().name;
        if (arch.empty()) {
          if (const auto *kvArch = parser.getMetadata("general.architecture")) {
            arch = kvArch->asString();
          }
        }
        if (!arch.empty()) {
          config_.architecture = arch;
        }

        loaded_ = true;
        return true;
      }
      // If GGUF has no tokens, fallback to directory-based loading below
    } catch (const std::exception &) {
      // If any exception occurs, fallback to directory-based loading below
    }
  }

  // Fallback to directory-based layout: config.json, tokenizer/, model.bin
  // Load configuration
  if (!loadConfig(modelPath + "/config.json")) {
    return false;
  }

  // Load tokenizer (vocab/merges files)
  if (!loadTokenizer(modelPath + "/tokenizer")) {
    return false;
  }

  // Load model weights (dummy)
  if (!loadModel(modelPath + "/model.bin")) {
    return false;
  }

  loaded_ = true;
  return true;
}

bool BaseModel::isLoaded() const { return loaded_; }

void BaseModel::unload() {
  tokenizer_.reset();
  vocabulary_.reset();
  loaded_ = false;
  modelPath_.clear();
  metadata_.clear();
}

std::vector<int32_t> BaseModel::encode(const std::string &text,
                                       bool addSpecial) {
  if (!tokenizer_) {
    return {};
  }
  return tokenizer_->encode(text, addSpecial);
}

std::string BaseModel::decode(const std::vector<int32_t> &tokens) {
  if (!tokenizer_) {
    return "";
  }
  return tokenizer_->decode(tokens);
}

std::vector<int32_t> BaseModel::generate(const std::vector<int32_t> &prompt,
                                         size_t maxTokens) {
  if (!loaded_ || !tokenizer_) {
    return {};
  }

  std::vector<int32_t> result = prompt;

  for (size_t i = 0; i < maxTokens; ++i) {
    // Limit context to model's context length
    std::vector<int32_t> context = result;
    if (context.size() > config_.context_length) {
      context = std::vector<int32_t>(result.end() - config_.context_length,
                                     result.end());
    }

    int32_t nextToken = sampleNext(context);
    if (nextToken < 0) {
      break;
    }

    result.push_back(nextToken);

    // Check for end-of-sequence token
    if (tokenizer_->isSpecial(nextToken, Special::EOS)) {
      break;
    }
  }

  return result;
}

std::string BaseModel::generateText(const std::string &prompt,
                                    size_t maxTokens) {
  auto promptTokens = encode(prompt, true);
  auto generatedTokens = generate(promptTokens, maxTokens);
  return decode(generatedTokens);
}

const ModelConfig &BaseModel::getConfig() const { return config_; }

const TextProcessor *BaseModel::getTokenizer() const {
  return tokenizer_.get();
}

size_t BaseModel::getVocabSize() const { return config_.vocab_size; }

size_t BaseModel::getContextLength() const { return config_.context_length; }

std::string BaseModel::getModelName() const { return modelName_; }

std::string BaseModel::getModelVersion() const { return modelVersion_; }

std::map<std::string, std::string> BaseModel::getMetadata() const {
  return metadata_;
}

bool BaseModel::loadModel(const std::string &modelPath) {
  // Basic implementation - just check if file exists
  std::ifstream file(modelPath, std::ios::binary);
  if (!file.is_open()) {
    return false;
  }

  // In a real implementation, this would load model weights
  // For now, just verify the file exists
  file.close();
  return true;
}

bool BaseModel::loadTokenizer(const std::string &tokenizerPath) {
  // Try to load vocabulary
  vocabulary_ = std::make_shared<Vocabulary>();

  // Load vocabulary from files (simplified implementation)
  std::vector<std::string> values;
  std::vector<int32_t> types;
  std::vector<float> scores;
  std::vector<std::string> merges;

  // Load vocabulary file
  std::ifstream vocabFile(tokenizerPath + "/vocab.txt");
  if (vocabFile.is_open()) {
    std::string line;
    while (std::getline(vocabFile, line)) {
      if (!line.empty()) {
        values.push_back(line);
        types.push_back(0); // Normal token
        scores.push_back(0.0f);
      }
    }
    vocabFile.close();
  }

  // Load merges file
  std::ifstream mergesFile(tokenizerPath + "/merges.txt");
  if (mergesFile.is_open()) {
    std::string line;
    while (std::getline(mergesFile, line)) {
      if (!line.empty()) {
        merges.push_back(line);
      }
    }
    mergesFile.close();
  }

  // Initialize vocabulary
  vocabulary_->initialize(values, types, scores, merges);

  if (values.empty()) {
    return false;
  }

  // Use factory to determine tokenizer type based on architecture and overrides
  TokenizerFactoryOptions opts;
  opts.override_type = config_.tokenizer_type; // respect config if provided
  tokenizer_ = createTextProcessorForArchitecture(config_.architecture,
                                                  vocabulary_, opts);

  config_.vocab_size = vocabulary_->size();
  return true;
}

bool BaseModel::loadConfig(const std::string &configPath) {
  std::ifstream file(configPath);
  if (!file.is_open()) {
    // Use default configuration
    config_ = ModelConfig();
    return true;
  }

  // Simple JSON-like parsing (basic implementation)
  std::string line;
  while (std::getline(file, line)) {
    // Remove whitespace
    line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());

    // Parse key-value pairs
    size_t colonPos = line.find(':');
    if (colonPos != std::string::npos) {
      std::string key = line.substr(0, colonPos);
      std::string value = line.substr(colonPos + 1);

      // Remove quotes
      if (!key.empty() && key.front() == '"' && key.back() == '"') {
        key = key.substr(1, key.length() - 2);
      }
      if (!value.empty() && value.front() == '"' && value.back() == '"') {
        value = value.substr(1, value.length() - 2);
      }

      // Remove trailing comma
      if (!value.empty() && value.back() == ',') {
        value.pop_back();
      }

      // Set configuration values
      if (key == "architecture") {
        config_.architecture = value;
      } else if (key == "tokenizer_type") {
        config_.tokenizer_type = value;
      } else if (key == "vocab_size") {
        config_.vocab_size = std::stoul(value);
      } else if (key == "context_length") {
        config_.context_length = std::stoul(value);
      } else if (key == "embedding_dim") {
        config_.embedding_dim = std::stoul(value);
      } else if (key == "num_layers") {
        config_.num_layers = std::stoul(value);
      } else if (key == "num_heads") {
        config_.num_heads = std::stoul(value);
      } else if (key == "temperature") {
        config_.temperature = std::stod(value);
      } else if (key == "top_p") {
        config_.top_p = std::stod(value);
      } else if (key == "top_k") {
        config_.top_k = std::stoi(value);
      }
    }
  }

  return true;
}

int32_t BaseModel::sampleNext(const std::vector<int32_t> &context) {
  // Simple random sampling implementation
  std::vector<double> logits = computeLogits(context);
  return sampleFromLogits(logits);
}

std::vector<double>
BaseModel::computeLogits(const std::vector<int32_t> &context) {
  // Basic implementation: random logits
  size_t vocabSize = getVocabSize();
  if (vocabSize == 0) {
    vocabSize = 10000; // Default size for basic implementation
  }

  std::vector<double> logits(vocabSize);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> dis(0, 1);

  for (double &logit : logits) {
    logit = dis(gen);
  }

  return logits;
}

int32_t BaseModel::sampleFromLogits(const std::vector<double>& logits) {
    if (logits.empty()) {
        return -1;
    }
    
    // Temperature scaling and softmax-like sampling
    double temperature = std::max(0.1, std::min(2.0, config_.temperature));
    
    // Compute probabilities (approximate)
    std::vector<double> probs(logits.size());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] / temperature);
        sum += probs[i];
    }
    for (double& p : probs) {
        p /= sum;
    }
    
    // Sample from distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d(probs.begin(), probs.end());
    
    return static_cast<int32_t>(d(gen));
}

// ModelFactory implementation
std::map<std::string, ModelFactory::ModelCreator> ModelFactory::creators_;

std::unique_ptr<Model> ModelFactory::createModel(const std::string& modelType) {
    auto it = creators_.find(modelType);
    if (it != creators_.end()) {
        return it->second();
    }
    
    // Default to BaseModel
    return std::make_unique<BaseModel>();
}

std::unique_ptr<Model> ModelFactory::loadModel(const std::string& modelPath) {
    auto model = std::make_unique<BaseModel>();
    if (model->load(modelPath)) {
        return std::move(model);
    }
    return nullptr;
}

std::vector<std::string> ModelFactory::getSupportedModels() {
    std::vector<std::string> models;
    models.push_back("BaseModel");
    
    for (const auto& pair : creators_) {
        models.push_back(pair.first);
    }
    
    return models;
}

void ModelFactory::registerModel(const std::string& modelType, ModelCreator creator) {
    creators_[modelType] = creator;
}

} // namespace model
} // namespace duorou