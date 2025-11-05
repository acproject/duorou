#include "tokenizer_factory.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>

#include "byte_pair_encoding.h"
#include "sentence_piece.h"

namespace duorou {
namespace model {

namespace {
std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
    return s;
}

std::string getEnv(const char* key) {
    const char* v = std::getenv(key);
    return v ? std::string(v) : std::string();
}

// Default patterns
// GPT-2 style generic BPE pattern used in BaseModel
const char* kDefaultGpt2Pattern =
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

// Qwen specific BPE pattern used in QwenTextModel
// Updated to properly handle Chinese characters using Unicode property classes
const char* kQwenPattern =
    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}+| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

// Registry map
using Registry = std::unordered_map<std::string, TextProcessorCreator>;

Registry& registry() {
    static Registry r;
    return r;
}

void ensureDefaultRegistry() {
    static bool inited = false;
    if (inited) return;
    inited = true;

    // "llama" -> SentencePiece
    registerTextProcessor("llama", [](const KVMap&, std::shared_ptr<Vocabulary> vocab, const TokenizerFactoryOptions&) {
        return std::make_unique<SentencePiece>(vocab);
    });

    // "mistral" -> SentencePiece
    registerTextProcessor("mistral", [](const KVMap&, std::shared_ptr<Vocabulary> vocab, const TokenizerFactoryOptions&) {
        return std::make_unique<SentencePiece>(vocab);
    });

    // "gpt2" and many BPE-like -> BPE with default GPT-2 pattern
    auto bpe_default = [](const KVMap& kv, std::shared_ptr<Vocabulary> vocab, const TokenizerFactoryOptions& opts) {
        std::string pattern = opts.override_bpe_pattern;
        if (pattern.empty()) pattern = getEnv("DUOROU_BPE_PATTERN");
        if (pattern.empty()) pattern = kDefaultGpt2Pattern;
        return std::make_unique<BytePairEncoding>(pattern, vocab);
    };

    registerTextProcessor("gpt2", bpe_default);
    registerTextProcessor("bert", bpe_default); // WordPiece not implemented; approximate with BPE
    registerTextProcessor("t5", bpe_default);
    registerTextProcessor("rwkv", bpe_default);
    registerTextProcessor("plamo2", bpe_default);

    // Qwen -> BPE with Qwen pattern
    registerTextProcessor("qwen", [](const KVMap&, std::shared_ptr<Vocabulary> vocab, const TokenizerFactoryOptions& opts) {
        std::string pattern = opts.override_bpe_pattern;
        if (pattern.empty()) pattern = getEnv("DUOROU_BPE_PATTERN");
        if (pattern.empty()) pattern = kQwenPattern;
        return std::make_unique<BytePairEncoding>(pattern, vocab);
    });
}

// Decide tokenizer type (fallback) based on architecture name
std::string decideTypeFromArch(const std::string& archName) {
    std::string a = toLower(archName);
    if (a.find("qwen") != std::string::npos) return "bpe";
    if (a.find("llama") != std::string::npos) return "spm";
    if (a.find("mistral") != std::string::npos) return "spm";
    // Fallback to BPE to match previous default behavior
    return "bpe";
}

// Decide BPE pattern based on architecture
std::string decidePatternFromArch(const std::string& archName) {
    std::string a = toLower(archName);
    if (a.find("qwen") != std::string::npos) return std::string(kQwenPattern);
    return std::string(kDefaultGpt2Pattern);
}

} // namespace

void registerTextProcessor(const std::string& key, TextProcessorCreator creator) {
    ensureDefaultRegistry();
    registry()[toLower(key)] = std::move(creator);
}

std::unique_ptr<TextProcessor> getTextProcessor(
    const KVMap& kv,
    std::shared_ptr<Vocabulary> vocab,
    const TokenizerFactoryOptions& opts) {
    ensureDefaultRegistry();

    // Environment override for type
    std::string type = toLower(opts.override_type);
    if (type.empty()) type = toLower(getEnv("DUOROU_TOKENIZER_TYPE"));

    // Read tokenizer model and pre from kv
    auto itModel = kv.find("tokenizer.ggml.model");
    std::string modelKey = (itModel != kv.end()) ? toLower(itModel->second) : std::string();

    if (!type.empty()) {
        if (type == "spm" || type == "sentencepiece" || type == "sentence_piece") {
            return std::make_unique<SentencePiece>(vocab);
        }
        // BPE fallback
        std::string pattern = opts.override_bpe_pattern.empty() ? getEnv("DUOROU_BPE_PATTERN") : opts.override_bpe_pattern;
        if (pattern.empty()) {
            // If model suggests Qwen, prefer Qwen pattern
            if (modelKey.find("qwen") != std::string::npos) pattern = kQwenPattern;
            else pattern = kDefaultGpt2Pattern;
        }
        return std::make_unique<BytePairEncoding>(pattern, vocab);
    }

    // Try registry by tokenizer model
    auto it = registry().find(modelKey);
    if (it != registry().end()) {
        return (it->second)(kv, std::move(vocab), opts);
    }

    // Try derive from pre-tokenizer hint
    auto itPre = kv.find("tokenizer.ggml.pre");
    if (itPre != kv.end()) {
        const std::string pre = toLower(itPre->second);
        if (pre.find("llama") != std::string::npos) {
            return std::make_unique<SentencePiece>(vocab);
        }
        // common BPE presets
        std::string pattern = opts.override_bpe_pattern.empty() ? getEnv("DUOROU_BPE_PATTERN") : opts.override_bpe_pattern;
        if (pattern.empty()) pattern = kDefaultGpt2Pattern;
        return std::make_unique<BytePairEncoding>(pattern, vocab);
    }

    // Ultimate fallback: BPE default
    std::string pattern = opts.override_bpe_pattern.empty() ? getEnv("DUOROU_BPE_PATTERN") : opts.override_bpe_pattern;
    if (pattern.empty()) pattern = kDefaultGpt2Pattern;
    return std::make_unique<BytePairEncoding>(pattern, vocab);
}

std::unique_ptr<TextProcessor> createTextProcessorForArchitecture(
    const std::string& architecture,
    std::shared_ptr<Vocabulary> vocab,
    const TokenizerFactoryOptions& opts) {
    std::cout << "[DEBUG] createTextProcessorForArchitecture called with architecture='" << architecture << "'" << std::endl;
    ensureDefaultRegistry();
    
    // Environment overrides
    std::string envType = toLower(getEnv("DUOROU_TOKENIZER_TYPE")); // "bpe" or "spm"
    std::string envPattern = getEnv("DUOROU_BPE_PATTERN");

    // Determine type
    std::string type = toLower(opts.override_type);
    if (type.empty()) type = envType;
    if (type.empty()) type = decideTypeFromArch(architecture);
    
    std::cout << "[DEBUG] Determined tokenizer type: '" << type << "'" << std::endl;

    if (type == "spm" || type == "sentencepiece" || type == "sentence_piece") {
        std::cout << "[DEBUG] Creating SentencePiece tokenizer" << std::endl;
        return std::make_unique<SentencePiece>(vocab);
    }

    // BPE path
    std::string pattern = opts.override_bpe_pattern.empty() ? envPattern : opts.override_bpe_pattern;
    if (pattern.empty()) pattern = decidePatternFromArch(architecture);
    std::cout << "[DEBUG] Using BPE pattern: '" << pattern << "'" << std::endl;
    std::cout << "[DEBUG] Creating BytePairEncoding tokenizer..." << std::endl;
    
    try {
        auto tokenizer = std::make_unique<BytePairEncoding>(pattern, vocab);
        std::cout << "[DEBUG] BytePairEncoding tokenizer created successfully!" << std::endl;
        return tokenizer;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to create BytePairEncoding tokenizer: " << e.what() << std::endl;
        return nullptr;
    }
}

std::shared_ptr<Vocabulary> createVocabularyFromGGUF(
    const duorou::extensions::ollama::GGUFParser& parser) {
    // Read tokens from GGUF
    std::vector<std::string> tokens;
    if (const auto* kvTokens = parser.getMetadata("tokenizer.ggml.tokens")) {
        tokens = kvTokens->asStringArray();
    }
    
    if (tokens.empty()) {
        std::cerr << "[ERROR] No tokens found in GGUF file" << std::endl;
        return nullptr;
    }
    
    // Read token types
    std::vector<int32_t> types;
    if (const auto* kvTypes = parser.getMetadata("tokenizer.ggml.token_type")) {
        types = kvTypes->asInt32Array();
    }
    
    // If no types provided, default all to normal
    if (types.empty() && !tokens.empty()) {
        types.assign(tokens.size(), TOKEN_TYPE_NORMAL);
    }
    
    // Read token scores (optional) - currently not implemented in GGUFKeyValue
    std::vector<float> scores;
    // TODO: Implement scores reading when asFloat32Array is available
    // if (const auto* kvScores = parser.getMetadata("tokenizer.ggml.scores")) {
    //     scores = kvScores->asFloat32Array();
    // }
    
    // Read merges
    std::vector<std::string> merges;
    if (const auto* kvMerges = parser.getMetadata("tokenizer.ggml.merges")) {
        merges = kvMerges->asStringArray();
    }
    
    // Create and initialize vocabulary
    auto vocab = std::make_shared<Vocabulary>();
    vocab->initialize(tokens, types, scores, merges);
    
    // Set special tokens based on GGUF metadata
    // BOS token
    if (const auto* kvBOS = parser.getMetadata("tokenizer.ggml.bos_token_id")) {
        std::vector<int32_t> bos = {kvBOS->asInt32()};
        vocab->setBOS(bos, false); // Don't auto-add BOS by default
    }
    
    // EOS token
    if (const auto* kvEOS = parser.getMetadata("tokenizer.ggml.eos_token_id")) {
        std::vector<int32_t> eos = {kvEOS->asInt32()};
        vocab->setEOS(eos, false); // Don't auto-add EOS by default
    }
    
    // PAD token
    if (const auto* kvPAD = parser.getMetadata("tokenizer.ggml.pad_token_id")) {
        std::vector<int32_t> pad = {kvPAD->asInt32()};
        vocab->setPAD(pad);
    }
    
    // UNK token
    if (const auto* kvUNK = parser.getMetadata("tokenizer.ggml.unk_token_id")) {
        std::vector<int32_t> unk = {kvUNK->asInt32()};
        vocab->setUNK(unk);
    }
    
    std::cout << "[INFO] Created vocabulary from GGUF: " << tokens.size() 
              << " tokens, " << merges.size() << " merges" << std::endl;
    
    return vocab;
}

std::unique_ptr<TextProcessor> createTextProcessorFromGGUF(
    const duorou::extensions::ollama::GGUFParser& parser,
    std::shared_ptr<Vocabulary> vocab,
    const TokenizerFactoryOptions& opts) {
    // Build kv map from GGUF metadata
    KVMap kv;
    if (auto* kvModel = parser.getMetadata("tokenizer.ggml.model")) kv["tokenizer.ggml.model"] = kvModel->asString();
    if (auto* kvPre   = parser.getMetadata("tokenizer.ggml.pre"))   kv["tokenizer.ggml.pre"]   = kvPre->asString();

    // Prefer tokenizer registry by kv; fallback to architecture-derived
    if (!kv.empty()) {
        return getTextProcessor(kv, std::move(vocab), opts);
    }

    // Fallback: prefer architecture name from parser
    std::string arch = parser.getArchitecture().name;

    // Fallback: try to read metadata directly if empty
    if (arch.empty()) {
        if (auto* gkv = parser.getMetadata("general.architecture")) {
            arch = gkv->asString();
        }
    }

    return createTextProcessorForArchitecture(arch, std::move(vocab), opts);
}

std::unique_ptr<TextProcessor> createTextProcessorFromGGUF(
    const duorou::extensions::ollama::GGUFParser& parser,
    const TokenizerFactoryOptions& opts) {
    // Create vocabulary from GGUF
    auto vocab = createVocabularyFromGGUF(parser);
    if (!vocab) {
        std::cerr << "[ERROR] Failed to create vocabulary from GGUF" << std::endl;
        return nullptr;
    }
    
    // Create text processor with the vocabulary
    return createTextProcessorFromGGUF(parser, vocab, opts);
}

} // namespace model
} // namespace duorou