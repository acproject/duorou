#include "tokenizer_factory.h"

#include <algorithm>
#include <cstdlib>

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
const char* kQwenPattern =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

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
    // Environment overrides
    std::string envType = toLower(getEnv("DUOROU_TOKENIZER_TYPE")); // "bpe" or "spm"
    std::string envPattern = getEnv("DUOROU_BPE_PATTERN");

    // Determine type
    std::string type = toLower(opts.override_type);
    if (type.empty()) type = envType;
    if (type.empty()) type = decideTypeFromArch(architecture);

    if (type == "spm" || type == "sentencepiece" || type == "sentence_piece") {
        return std::make_unique<SentencePiece>(vocab);
    }

    // BPE path
    std::string pattern = opts.override_bpe_pattern.empty() ? envPattern : opts.override_bpe_pattern;
    if (pattern.empty()) pattern = decidePatternFromArch(architecture);
    return std::make_unique<BytePairEncoding>(pattern, vocab);
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

} // namespace model
} // namespace duorou