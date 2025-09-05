#ifndef TEXT_PROCESSOR_H
#define TEXT_PROCESSOR_H

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cstdint>
#include <algorithm>
#include <regex>

namespace duorou {
namespace extensions {
namespace ollama {

// Token types based on ollama's implementation
enum TokenType {
    TOKEN_TYPE_NORMAL = 1,
    TOKEN_TYPE_UNKNOWN,
    TOKEN_TYPE_CONTROL,
    TOKEN_TYPE_USER_DEFINED,
    TOKEN_TYPE_UNUSED,
    TOKEN_TYPE_BYTE
};

// Fragment structure used by text processors
struct Fragment {
    std::string value;
    std::vector<int32_t> ids;
    
    Fragment(const std::string& v) : value(v) {}
    Fragment(const std::string& v, const std::vector<int32_t>& i) : value(v), ids(i) {}
};

// Special token types
enum Special {
    SPECIAL_BOS = 0,
    SPECIAL_EOS
};

// Vocabulary class that manages token mappings and special tokens
class Vocabulary {
public:
    Vocabulary() = default;
    ~Vocabulary() = default;

    // Initialize vocabulary from vectors
    void initialize(const std::vector<std::string>& values,
                   const std::vector<int32_t>& types,
                   const std::vector<float>& scores,
                   const std::vector<std::string>& merges = {});

    // Token encoding/decoding
    int32_t encode(const std::string& token) const;
    std::string decode(int32_t id) const;

    // Special token management
    bool is(int32_t id, Special special) const;
    void setBOS(const std::vector<int32_t>& bos_tokens, bool add_bos = false);
    void setEOS(const std::vector<int32_t>& eos_tokens, bool add_eos = false);

    // Merge operations for BPE
    int merge(const std::string& left, const std::string& right) const;

    // Get special vocabulary (control and user-defined tokens)
    std::vector<std::string> getSpecialVocabulary() const;

    // Add special tokens to token sequence
    std::vector<int32_t> addSpecials(const std::vector<int32_t>& ids) const;

    // Getters
    const std::vector<std::string>& getValues() const { return values_; }
    const std::vector<int32_t>& getTypes() const { return types_; }
    const std::vector<float>& getScores() const { return scores_; }
    const std::vector<std::string>& getMerges() const { return merges_; }
    size_t size() const { return values_.size(); }

private:
    std::vector<std::string> values_;     // Token strings
    std::vector<int32_t> types_;          // Token types
    std::vector<float> scores_;           // Token scores for merging
    std::vector<std::string> merges_;     // Merge rules for BPE

    std::vector<int32_t> bos_tokens_;     // Beginning of sequence tokens
    std::vector<int32_t> eos_tokens_;     // End of sequence tokens
    bool add_bos_ = false;                // Whether to add BOS automatically
    bool add_eos_ = false;                // Whether to add EOS automatically

    // Cached mappings for performance
    mutable std::unordered_map<std::string, int32_t> token_to_id_;
    mutable std::unordered_map<std::string, int32_t> merge_map_;
    mutable std::vector<std::string> special_vocab_;
    mutable bool token_to_id_cached_ = false;
    mutable bool merge_map_cached_ = false;
    mutable bool special_vocab_cached_ = false;

    void buildTokenToIdCache() const;
    void buildMergeMapCache() const;
    void buildSpecialVocabCache() const;
};

// Abstract base class for text processors
class TextProcessor {
public:
    virtual ~TextProcessor() = default;

    // Encode text to token IDs
    virtual std::vector<int32_t> encode(const std::string& text, bool add_special = true) = 0;

    // Decode token IDs to text
    virtual std::string decode(const std::vector<int32_t>& tokens) = 0;

    // Check if token is a special token
    virtual bool is(int32_t token_id, Special special) const = 0;

    // Get vocabulary
    virtual const Vocabulary* getVocabulary() const = 0;

    // Get vocabulary size
    virtual size_t getVocabSize() const = 0;
};

// Factory function to create text processor based on type
std::unique_ptr<TextProcessor> createTextProcessor(const std::string& type, 
                                                  std::shared_ptr<Vocabulary> vocab,
                                                  const std::string& pre_tokenizer_regex = "");

} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // TEXT_PROCESSOR_H