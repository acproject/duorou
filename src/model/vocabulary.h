#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include <mutex>
#include "text_processor.h"

namespace duorou {
namespace model {

// Token type constants
const int32_t TOKEN_TYPE_NORMAL = 0;
const int32_t TOKEN_TYPE_CONTROL = 1;
const int32_t TOKEN_TYPE_USER_DEFINED = 2;
// Extended token types for compatibility with Go/SentencePiece ecosystems
// Note: Current implementation only treats CONTROL and USER_DEFINED as special.
// UNKNOWN/UNUSED/BYTE behave like NORMAL unless explicitly handled elsewhere.
const int32_t TOKEN_TYPE_UNKNOWN = 3;  // e.g., <unk>
const int32_t TOKEN_TYPE_UNUSED  = 4;  // reserved/unused slots
const int32_t TOKEN_TYPE_BYTE    = 5;  // byte-fallback tokens in some BPE vocabs

/**
 * Vocabulary class for managing tokens, scores, and merges
 * Similar to Ollama's Vocabulary class
 */
class Vocabulary {
public:
    // Constructor and destructor
    Vocabulary() = default;
    
    // Destructor
    ~Vocabulary() = default;
    
    // Delete copy and move constructors and assignment operators (due to std::once_flag)
    Vocabulary(const Vocabulary& other) = delete;
    Vocabulary& operator=(const Vocabulary& other) = delete;
    Vocabulary(Vocabulary&& other) = delete;
    Vocabulary& operator=(Vocabulary&& other) = delete;
    
    /**
     * Initialize vocabulary with tokens, types, scores, and merges
     */
    void initialize(const std::vector<std::string>& values,
                   const std::vector<int32_t>& types,
                   const std::vector<float>& scores = std::vector<float>(),
                   const std::vector<std::string>& merges = std::vector<std::string>());
    
    /**
     * Check if a token ID represents a special token
     */
    bool isSpecial(int32_t id, Special special) const;
    
    /**
     * Add special tokens to the token sequence
     */
    std::vector<int32_t> addSpecials(const std::vector<int32_t>& ids) const;
    
    /**
     * Encode a string to token ID
     * @param token String token to encode
     * @return Token ID, or -1 if not found
     */
    int32_t encode(const std::string& token) const;
    
    /**
     * Decode a token ID to string
     * @param id Token ID to decode
     * @return String token
     */
    std::string decode(int32_t id) const;
    
    /**
     * Get special vocabulary (control and user-defined tokens)
     */
    std::vector<std::string> getSpecialVocabulary() const;
    
    /**
     * Get merge rank for a pair of tokens
     * @param left Left token
     * @param right Right token
     * @return Merge rank, or -1 if not found
     */
    int getMergeRank(const std::string& left, const std::string& right) const;
    
    /**
     * Set BOS (Beginning of Sequence) tokens
     */
    void setBOS(const std::vector<int32_t>& bos, bool addBOS = false);
    
    /**
     * Set EOS (End of Sequence) tokens
     */
    void setEOS(const std::vector<int32_t>& eos, bool addEOS = false);

    /**
     * Set PAD tokens
     */
    void setPAD(const std::vector<int32_t>& pad);

    /**
     * Set UNK tokens
     */
    void setUNK(const std::vector<int32_t>& unk);

    /**
     * Get the first id of a given Special token, or -1 if not present
     */
    int32_t getSpecialId(Special special) const;
    
    // Getters
    const std::vector<std::string>& getValues() const { return values_; }
    const std::vector<int32_t>& getTypes() const { return types_; }
    const std::vector<float>& getScores() const { return scores_; }
    const std::vector<std::string>& getMerges() const { return merges_; }
    size_t size() const { return values_.size(); }
    
private:
    // Core vocabulary data
    std::vector<std::string> values_;
    std::vector<int32_t> types_;
    std::vector<float> scores_;
    std::vector<std::string> merges_;
    
    // Special tokens
    std::vector<int32_t> bos_;
    std::vector<int32_t> eos_;
    std::vector<int32_t> pad_;
    std::vector<int32_t> unk_;
    bool addBOS_ = false;
    bool addEOS_ = false;
    
    // Cached data for fast lookup
    mutable std::once_flag specialOnce_;
    mutable std::vector<std::string> specialTokens_;
    
    mutable std::once_flag valuesOnce_;
    mutable std::unordered_map<std::string, int32_t> tokenToId_;
    
    mutable std::once_flag mergeOnce_;
    mutable std::unordered_map<std::string, int32_t> mergeMap_;
    
    // Helper methods
    void buildTokenMap() const;
    void buildSpecialTokens() const;
    void buildMergeMap() const;

    // GPT-2 byte-level BPE decoding
    std::string decodeText(const std::string& text) const;

    // Auto-detect PAD/UNK ids by common token strings (e.g., "<pad>", "<unk>")
    void autodetectPadUnk();
};

} // namespace model
} // namespace duorou