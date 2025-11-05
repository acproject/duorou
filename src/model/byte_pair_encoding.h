#pragma once

#include "text_processor.h"
#include "vocabulary.h"
#include <regex>
#include <memory>
#include <queue>

namespace duorou {
namespace model {

/**
 * Fragment represents a text fragment and its corresponding token IDs
 */
struct Fragment {
    std::string value;
    std::vector<int32_t> ids;
    
    Fragment(const std::string& v) : value(v) {}
    Fragment(const std::string& v, const std::vector<int32_t>& i) : value(v), ids(i) {}
};

/**
 * Pair represents a pair of positions and its merge rank
 */
struct Pair {
    int a, b;
    int rank;
    std::string value;
    
    Pair(int a_, int b_, int rank_, const std::string& value_) 
        : a(a_), b(b_), rank(rank_), value(value_) {}
    
    // For priority queue (min-heap based on rank)
    bool operator>(const Pair& other) const {
        return rank > other.rank;
    }
};

/**
 * Merge represents a merge operation in the BPE algorithm
 */
struct Merge {
    int prev, next;
    std::vector<uint32_t> runes;
    
    Merge() : prev(-1), next(-1) {}
};

/**
 * BytePairEncoding tokenizer implementation
 * Similar to Ollama's BytePairEncoding
 */
class BytePairEncoding : public TextProcessor {
public:
    /**
     * Constructor
     * @param pattern Regular expression pattern for pre-tokenization
     * @param vocab Vocabulary to use
     */
    BytePairEncoding(const std::string& pattern, std::shared_ptr<Vocabulary> vocab);
    
    // Destructor
    ~BytePairEncoding() override = default;
    
    // TextProcessor interface implementation
    std::vector<int32_t> encode(const std::string& text, bool addSpecial = true) override;
    std::string decode(const std::vector<int32_t>& ids) override;
    bool isSpecial(int32_t id, Special special) const override;
    const Vocabulary* getVocabulary() const override;
    size_t getVocabSize() const override;
    
private:
    std::regex preTokenizeRegex_;
    std::shared_ptr<Vocabulary> vocab_;
    
    /**
     * Split text using the pre-tokenization regex
     */
    std::vector<std::string> split(const std::string& text) const;
    
    /**
     * Convert byte to Unicode codepoint for BPE processing
     */
    uint32_t byteToUnicode(uint8_t byte) const;
    
    /**
     * Convert Unicode codepoint back to byte
     */
    uint8_t unicodeToByte(uint32_t codepoint) const;
    
    /**
     * Process special tokens in fragments
     */
    std::vector<Fragment> processSpecialTokens(const std::vector<Fragment>& fragments) const;
    
    /**
     * Apply BPE merges to a text fragment
     */
    std::vector<int32_t> applyBPE(const std::string& text) const;
};

} // namespace model
} // namespace duorou