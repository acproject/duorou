#ifndef SENTENCEPIECE_PROCESSOR_H
#define SENTENCEPIECE_PROCESSOR_H

#include <string>
#include <vector>
#include <memory>
#include <queue>
#include <cstdint>
#include <functional>
#include "text_processor.h"

namespace duorou {
namespace extensions {
namespace ollama {

// Candidate structure for merge operations
struct Candidate {
    int a, b;
    float score;
    int size;
    
    Candidate(int a_val, int b_val, float score_val, int size_val)
        : a(a_val), b(b_val), score(score_val), size(size_val) {}
};

// Comparison function for priority queue
struct CandidateComparator {
    bool operator()(const std::shared_ptr<Candidate>& a, const std::shared_ptr<Candidate>& b) const {
        if (a->score != b->score) {
            return a->score < b->score; // Higher score has higher priority
        }
        return a->a > b->a; // Lower index has higher priority
    }
};

// Merge structure for tokenization
struct SPMerge {
    int p, n; // previous and next indices
    std::vector<char32_t> runes; // Unicode code points
    
    SPMerge() : p(-1), n(-1) {}
};

// SentencePiece text processor implementation
class SentencePieceProcessor : public TextProcessor {
public:
    explicit SentencePieceProcessor(std::shared_ptr<Vocabulary> vocab);
    virtual ~SentencePieceProcessor() = default;

    // TextProcessor interface implementation
    std::vector<int32_t> encode(const std::string& text, bool add_special = true) override;
    std::string decode(const std::vector<int32_t>& tokens) override;
    bool is(int32_t token_id, Special special) const override;
    const Vocabulary* getVocabulary() const override;
    size_t getVocabSize() const override;

private:
    std::shared_ptr<Vocabulary> vocab_;
    int max_token_len_;
    
    // Helper functions
    std::vector<Fragment> processSpecialTokens(const std::string& text) const;
    std::vector<int32_t> tokenizeFragment(const std::string& text) const;
    std::string utf8ToUnicode(const std::string& text) const;
    std::string unicodeToUtf8(const std::vector<char32_t>& runes) const;
    std::vector<char32_t> stringToRunes(const std::string& text) const;
    std::string runesToString(const std::vector<char32_t>& runes) const;
    
    // Fallback mechanism
    std::vector<int32_t> characterLevelFallback(const std::string& text) const;
    
    // UTF-8 utility functions
    std::string bytesToUTF8(const std::vector<uint8_t>& bytes) const;
    bool isValidUTF8(const std::string& str) const;
    std::string cleanInvalidUTF8(const std::string& str) const;
    
    // Constants
    static const std::string WHITESPACE_SEP;
};

} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // SENTENCEPIECE_PROCESSOR_H