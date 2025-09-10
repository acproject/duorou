#ifndef BPE_PROCESSOR_H
#define BPE_PROCESSOR_H

#include "../text_processor.h"
#include <string>
#include <vector>
#include <regex>
#include <memory>
#include <queue>
#include <unordered_map>

namespace duorou {
namespace extensions {
namespace ollama {

struct BPEMerge {
    int p, n;
    std::vector<char32_t> runes;
    
    BPEMerge() : p(-1), n(-1) {}
};

// Pair structure for BPE merging
struct BPEPair {
    int a, b;
    int rank;
    std::string value;
    
    BPEPair(int a_val, int b_val, int rank_val, const std::string& value_val)
        : a(a_val), b(b_val), rank(rank_val), value(value_val) {}
};

// Comparison function for BPE pairs
struct BPEPairComparator {
    bool operator()(const std::shared_ptr<BPEPair>& a, const std::shared_ptr<BPEPair>& b) const {
        return a->rank > b->rank; // Lower rank has higher priority
    }
};

// BytePairEncoding text processor implementation
class BPEProcessor : public TextProcessor {
public:
    BPEProcessor(const std::string& pre_tokenizer_regex, std::shared_ptr<Vocabulary> vocab);
    virtual ~BPEProcessor() = default;

    // TextProcessor interface implementation
    std::vector<int32_t> encode(const std::string& text, bool add_special = true) override;
    std::string decode(const std::vector<int32_t>& tokens) override;
    bool is(int32_t token_id, Special special) const override;
    const Vocabulary* getVocabulary() const override;
    size_t getVocabSize() const override;

private:
    std::shared_ptr<Vocabulary> vocab_;
    std::regex pre_tokenizer_;
    
    // 性能优化：缓存机制
  mutable std::unordered_map<std::string, std::vector<int32_t>> encode_cache_;
  mutable std::unordered_map<std::string, std::string> decode_cache_;
  static constexpr size_t MAX_CACHE_SIZE = 10000;
    
    // Helper functions
    std::vector<std::string> splitText(const std::string& text) const;
    std::vector<Fragment> processSpecialTokens(const std::string& text) const;
    std::vector<int32_t> tokenizeFragment(const std::string& text) const;
    std::string preprocessBytes(const std::string& text) const;
    std::string postprocessBytes(const std::string& text) const;
    char32_t mapByte(unsigned char byte) const;
    unsigned char unmapByte(char32_t rune) const;
    std::vector<char32_t> stringToRunes(const std::string& text) const;
    std::string runesToString(const std::vector<char32_t>& runes) const;
    
    // Fallback mechanism
    std::vector<int32_t> characterLevelFallback(const std::string& text) const;
};

} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // BPE_PROCESSOR_H