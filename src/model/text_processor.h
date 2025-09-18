#pragma once

#include <vector>
#include <string>
#include <cstdint>

namespace duorou {
namespace model {

// Forward declarations
class Vocabulary;

// Token types
enum class Special {
    PAD = 0,
    UNK = 1,
    BOS = 2,
    EOS = 3
};

// Text processor interface
class TextProcessor {
public:
    virtual ~TextProcessor() = default;
    
    // Encode text to token IDs
    virtual std::vector<int32_t> encode(const std::string& text, bool addSpecial = true) = 0;
    
    // Decode token IDs to text
    virtual std::string decode(const std::vector<int32_t>& ids) = 0;
    
    // Check if token ID is special
    virtual bool isSpecial(int32_t id, Special special) const = 0;
    
    // Get vocabulary
    virtual const Vocabulary* getVocabulary() const = 0;
    
    // Get vocabulary size
    virtual size_t getVocabSize() const = 0;
};

} // namespace model
} // namespace duorou