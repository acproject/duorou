#pragma once

#include "text_processor.h"
#include "vocabulary.h"
#include <memory>
#include <string>
#include <vector>
#include <cstdint>

namespace duorou {
namespace model {

class SentencePiece : public TextProcessor {
public:
    explicit SentencePiece(std::shared_ptr<Vocabulary> vocab);
    
    // TextProcessor interface
    std::vector<int32_t> encode(const std::string& text, bool addSpecial = true) override;
    std::string decode(const std::vector<int32_t>& ids) override;
    bool isSpecial(int32_t id, Special special) const override;
    const Vocabulary* getVocabulary() const override;
    size_t getVocabSize() const override;

private:
    std::shared_ptr<Vocabulary> vocab_;
    
    // SentencePiece specific methods
    std::vector<int32_t> encodeText(const std::string& text) const;
    std::string normalizeText(const std::string& text) const;
    std::vector<std::string> splitByWhitespace(const std::string& text) const;
    std::vector<int32_t> encodeWord(const std::string& word) const;
    
    // Unigram language model scoring
    double calculateScore(const std::vector<std::string>& tokens) const;
    std::vector<std::string> viterbiDecode(const std::string& text) const;
    
    // Helper methods
    bool isWhitespace(char c) const;
    std::string addSpacePrefix(const std::string& text) const;
    std::string removeSpacePrefix(const std::string& text) const;
};

} // namespace model
} // namespace duorou