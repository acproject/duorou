#include "sentence_piece.h"
#include <algorithm>
#include <sstream>
#include <cmath>
#include <limits>

namespace duorou {
namespace model {

SentencePiece::SentencePiece(std::shared_ptr<Vocabulary> vocab)
    : vocab_(vocab) {
}

std::vector<int32_t> SentencePiece::encode(const std::string& text, bool addSpecial) {
    std::vector<int32_t> result = encodeText(text);
    
    if (addSpecial && !result.empty()) {
        result = vocab_->addSpecials(result);
    }
    
    return result;
}

std::string SentencePiece::decode(const std::vector<int32_t>& ids) {
    std::ostringstream result;
    
    for (int32_t id : ids) {
        std::string token = vocab_->decode(id);
        
        // Remove space prefix if present (SentencePiece space symbol)
        if (token.length() >= 3 && token.substr(0, 3) == "\xe2\x96\x81") {
            token = token.substr(3);
            if (!result.str().empty()) {
                result << " ";
            }
        }
        
        result << token;
    }
    
    return result.str();
}

bool SentencePiece::isSpecial(int32_t id, Special special) const {
    return vocab_->isSpecial(id, special);
}

const Vocabulary* SentencePiece::getVocabulary() const {
    return vocab_.get();
}

size_t SentencePiece::getVocabSize() const {
    return vocab_->size();
}

std::vector<int32_t> SentencePiece::encodeText(const std::string& text) const {
    std::string normalized = normalizeText(text);
    std::vector<std::string> words = splitByWhitespace(normalized);
    
    std::vector<int32_t> result;
    
    for (const auto& word : words) {
        if (word.empty()) continue;
        
        std::string processedWord = addSpacePrefix(word);
        auto tokens = encodeWord(processedWord);
        result.insert(result.end(), tokens.begin(), tokens.end());
    }
    
    return result;
}

std::string SentencePiece::normalizeText(const std::string& text) const {
    // Basic normalization - convert to lowercase and handle whitespace
    std::string result;
    result.reserve(text.length());
    
    bool lastWasSpace = false;
    for (char c : text) {
        if (isWhitespace(c)) {
            if (!lastWasSpace && !result.empty()) {
                result += ' ';
                lastWasSpace = true;
            }
        } else {
            result += c;
            lastWasSpace = false;
        }
    }
    
    return result;
}

std::vector<std::string> SentencePiece::splitByWhitespace(const std::string& text) const {
    std::vector<std::string> result;
    std::istringstream iss(text);
    std::string word;
    
    while (iss >> word) {
        result.push_back(word);
    }
    
    return result;
}

std::vector<int32_t> SentencePiece::encodeWord(const std::string& word) const {
    // Use Viterbi algorithm to find the best segmentation
    auto tokens = viterbiDecode(word);
    
    std::vector<int32_t> result;
    for (const auto& token : tokens) {
        int32_t id = vocab_->encode(token);
        if (id >= 0) {
            result.push_back(id);
        }
    }
    
    return result;
}

double SentencePiece::calculateScore(const std::vector<std::string>& tokens) const {
    double score = 0.0;
    
    for (const auto& token : tokens) {
        // Simple scoring based on token length (longer tokens are preferred)
        double tokenScore = std::log(static_cast<double>(token.length()));
        if (vocab_->encode(token) < 0) {
            // Unknown token penalty
            tokenScore = -10.0;
        }
        score += tokenScore;
    }
    
    return score;
}

std::vector<std::string> SentencePiece::viterbiDecode(const std::string& text) const {
    if (text.empty()) {
        return {};
    }
    
    size_t len = text.length();
    std::vector<double> scores(len + 1, -std::numeric_limits<double>::infinity());
    std::vector<int> prev(len + 1, -1);
    
    scores[0] = 0.0;
    
    // Dynamic programming
    for (size_t i = 0; i < len; ++i) {
        if (scores[i] == -std::numeric_limits<double>::infinity()) {
            continue;
        }
        
        // Try all possible tokens starting at position i
        for (size_t j = i + 1; j <= len; ++j) {
            std::string token = text.substr(i, j - i);
            int32_t tokenId = vocab_->encode(token);
            
            if (tokenId >= 0) {
                // Simple scoring based on token length (longer tokens are preferred)
                double tokenScore = std::log(static_cast<double>(token.length()));
                double newScore = scores[i] + tokenScore;
                if (newScore > scores[j]) {
                    scores[j] = newScore;
                    prev[j] = static_cast<int>(i);
                }
            }
        }
    }
    
    // Backtrack to find the best path
    std::vector<std::string> result;
    int pos = static_cast<int>(len);
    
    while (pos > 0 && prev[pos] >= 0) {
        int start = prev[pos];
        result.push_back(text.substr(start, pos - start));
        pos = start;
    }
    
    std::reverse(result.begin(), result.end());
    
    // If no valid segmentation found, fall back to character-level
    if (result.empty() && !text.empty()) {
        for (char c : text) {
            result.push_back(std::string(1, c));
        }
    }
    
    return result;
}

bool SentencePiece::isWhitespace(char c) const {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

std::string SentencePiece::addSpacePrefix(const std::string& text) const {
    // SentencePiece uses U+2581 (▁) as space symbol (UTF-8 encoded)
    return "\xe2\x96\x81" + text;
}

std::string SentencePiece::removeSpacePrefix(const std::string& text) const {
    if (text.length() >= 3 && text.substr(0, 3) == "\xe2\x96\x81") {
        return text.substr(3);
    }
    return text;
}

} // namespace model
} // namespace duorou