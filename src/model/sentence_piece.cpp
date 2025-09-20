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
    // Basic normalization - collapse consecutive whitespace to a single space
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
    const auto& scores = vocab_->getScores();
    for (const auto& token : tokens) {
        int32_t id = vocab_->encode(token);
        if (id >= 0 && static_cast<size_t>(id) < scores.size()) {
            score += static_cast<double>(scores[id]);
        } else {
            // Unknown token heavy penalty
            score += -10.0;
        }
    }
    return score;
}

std::vector<std::string> SentencePiece::viterbiDecode(const std::string& text) const {
    if (text.empty()) {
        return {};
    }
    
    size_t len = text.length();
    std::vector<double> dp(len + 1, -std::numeric_limits<double>::infinity());
    std::vector<int> prev(len + 1, -1);
    dp[0] = 0.0;

    const auto& scores = vocab_->getScores();

    // Dynamic programming over substrings
    for (size_t i = 0; i < len; ++i) {
        if (dp[i] == -std::numeric_limits<double>::infinity()) continue;
        for (size_t j = i + 1; j <= len; ++j) {
            std::string token = text.substr(i, j - i);
            int32_t id = vocab_->encode(token);
            if (id >= 0) {
                double tokenScore = 0.0;
                if (static_cast<size_t>(id) < scores.size()) {
                    tokenScore = static_cast<double>(scores[id]);
                } else {
                    // If no explicit score, approximate by length preference
                    tokenScore = std::log(static_cast<double>(token.length()));
                }
                double newScore = dp[i] + tokenScore;
                if (newScore > dp[j]) {
                    dp[j] = newScore;
                    prev[j] = static_cast<int>(i);
                }
            }
        }
    }

    // Backtrack
    std::vector<std::string> result;
    int pos = static_cast<int>(len);
    while (pos > 0 && prev[pos] >= 0) {
        int start = prev[pos];
        result.push_back(text.substr(start, pos - start));
        pos = start;
    }
    std::reverse(result.begin(), result.end());

    // Fallback: if no path, try byte-level <0x..> tokens if present
    if ((result.empty() || prev[len] < 0) && !text.empty()) {
        result.clear();
        for (unsigned char b : std::string(text.begin(), text.end())) {
            char buf[6];
            std::snprintf(buf, sizeof(buf), "%02X", static_cast<unsigned int>(b));
            std::string hex(buf);
            std::string byteTok = std::string("<0x") + hex + ">";
            if (vocab_->encode(byteTok) >= 0) {
                result.push_back(byteTok);
            } else {
                // fallback to raw single-byte string
                result.push_back(std::string(1, static_cast<char>(b)));
            }
        }
    }

    // If still empty, final fallback to character-level split
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
    // SentencePiece uses U+2581 ( ) as space symbol (UTF-8 encoded)
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