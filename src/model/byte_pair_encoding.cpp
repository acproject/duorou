#include "byte_pair_encoding.h"
#include <algorithm>
#include <sstream>
#include <codecvt>
#include <locale>

namespace duorou {
namespace model {

BytePairEncoding::BytePairEncoding(const std::string& pattern, std::shared_ptr<Vocabulary> vocab)
    : preTokenizeRegex_(pattern), vocab_(vocab) {
}

std::vector<int32_t> BytePairEncoding::encode(const std::string& text, bool addSpecial) {
    // Start with a single fragment containing the entire text
    std::vector<Fragment> fragments = {Fragment(text)};
    
    // Process special tokens
    fragments = processSpecialTokens(fragments);
    
    std::vector<int32_t> result;
    
    for (const auto& fragment : fragments) {
        if (!fragment.ids.empty()) {
            // Fragment already has token IDs (special token)
            result.insert(result.end(), fragment.ids.begin(), fragment.ids.end());
            continue;
        }
        
        // Split the fragment using pre-tokenization regex
        auto splits = split(fragment.value);
        
        for (const auto& split : splits) {
            auto tokens = applyBPE(split);
            result.insert(result.end(), tokens.begin(), tokens.end());
        }
    }
    
    if (addSpecial && !result.empty()) {
        result = vocab_->addSpecials(result);
    }
    
    return result;
}

std::string BytePairEncoding::decode(const std::vector<int32_t>& ids) {
    std::ostringstream result;
    
    for (int32_t id : ids) {
        std::string token = vocab_->decode(id);
        
        // Iterate UTF-8 runes in token and map back to single bytes like Go implementation
        for (size_t i = 0; i < token.length(); ) {
            uint32_t codepoint = 0;
            int bytes = 1;
            
            unsigned char c = token[i];
            if (c < 0x80) {
                codepoint = c;
            } else if ((c & 0xE0) == 0xC0) {
                codepoint = (c & 0x1F) << 6;
                if (i + 1 < token.length()) {
                    codepoint |= (token[i + 1] & 0x3F);
                    bytes = 2;
                }
            } else if ((c & 0xF0) == 0xE0) {
                codepoint = (c & 0x0F) << 12;
                if (i + 2 < token.length()) {
                    codepoint |= (token[i + 1] & 0x3F) << 6;
                    codepoint |= (token[i + 2] & 0x3F);
                    bytes = 3;
                }
            } else if ((c & 0xF8) == 0xF0) {
                codepoint = (c & 0x07) << 18;
                if (i + 3 < token.length()) {
                    codepoint |= (token[i + 1] & 0x3F) << 12;
                    codepoint |= (token[i + 2] & 0x3F) << 6;
                    codepoint |= (token[i + 3] & 0x3F);
                    bytes = 4;
                }
            }

            // Align with Go's Decode mapping
            uint32_t r = codepoint;
            if (r == 0x0100) {
                // Skip writing NULL byte just like Go does
                i += bytes;
                continue;
            }
            if (r == 0x0143) {
                r = 0x00ad;
            } else if (r > 0x0100 && r <= 0x0120) {
                r = r - 0x0100;
            } else if (r > 0x0120 && r <= 0x0142) {
                r = r - 0x00a2;
            }

            // Always write a single byte (lower 8 bits), matching Go's WriteByte(byte(r))
            result << static_cast<char>(static_cast<uint8_t>(r & 0xFF));
            
            i += bytes;
        }
    }
    
    return result.str();
}

bool BytePairEncoding::isSpecial(int32_t id, Special special) const {
    return vocab_->isSpecial(id, special);
}

const Vocabulary* BytePairEncoding::getVocabulary() const {
    return vocab_.get();
}

size_t BytePairEncoding::getVocabSize() const {
    return vocab_->size();
}

std::vector<std::string> BytePairEncoding::split(const std::string& text) const {
    std::vector<std::string> result;
    std::sregex_iterator iter(text.begin(), text.end(), preTokenizeRegex_);
    std::sregex_iterator end;
    
    for (; iter != end; ++iter) {
        result.push_back(iter->str());
    }
    
    return result;
}

uint32_t BytePairEncoding::byteToUnicode(uint8_t byte) const {
    // Convert byte to Unicode codepoint for BPE processing (aligned with Go)
    if (byte == 0x00ad) {
        return 0x0143;
    } else if (byte <= 0x20) {
        return static_cast<uint32_t>(byte) + 0x0100;
    } else if (byte >= 0x7f && byte <= 0xa0) {
        return static_cast<uint32_t>(byte) + 0x00a2;
    } else {
        return byte;
    }
}

uint8_t BytePairEncoding::unicodeToByte(uint32_t codepoint) const {
    // Retained for completeness, but decode() now mirrors Go directly
    if (codepoint == 0x0100) {
        return 0x00; // NULL (handled specially in decode)
    } else if (codepoint == 0x0143) {
        return 0x00ad;
    } else if (codepoint > 0x0100 && codepoint <= 0x0120) {
        return static_cast<uint8_t>(codepoint - 0x0100);
    } else if (codepoint > 0x0120 && codepoint <= 0x0142) {
        return static_cast<uint8_t>(codepoint - 0x00a2);
    } else if (codepoint <= 0xFF) {
        return static_cast<uint8_t>(codepoint);
    } else {
        return 0; // Invalid/unmapped
    }
}

std::vector<Fragment> BytePairEncoding::processSpecialTokens(const std::vector<Fragment>& fragments) const {
    std::vector<Fragment> result = fragments;
    
    auto specialTokens = vocab_->getSpecialVocabulary();
    
    for (const auto& special : specialTokens) {
        int32_t id = vocab_->encode(special);
        if (id < 0) continue;
        
        for (size_t i = 0; i < result.size(); ++i) {
            Fragment& frag = result[i];
            if (!frag.ids.empty()) continue; // Already processed
            
            size_t pos = frag.value.find(special);
            if (pos == std::string::npos) continue;
            
            std::vector<Fragment> newFragments;
            
            // Add text before special token
            if (pos > 0) {
                newFragments.emplace_back(frag.value.substr(0, pos));
            }
            
            // Add special token
            newFragments.emplace_back(special, std::vector<int32_t>{id});
            
            // Add text after special token
            if (pos + special.length() < frag.value.length()) {
                newFragments.emplace_back(frag.value.substr(pos + special.length()));
            }
            
            // Replace the fragment
            result.erase(result.begin() + i);
            result.insert(result.begin() + i, newFragments.begin(), newFragments.end());
            i += newFragments.size() - 1;
        }
    }
    
    return result;
}

std::vector<int32_t> BytePairEncoding::applyBPE(const std::string& text) const {
    // Convert text to Unicode codepoints
    std::ostringstream unicodeText;
    for (uint8_t byte : text) {
        uint32_t codepoint = byteToUnicode(byte);
        // Convert codepoint to UTF-8 (only up to 3 bytes needed for our mapping)
        if (codepoint < 0x80) {
            unicodeText << static_cast<char>(codepoint);
        } else if (codepoint < 0x800) {
            unicodeText << static_cast<char>(0xC0 | (codepoint >> 6));
            unicodeText << static_cast<char>(0x80 | (codepoint & 0x3F));
        } else if (codepoint < 0x10000) {
            unicodeText << static_cast<char>(0xE0 | (codepoint >> 12));
            unicodeText << static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
            unicodeText << static_cast<char>(0x80 | (codepoint & 0x3F));
        }
    }
    
    std::string processedText = unicodeText.str();
    
    // Check if the entire text is in vocabulary
    int32_t id = vocab_->encode(processedText);
    if (id >= 0) {
        return {id};
    }
    
    // Convert to runes (UTF-8 characters)
    std::vector<std::string> runes;
    for (size_t i = 0; i < processedText.length(); ) {
        size_t len = 1;
        unsigned char c = processedText[i];
        if ((c & 0x80) == 0) {
            len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            len = 4;
        }
        
        if (i + len <= processedText.length()) {
            runes.push_back(processedText.substr(i, len));
        }
        i += len;
    }
    
    if (runes.empty()) {
        return {};
    }
    
    // Initialize merges (store indices into the runes array)
    std::vector<Merge> merges(runes.size());
    for (size_t i = 0; i < runes.size(); ++i) {
        merges[i].prev = (i > 0) ? static_cast<int>(i - 1) : -1;
        merges[i].next = (i < runes.size() - 1) ? static_cast<int>(i + 1) : -1;
        merges[i].runes.clear();
        merges[i].runes.push_back(static_cast<uint32_t>(i));
    }
    
    // Priority queue for pairs
    std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> pairs;
    
    // Add initial pairs
    for (size_t i = 0; i < runes.size() - 1; ++i) {
        int rank = vocab_->getMergeRank(runes[i], runes[i + 1]);
        if (rank >= 0) {
            pairs.emplace(static_cast<int>(i), static_cast<int>(i + 1), rank, runes[i] + runes[i + 1]);
        }
    }
    
    // Apply BPE merges
    while (!pairs.empty()) {
        Pair pair = pairs.top();
        pairs.pop();
        
        // Skip if indices are out of bounds
        if (pair.a < 0 || pair.b < 0 || 
            pair.a >= static_cast<int>(merges.size()) || 
            pair.b >= static_cast<int>(merges.size())) {
            continue;
        }
        
        Merge& left = merges[pair.a];
        Merge& right = merges[pair.b];
        
        // Check if merge is still valid
        if (left.runes.empty() || right.runes.empty()) {
            continue;
        }
        
        std::string leftStr, rightStr;
        for (const auto& idx : left.runes) leftStr += runes[idx];
        for (const auto& idx : right.runes) rightStr += runes[idx];
        
        if (leftStr + rightStr != pair.value) {
            continue;
        }
        
        // Check if merged token exists in vocabulary
        if (vocab_->encode(pair.value) < 0) {
            continue;
        }
        
        // Perform merge
        left.runes.insert(left.runes.end(), right.runes.begin(), right.runes.end());
        right.runes.clear();
        
        left.next = right.next;
        if (right.next >= 0) {
            merges[right.next].prev = pair.a;
        }
        
        // Add new pairs
        if (left.prev >= 0) {
            std::string prevStr;
            for (const auto& idx : merges[left.prev].runes) prevStr += runes[idx];
            int rank = vocab_->getMergeRank(prevStr, pair.value);
            if (rank >= 0) {
                pairs.emplace(left.prev, pair.a, rank, prevStr + pair.value);
            }
        }
        
        if (left.next >= 0) {
            std::string nextStr;
            for (const auto& idx : merges[left.next].runes) nextStr += runes[idx];
            int rank = vocab_->getMergeRank(pair.value, nextStr);
            if (rank >= 0) {
                pairs.emplace(pair.a, left.next, rank, pair.value + nextStr);
            }
        }
    }
    
    // Collect final tokens
    std::vector<int32_t> result;
    for (const auto& merge : merges) {
        if (!merge.runes.empty()) {
            std::string token;
            for (const auto& idx : merge.runes) {
                token += runes[idx];
            }
            int32_t id = vocab_->encode(token);
            if (id >= 0) {
                result.push_back(id);
            }
        }
    }
    
    return result;
}

} // namespace model
} // namespace duorou