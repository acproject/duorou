#include "vocabulary.h"
#include "../utils/string_utils.h"
#include <algorithm>
#include <sstream>

namespace duorou {
namespace model {

void Vocabulary::initialize(const std::vector<std::string>& values,
                           const std::vector<int32_t>& types,
                           const std::vector<float>& scores,
                           const std::vector<std::string>& merges) {
    // 解码可能包含十六进制转义序列的token字符串
    values_ = utils::decodeTokenStrings(values);
    types_ = types;
    scores_ = scores;
    merges_ = merges;
    
    // Clear cached data (once_flag cannot be reset, so we clear the cached results)
    specialTokens_.clear();
    tokenToId_.clear();
    mergeMap_.clear();

    // Try to autodetect PAD/UNK ids if any
    autodetectPadUnk();
}

bool Vocabulary::isSpecial(int32_t id, Special special) const {
    switch (special) {
        case Special::BOS:
            return std::find(bos_.begin(), bos_.end(), id) != bos_.end();
        case Special::EOS:
            return std::find(eos_.begin(), eos_.end(), id) != eos_.end();
        case Special::PAD:
            return std::find(pad_.begin(), pad_.end(), id) != pad_.end();
        case Special::UNK:
            return std::find(unk_.begin(), unk_.end(), id) != unk_.end();
        default:
            return false;
    }
}

std::vector<int32_t> Vocabulary::addSpecials(const std::vector<int32_t>& ids) const {
    std::vector<int32_t> result = ids;
    
    // Add BOS token if needed
    if (addBOS_ && !bos_.empty()) {
        if (result.empty() || std::find(bos_.begin(), bos_.end(), result[0]) == bos_.end()) {
            result.insert(result.begin(), bos_[0]);
        }
    }
    
    // Add EOS token if needed
    if (addEOS_ && !eos_.empty()) {
        if (result.empty() || std::find(eos_.begin(), eos_.end(), result.back()) == eos_.end()) {
            result.push_back(eos_[0]);
        }
    }
    
    return result;
}

int32_t Vocabulary::encode(const std::string& token) const {
    std::call_once(valuesOnce_, [this]() { buildTokenMap(); });
    
    auto it = tokenToId_.find(token);
    return (it != tokenToId_.end()) ? it->second : -1;
}

std::string Vocabulary::decode(int32_t id) const {
    if (id >= 0 && static_cast<size_t>(id) < values_.size()) {
        const std::string& token_text = values_[id];
        
        // Handle byte tokens in <0xXX> format
        if (token_text.length() == 6 && 
            token_text.substr(0, 3) == "<0x" && 
            token_text.back() == '>') {
            
            std::string hex_str = token_text.substr(3, 2);
            try {
                int byte_val = std::stoi(hex_str, nullptr, 16);
                return std::string(1, static_cast<char>(byte_val));
            } catch (const std::exception&) {
                // If hex parsing fails, return original token
                return token_text;
            }
        }
        
        // Handle placeholder tokens like <token_146895>
        // These often represent byte values that cannot be directly encoded as UTF-8
        if (token_text.length() > 7 && 
            token_text.substr(0, 7) == "<token_" && 
            token_text.back() == '>') {
            
            // Try to extract the token ID and convert it to a byte value
            std::string id_str = token_text.substr(7, token_text.length() - 8);
            try {
                int token_id = std::stoi(id_str);
                // For many tokenizers, high token IDs represent byte values
                // Try to map the token ID to a byte value
                if (token_id >= 0 && token_id <= 255) {
                    return std::string(1, static_cast<char>(token_id));
                } else if (token_id > 255) {
                    // For token IDs > 255, try to map them to byte values
                    // This is a heuristic approach - different tokenizers may use different mappings
                    int byte_val = token_id % 256;
                    return std::string(1, static_cast<char>(byte_val));
                }
            } catch (const std::exception&) {
                // If parsing fails, return original token
                return token_text;
            }
        }
        
        // For GPT-2 byte-level BPE, apply Unicode to byte decoding
        return decodeText(token_text);
    }
    return "";
}

std::vector<std::string> Vocabulary::getSpecialVocabulary() const {
    std::call_once(specialOnce_, [this]() { buildSpecialTokens(); });
    return specialTokens_;
}

int Vocabulary::getMergeRank(const std::string& left, const std::string& right) const {
    std::call_once(mergeOnce_, [this]() { buildMergeMap(); });
    
    std::string mergeKey = left + " " + right;
    auto it = mergeMap_.find(mergeKey);
    return (it != mergeMap_.end()) ? static_cast<int>(it->second) : -1;
}

void Vocabulary::setBOS(const std::vector<int32_t>& bos, bool addBOS) {
    bos_ = bos;
    addBOS_ = addBOS;
}

void Vocabulary::setEOS(const std::vector<int32_t>& eos, bool addEOS) {
    eos_ = eos;
    addEOS_ = addEOS;
}

void Vocabulary::setPAD(const std::vector<int32_t>& pad) {
    pad_ = pad;
}

void Vocabulary::setUNK(const std::vector<int32_t>& unk) {
    unk_ = unk;
}

int32_t Vocabulary::getSpecialId(Special special) const {
    const std::vector<int32_t>* vec = nullptr;
    switch (special) {
        case Special::PAD: vec = &pad_; break;
        case Special::UNK: vec = &unk_; break;
        case Special::BOS: vec = &bos_; break;
        case Special::EOS: vec = &eos_; break;
        default: break;
    }
    if (vec && !vec->empty()) return (*vec)[0];
    return -1;
}

void Vocabulary::buildTokenMap() const {
    tokenToId_.clear();
    for (size_t i = 0; i < values_.size(); ++i) {
        tokenToId_[values_[i]] = static_cast<int32_t>(i);
    }
}

void Vocabulary::buildSpecialTokens() const {
    specialTokens_.clear();
    for (size_t i = 0; i < values_.size(); ++i) {
        if (i < types_.size() && 
            (types_[i] == TOKEN_TYPE_CONTROL || types_[i] == TOKEN_TYPE_USER_DEFINED)) {
            specialTokens_.push_back(values_[i]);
        }
    }
}

void Vocabulary::buildMergeMap() const {
    mergeMap_.clear();
    for (size_t i = 0; i < merges_.size(); ++i) {
        mergeMap_[merges_[i]] = static_cast<int32_t>(i);
    }
}

std::string Vocabulary::decodeText(const std::string& text) const {
    std::string decoded_text;
    
    // Convert UTF-8 string to Unicode code points
    std::vector<uint32_t> codepoints;
    size_t i = 0;
    while (i < text.length()) {
        uint32_t codepoint = 0;
        uint8_t byte = static_cast<uint8_t>(text[i]);
        
        if (byte < 0x80) {
            // ASCII character
            codepoint = byte;
            i++;
        } else if ((byte & 0xE0) == 0xC0) {
            // 2-byte UTF-8 sequence
            if (i + 1 < text.length()) {
                codepoint = ((byte & 0x1F) << 6) | (static_cast<uint8_t>(text[i + 1]) & 0x3F);
                i += 2;
            } else {
                i++;
            }
        } else if ((byte & 0xF0) == 0xE0) {
            // 3-byte UTF-8 sequence
            if (i + 2 < text.length()) {
                codepoint = ((byte & 0x0F) << 12) | 
                           ((static_cast<uint8_t>(text[i + 1]) & 0x3F) << 6) | 
                           (static_cast<uint8_t>(text[i + 2]) & 0x3F);
                i += 3;
            } else {
                i++;
            }
        } else if ((byte & 0xF8) == 0xF0) {
            // 4-byte UTF-8 sequence
            if (i + 3 < text.length()) {
                codepoint = ((byte & 0x07) << 18) | 
                           ((static_cast<uint8_t>(text[i + 1]) & 0x3F) << 12) | 
                           ((static_cast<uint8_t>(text[i + 2]) & 0x3F) << 6) | 
                           (static_cast<uint8_t>(text[i + 3]) & 0x3F);
                i += 4;
            } else {
                i++;
            }
        } else {
            // Invalid UTF-8, skip
            i++;
        }
        
        if (codepoint > 0) {
            codepoints.push_back(codepoint);
        }
    }
    
    // Convert Unicode code points to bytes using GPT-2 mapping
    for (uint32_t cpt : codepoints) {
        uint8_t byte_val = 0;
        
        // GPT-2 byte-level BPE mapping
        if (cpt >= 0x21 && cpt <= 0x7E) {
            // Printable ASCII: '!' to '~'
            byte_val = static_cast<uint8_t>(cpt);
        } else if (cpt >= 0xA1 && cpt <= 0xAC) {
            // Extended ASCII: '¡' to '¬'
            byte_val = static_cast<uint8_t>(cpt);
        } else if (cpt >= 0xAE && cpt <= 0xFF) {
            // Extended ASCII: '®' to 'ÿ'
            byte_val = static_cast<uint8_t>(cpt);
        } else if (cpt >= 256) {
            // High Unicode code points map to unmapped bytes
            // This is a simplified mapping - in practice, GPT-2 uses a specific mapping table
            byte_val = static_cast<uint8_t>(cpt - 256);
        } else {
            // For other code points, use the code point value directly if it fits in a byte
            byte_val = static_cast<uint8_t>(cpt);
        }
        
        decoded_text += static_cast<char>(byte_val);
    }
    
    return decoded_text;
}

void Vocabulary::autodetectPadUnk() {
    // Build token map if not built
    std::call_once(valuesOnce_, [this]() { buildTokenMap(); });
    auto setIfFound = [&](const std::string& key, std::vector<int32_t>& dst) {
        auto it = tokenToId_.find(key);
        if (it != tokenToId_.end()) {
            if (dst.empty()) dst.push_back(it->second);
        }
    };
    // Common representations
    setIfFound("<pad>", pad_);
    setIfFound("<PAD>", pad_);
    setIfFound("[PAD]", pad_);

    setIfFound("<unk>", unk_);
    setIfFound("<UNK>", unk_);
    setIfFound("[UNK]", unk_);
}

} // namespace model
} // namespace duorou