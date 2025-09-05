#include "text_processor.h"
#include "sentencepiece_processor.h"
#include "bpe_processor.h"
#include <algorithm>
#include <sstream>
#include <iostream>

namespace duorou {
namespace extensions {
namespace ollama {

void Vocabulary::initialize(const std::vector<std::string>& values,
                           const std::vector<int32_t>& types,
                           const std::vector<float>& scores,
                           const std::vector<std::string>& merges) {
    values_ = values;
    types_ = types;
    scores_ = scores;
    merges_ = merges;
    
    // Reset cache flags
    token_to_id_cached_ = false;
    merge_map_cached_ = false;
    special_vocab_cached_ = false;
    token_to_id_.clear();
    merge_map_.clear();
    special_vocab_.clear();
}

int32_t Vocabulary::encode(const std::string& token) const {
    if (!token_to_id_cached_) {
        buildTokenToIdCache();
    }
    
    auto it = token_to_id_.find(token);
    if (it != token_to_id_.end()) {
        return it->second;
    }
    return -1;
}

std::string Vocabulary::decode(int32_t id) const {
    if (id >= 0 && id < static_cast<int32_t>(values_.size())) {
        return values_[id];
    }
    return "";
}

bool Vocabulary::is(int32_t id, Special special) const {
    switch (special) {
        case SPECIAL_BOS:
            return std::find(bos_tokens_.begin(), bos_tokens_.end(), id) != bos_tokens_.end();
        case SPECIAL_EOS:
            return std::find(eos_tokens_.begin(), eos_tokens_.end(), id) != eos_tokens_.end();
        default:
            return false;
    }
}

void Vocabulary::setBOS(const std::vector<int32_t>& bos_tokens, bool add_bos) {
    bos_tokens_ = bos_tokens;
    add_bos_ = add_bos;
}

void Vocabulary::setEOS(const std::vector<int32_t>& eos_tokens, bool add_eos) {
    eos_tokens_ = eos_tokens;
    add_eos_ = add_eos;
}

int Vocabulary::merge(const std::string& left, const std::string& right) const {
    if (!merge_map_cached_) {
        buildMergeMapCache();
    }
    
    std::string merge_key = left + " " + right;
    auto it = merge_map_.find(merge_key);
    if (it != merge_map_.end()) {
        return static_cast<int>(it->second);
    }
    return -1;
}

std::vector<std::string> Vocabulary::getSpecialVocabulary() const {
    if (!special_vocab_cached_) {
        buildSpecialVocabCache();
    }
    return special_vocab_;
}

std::vector<int32_t> Vocabulary::addSpecials(const std::vector<int32_t>& ids) const {
    std::vector<int32_t> result = ids;
    
    if (add_bos_ && !bos_tokens_.empty()) {
        // Check if BOS token already exists
        if (std::find(bos_tokens_.begin(), bos_tokens_.end(), result[0]) == bos_tokens_.end()) {
            result.insert(result.begin(), bos_tokens_[0]);
        }
    }
    
    if (add_eos_ && !eos_tokens_.empty()) {
        // Check if EOS token already exists
        if (std::find(eos_tokens_.begin(), eos_tokens_.end(), result.back()) == eos_tokens_.end()) {
            result.push_back(eos_tokens_[0]);
        }
    }
    
    return result;
}

void Vocabulary::buildTokenToIdCache() const {
    token_to_id_.clear();
    for (size_t i = 0; i < values_.size(); ++i) {
        token_to_id_[values_[i]] = static_cast<int32_t>(i);
    }
    token_to_id_cached_ = true;
}

void Vocabulary::buildMergeMapCache() const {
    merge_map_.clear();
    for (size_t i = 0; i < merges_.size(); ++i) {
        merge_map_[merges_[i]] = static_cast<int32_t>(i);
    }
    merge_map_cached_ = true;
}

void Vocabulary::buildSpecialVocabCache() const {
    special_vocab_.clear();
    for (size_t i = 0; i < values_.size(); ++i) {
        if (types_[i] == TOKEN_TYPE_CONTROL || types_[i] == TOKEN_TYPE_USER_DEFINED) {
            special_vocab_.push_back(values_[i]);
        }
    }
    special_vocab_cached_ = true;
}

// Factory function implementation
std::unique_ptr<TextProcessor> createTextProcessor(const std::string& type, 
                                                  std::shared_ptr<Vocabulary> vocab,
                                                  const std::string& pre_tokenizer_regex) {
    if (type == "sentencepiece") {
        return std::make_unique<SentencePieceProcessor>(vocab);
    } else if (type == "bpe") {
        return std::make_unique<BPEProcessor>(pre_tokenizer_regex, vocab);
    }
    return nullptr;
}

} // namespace ollama
} // namespace extensions
} // namespace duorou