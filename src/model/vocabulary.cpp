#include "vocabulary.h"
#include <algorithm>
#include <sstream>

namespace duorou {
namespace model {

void Vocabulary::initialize(const std::vector<std::string>& values,
                           const std::vector<int32_t>& types,
                           const std::vector<float>& scores,
                           const std::vector<std::string>& merges) {
    values_ = values;
    types_ = types;
    scores_ = scores;
    merges_ = merges;
    
    // Clear cached data (once_flag cannot be reset, so we clear the cached results)
    specialTokens_.clear();
    tokenToId_.clear();
    mergeMap_.clear();
}

bool Vocabulary::isSpecial(int32_t id, Special special) const {
    switch (special) {
        case Special::BOS:
            return std::find(bos_.begin(), bos_.end(), id) != bos_.end();
        case Special::EOS:
            return std::find(eos_.begin(), eos_.end(), id) != eos_.end();
        default:
            return false;
    }
}

std::vector<int32_t> Vocabulary::addSpecials(const std::vector<int32_t>& ids) const {
    std::vector<int32_t> result = ids;
    
    // Add BOS token if needed
    if (addBOS_ && !bos_.empty()) {
        if (std::find(bos_.begin(), bos_.end(), result[0]) == bos_.end()) {
            result.insert(result.begin(), bos_[0]);
        }
    }
    
    // Add EOS token if needed
    if (addEOS_ && !eos_.empty()) {
        if (std::find(eos_.begin(), eos_.end(), result.back()) == eos_.end()) {
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
        return values_[id];
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

} // namespace model
} // namespace duorou