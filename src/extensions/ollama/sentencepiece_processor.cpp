#include "sentencepiece_processor.h"
#include <algorithm>
#include <sstream>
#include <iostream>
#include <codecvt>
#include <locale>
#include <iomanip>

namespace duorou {
namespace extensions {
namespace ollama {

const std::string SentencePieceProcessor::WHITESPACE_SEP = "▁";

SentencePieceProcessor::SentencePieceProcessor(std::shared_ptr<Vocabulary> vocab)
    : vocab_(vocab), max_token_len_(0) {
    
    // Calculate maximum token length
    const auto& values = vocab_->getValues();
    const auto& types = vocab_->getTypes();
    
    for (size_t i = 0; i < values.size(); ++i) {
        switch (types[i]) {
            case TOKEN_TYPE_NORMAL:
            case TOKEN_TYPE_USER_DEFINED:
            case TOKEN_TYPE_UNUSED:
                max_token_len_ = std::max(max_token_len_, static_cast<int>(values[i].length()));
                break;
            default:
                break;
        }
    }
}

std::vector<int32_t> SentencePieceProcessor::encode(const std::string& text, bool add_special) {
    // Process special tokens first
    std::vector<Fragment> fragments = processSpecialTokens(text);
    
    std::vector<int32_t> ids;
    for (const auto& frag : fragments) {
        if (!frag.ids.empty()) {
            ids.insert(ids.end(), frag.ids.begin(), frag.ids.end());
            continue;
        }
        
        // Replace spaces with whitespace separator
        std::string processed_text = frag.value;
        size_t pos = 0;
        while ((pos = processed_text.find(" ", pos)) != std::string::npos) {
            processed_text.replace(pos, 1, WHITESPACE_SEP);
            pos += WHITESPACE_SEP.length();
        }
        
        // Try direct vocabulary lookup first
        int32_t direct_id = vocab_->encode(processed_text);
        if (direct_id >= 0) {
            ids.push_back(direct_id);
            continue;
        }
        
        // Tokenize using merge algorithm
        std::vector<int32_t> fragment_ids = tokenizeFragment(processed_text);
        ids.insert(ids.end(), fragment_ids.begin(), fragment_ids.end());
    }
    
    if (add_special && !ids.empty()) {
        ids = vocab_->addSpecials(ids);
    }
    
    return ids;
}

std::string SentencePieceProcessor::decode(const std::vector<int32_t>& tokens) {
    std::stringstream result;
    bool first_token = true;
    
    for (int32_t token_id : tokens) {
        std::string token_str = vocab_->decode(token_id);
        
        // Skip special tokens during decoding
        if (vocab_->is(token_id, SPECIAL_BOS) || vocab_->is(token_id, SPECIAL_EOS)) {
            continue;
        }
        
        // Handle Qwen2-specific special tokens
        if (token_str == "<|im_start|>" || token_str == "<|im_end|>" || 
            token_str == "<|endoftext|>" || token_str.find("<|vision_") == 0 ||
            token_str.find("<|image_") == 0 || token_str.find("<|video_") == 0) {
            continue;
        }
        
        // Skip invalid tokens (PAD tokens, UNK tokens, etc.)
        if (token_str.empty() || token_str == "<unk>" || 
            token_str.find("[PAD") == 0 || token_str.find("<pad>") == 0 ||
            token_str.find("<|pad|>") == 0) {
            continue;
        }
        
        // Handle Ġ prefix (GPT-style space encoding)
        if (token_str.length() >= 3 && token_str.substr(0, 3) == "Ġ") {
            if (!first_token) {
                result << " ";
            }
            token_str = token_str.substr(3); // Remove Ġ prefix
        }
        // Handle ▁ prefix (SentencePiece-style space encoding)
        else if (token_str.find(WHITESPACE_SEP) == 0) {
            if (!first_token) {
                result << " ";
            }
            token_str = token_str.substr(WHITESPACE_SEP.length());
        }
        // Replace internal whitespace separators with spaces
        else {
            size_t pos = 0;
            while ((pos = token_str.find(WHITESPACE_SEP, pos)) != std::string::npos) {
                token_str.replace(pos, WHITESPACE_SEP.length(), " ");
                pos += 1;
            }
        }
        
        // Handle byte tokens like "<0xEA>"
        if (token_str.length() == 6 && 
            token_str.substr(0, 3) == "<0x" && 
            token_str.back() == '>') {
            try {
                std::string hex_str = token_str.substr(3, 2);
                unsigned long byte_val = std::stoul(hex_str, nullptr, 16);
                if (byte_val <= 255) {
                    result << static_cast<char>(byte_val);
                    first_token = false;
                    continue;
                }
            } catch (const std::exception&) {
                // Fall through to normal string handling
            }
        }
        
        // Handle Chinese characters and other Unicode properly
        if (!token_str.empty()) {
            result << token_str;
        }
        
        first_token = false;
    }
    
    return result.str();
}

bool SentencePieceProcessor::is(int32_t token_id, Special special) const {
    return vocab_->is(token_id, special);
}

const Vocabulary* SentencePieceProcessor::getVocabulary() const {
    return vocab_.get();
}

size_t SentencePieceProcessor::getVocabSize() const {
    return vocab_->size();
}

std::vector<Fragment> SentencePieceProcessor::processSpecialTokens(const std::string& text) const {
    std::vector<Fragment> fragments = {Fragment(text)};
    
    // Process each special token
    for (const std::string& special : vocab_->getSpecialVocabulary()) {
        int32_t special_id = vocab_->encode(special);
        if (special_id < 0) continue;
        
        for (size_t i = 0; i < fragments.size(); ++i) {
            Fragment& frag = fragments[i];
            if (!frag.ids.empty()) continue; // Skip already processed fragments
            
            std::vector<Fragment> middle;
            size_t pos = frag.value.find(special);
            
            if (pos == std::string::npos) {
                middle.push_back(frag);
            } else {
                if (pos > 0) {
                    middle.push_back(Fragment(frag.value.substr(0, pos)));
                }
                middle.push_back(Fragment(special, {special_id}));
                
                std::string rest = frag.value.substr(pos + special.length());
                if (!rest.empty()) {
                    middle.push_back(Fragment(rest));
                }
            }
            
            // Replace current fragment with processed fragments
            fragments.erase(fragments.begin() + i);
            fragments.insert(fragments.begin() + i, middle.begin(), middle.end());
            i += middle.size() - 1; // Adjust index
        }
    }
    
    return fragments;
}

std::vector<int32_t> SentencePieceProcessor::tokenizeFragment(const std::string& text) const {
    std::vector<char32_t> runes = stringToRunes(text);
    if (runes.empty()) return {};
    
    std::vector<SPMerge> merges(runes.size());
    for (size_t i = 0; i < runes.size(); ++i) {
        merges[i].p = static_cast<int>(i) - 1;
        merges[i].n = static_cast<int>(i) + 1;
        merges[i].runes = {runes[i]};
    }
    
    // Priority queue for merge candidates
    std::priority_queue<std::shared_ptr<Candidate>, 
                       std::vector<std::shared_ptr<Candidate>>, 
                       CandidateComparator> queue;
    
    // Helper function to create merge candidates
    auto createCandidate = [&](int a, int b) -> std::shared_ptr<Candidate> {
        if (a < 0 || b >= static_cast<int>(runes.size())) {
            return nullptr;
        }
        
        std::string left = runesToString(merges[a].runes);
        std::string right = runesToString(merges[b].runes);
        std::string combined = left + right;
        
        int32_t token_id = vocab_->encode(combined);
        if (token_id >= 0) {
            const auto& scores = vocab_->getScores();
            float score = (token_id < static_cast<int32_t>(scores.size())) ? scores[token_id] : 0.0f;
            return std::make_shared<Candidate>(a, b, score, static_cast<int>(combined.length()));
        }
        
        return nullptr;
    };
    
    // Initialize queue with adjacent pairs
    for (size_t i = 0; i < runes.size() - 1; ++i) {
        auto candidate = createCandidate(static_cast<int>(i), static_cast<int>(i) + 1);
        if (candidate) {
            queue.push(candidate);
        }
    }
    
    // Process merges
    while (!queue.empty()) {
        auto candidate = queue.top();
        queue.pop();
        
        SPMerge& left = merges[candidate->a];
        SPMerge& right = merges[candidate->b];
        
        // Check if merge is still valid
        if (left.runes.empty() || right.runes.empty() ||
            static_cast<int>(runesToString(left.runes).length() + runesToString(right.runes).length()) != candidate->size) {
            continue;
        }
        
        // Perform merge
        left.runes.insert(left.runes.end(), right.runes.begin(), right.runes.end());
        right.runes.clear();
        left.n = right.n;
        
        if (right.n < static_cast<int>(merges.size())) {
            merges[right.n].p = candidate->a;
        }
        
        // Add new candidates
        if (auto new_candidate = createCandidate(left.p, candidate->a)) {
            queue.push(new_candidate);
        }
        if (auto new_candidate = createCandidate(candidate->a, left.n)) {
            queue.push(new_candidate);
        }
    }
    
    // Collect final tokens
    std::vector<int32_t> result;
    for (const auto& merge : merges) {
        if (!merge.runes.empty()) {
            std::string token_str = runesToString(merge.runes);
            int32_t token_id = vocab_->encode(token_str);
            
            if (token_id >= 0) {
                result.push_back(token_id);
            } else {
                // Fallback to byte tokenization
                for (char c : token_str) {
                    std::stringstream ss;
                    ss << "<0x" << std::hex << std::uppercase << std::setfill('0') << std::setw(2) 
                       << static_cast<unsigned char>(c) << ">";
                    int32_t byte_token_id = vocab_->encode(ss.str());
                    if (byte_token_id >= 0) {
                        result.push_back(byte_token_id);
                    }
                }
            }
        }
    }
    
    return result;
}

std::vector<char32_t> SentencePieceProcessor::stringToRunes(const std::string& text) const {
    std::vector<char32_t> runes;
    
    // Simple UTF-8 to UTF-32 conversion
    for (size_t i = 0; i < text.length(); ) {
        unsigned char c = text[i];
        char32_t codepoint;
        
        if (c < 0x80) {
            codepoint = c;
            i += 1;
        } else if ((c & 0xE0) == 0xC0) {
            if (i + 1 >= text.length()) break;
            codepoint = ((c & 0x1F) << 6) | (text[i + 1] & 0x3F);
            i += 2;
        } else if ((c & 0xF0) == 0xE0) {
            if (i + 2 >= text.length()) break;
            codepoint = ((c & 0x0F) << 12) | ((text[i + 1] & 0x3F) << 6) | (text[i + 2] & 0x3F);
            i += 3;
        } else if ((c & 0xF8) == 0xF0) {
            if (i + 3 >= text.length()) break;
            codepoint = ((c & 0x07) << 18) | ((text[i + 1] & 0x3F) << 12) | 
                       ((text[i + 2] & 0x3F) << 6) | (text[i + 3] & 0x3F);
            i += 4;
        } else {
            // Invalid UTF-8, skip
            i += 1;
            continue;
        }
        
        runes.push_back(codepoint);
    }
    
    return runes;
}

std::string SentencePieceProcessor::runesToString(const std::vector<char32_t>& runes) const {
    std::string result;
    
    for (char32_t codepoint : runes) {
        if (codepoint < 0x80) {
            result += static_cast<char>(codepoint);
        } else if (codepoint < 0x800) {
            result += static_cast<char>(0xC0 | (codepoint >> 6));
            result += static_cast<char>(0x80 | (codepoint & 0x3F));
        } else if (codepoint < 0x10000) {
            result += static_cast<char>(0xE0 | (codepoint >> 12));
            result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
            result += static_cast<char>(0x80 | (codepoint & 0x3F));
        } else {
            result += static_cast<char>(0xF0 | (codepoint >> 18));
            result += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F));
            result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
            result += static_cast<char>(0x80 | (codepoint & 0x3F));
        }
    }
    
    return result;
}

} // namespace ollama
} // namespace extensions
} // namespace duorou