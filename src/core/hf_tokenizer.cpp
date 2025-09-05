#include "hf_tokenizer.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <regex>

namespace duorou {

// Simple JSON parser for tokenizer files
class TokenizerJsonParser {
public:
    static std::unordered_map<std::string, int32_t> parseVocab(const std::string& json) {
        std::unordered_map<std::string, int32_t> vocab;
        
        size_t pos = 0;
        while (pos < json.length()) {
            // Find token string
            size_t token_start = json.find('\"', pos);
            if (token_start == std::string::npos) break;
            
            size_t token_end = token_start + 1;
            while (token_end < json.length() && json[token_end] != '\"') {
                if (json[token_end] == '\\') {
                    token_end += 2;  // Skip escaped character
                } else {
                    token_end++;
                }
            }
            
            if (token_end >= json.length()) break;
            
            std::string token = json.substr(token_start + 1, token_end - token_start - 1);
            
            // Find colon
            size_t colon_pos = json.find(':', token_end);
            if (colon_pos == std::string::npos) break;
            
            // Find token ID
            size_t id_start = colon_pos + 1;
            while (id_start < json.length() && std::isspace(json[id_start])) {
                id_start++;
            }
            
            size_t id_end = id_start;
            while (id_end < json.length() && std::isdigit(json[id_end])) {
                id_end++;
            }
            
            if (id_end > id_start) {
                try {
                    int32_t token_id = std::stoi(json.substr(id_start, id_end - id_start));
                    vocab[token] = token_id;
                } catch (const std::exception& e) {
                    std::cerr << "Error parsing token ID for '" << token << "': " << e.what() << std::endl;
                }
            }
            
            pos = id_end;
        }
        
        return vocab;
    }
    
    static std::unordered_map<std::string, int32_t> parseSpecialTokens(const std::string& json) {
        std::unordered_map<std::string, int32_t> special_tokens;
        
        // Find "added_tokens_decoder" section
        size_t decoder_pos = json.find("\"added_tokens_decoder\"");
        if (decoder_pos == std::string::npos) {
            return special_tokens;
        }
        
        // Find opening brace
        size_t brace_start = json.find("{", decoder_pos);
        if (brace_start == std::string::npos) {
            return special_tokens;
        }
        
        size_t pos = brace_start + 1;
        while (pos < json.length()) {
            // Find token ID (numeric key)
            size_t id_start = json.find('\"', pos);
            if (id_start == std::string::npos) break;
            
            size_t id_end = json.find('\"', id_start + 1);
            if (id_end == std::string::npos) break;
            
            std::string id_str = json.substr(id_start + 1, id_end - id_start - 1);
            
            // Check if this is a numeric token ID
            bool is_numeric = true;
            for (char c : id_str) {
                if (!std::isdigit(c)) {
                    is_numeric = false;
                    break;
                }
            }
            
            if (!is_numeric) {
                // Skip non-numeric keys
                pos = id_end + 1;
                continue;
            }
            
            int32_t token_id;
            try {
                token_id = std::stoi(id_str);
            } catch (const std::exception& e) {
                pos = id_end + 1;
                continue;
            }
            
            // Find content
            size_t content_pos = json.find("\"content\"", id_end);
            if (content_pos == std::string::npos) {
                pos = id_end + 1;
                continue;
            }
            
            size_t content_start = json.find('\"', content_pos + 9);
            if (content_start == std::string::npos) {
                pos = id_end + 1;
                continue;
            }
            
            size_t content_end = content_start + 1;
            while (content_end < json.length() && json[content_end] != '\"') {
                if (json[content_end] == '\\') {
                    content_end += 2;
                } else {
                    content_end++;
                }
            }
            
            if (content_end >= json.length()) break;
            
            std::string content = json.substr(content_start + 1, content_end - content_start - 1);
            special_tokens[content] = token_id;
            
            pos = content_end + 1;
        }
        
        return special_tokens;
    }
};

HFTokenizer::HFTokenizer() 
    : vocab_size_(0), eos_token_id_(-1), bos_token_id_(-1), 
      pad_token_id_(-1), unk_token_id_(-1),
      vision_start_token_(151652), vision_end_token_(151656) {
}

HFTokenizer::~HFTokenizer() {}

bool HFTokenizer::loadFromDirectory(const std::string& model_dir) {
    model_dir_ = model_dir;
    
    // Load vocabulary
    if (!loadVocabulary(model_dir + "/vocab.json")) {
        std::cerr << "Failed to load vocabulary" << std::endl;
        return false;
    }
    
    // Load merges
    if (!loadMerges(model_dir + "/merges.txt")) {
        std::cerr << "Failed to load merges" << std::endl;
        return false;
    }
    
    // Load tokenizer config
    if (!loadTokenizerConfig(model_dir + "/tokenizer_config.json")) {
        std::cerr << "Failed to load tokenizer config" << std::endl;
        return false;
    }
    
    return true;
}

bool HFTokenizer::loadVocabulary(const std::string& vocab_file) {
    std::ifstream file(vocab_file);
    if (!file.is_open()) {
        return false;
    }
    
    std::string json_content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
    file.close();
    
    token_to_id_ = TokenizerJsonParser::parseVocab(json_content);
    
    // Build reverse mapping
    for (const auto& pair : token_to_id_) {
        id_to_token_[pair.second] = pair.first;
    }
    
    vocab_size_ = token_to_id_.size();
    return !token_to_id_.empty();
}

bool HFTokenizer::loadMerges(const std::string& merges_file) {
    std::ifstream file(merges_file);
    if (!file.is_open()) {
        return false;
    }
    
    std::string line;
    bool first_line = true;
    
    while (std::getline(file, line)) {
        if (first_line) {
            first_line = false;
            continue;  // Skip header
        }
        
        if (line.empty()) continue;
        
        size_t space_pos = line.find(' ');
        if (space_pos != std::string::npos) {
            std::string first = line.substr(0, space_pos);
            std::string second = line.substr(space_pos + 1);
            bpe_merges_.push_back({first, second});
        }
    }
    
    return true;
}

bool HFTokenizer::loadTokenizerConfig(const std::string& config_file) {
    std::ifstream file(config_file);
    if (!file.is_open()) {
        return false;
    }
    
    std::string json_content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
    file.close();
    
    special_tokens_ = TokenizerJsonParser::parseSpecialTokens(json_content);
    
    // Set special token IDs
    auto it = special_tokens_.find("<|endoftext|>");
    if (it != special_tokens_.end()) {
        eos_token_id_ = it->second;
    }
    
    it = special_tokens_.find("<|im_start|>");
    if (it != special_tokens_.end()) {
        bos_token_id_ = it->second;
    }
    
    // Set pretokenization regex for Qwen
    // Very simple regex to avoid compilation issues
    pretokenize_regex_ = std::regex(R"([a-zA-Z]+|[0-9]+|[^\s\w]+|\s+)");
    
    return true;
}

std::vector<std::string> HFTokenizer::preTokenize(const std::string& text) const {
    std::vector<std::string> tokens;
    
    std::sregex_iterator iter(text.begin(), text.end(), pretokenize_regex_);
    std::sregex_iterator end;
    
    for (; iter != end; ++iter) {
        std::string match = iter->str();
        if (!match.empty()) {
            tokens.push_back(match);
        }
    }
    
    return tokens;
}

std::string HFTokenizer::applyBPE(const std::string& word) const {
    if (word.length() <= 1) {
        return word;
    }
    
    std::vector<std::string> word_chars;
    for (char c : word) {
        word_chars.push_back(std::string(1, c));
    }
    
    while (word_chars.size() > 1) {
        int best_merge_idx = -1;
        size_t best_merge_pos = 0;
        
        for (size_t i = 0; i < bpe_merges_.size(); ++i) {
            const auto& merge = bpe_merges_[i];
            
            for (size_t j = 0; j < word_chars.size() - 1; ++j) {
                if (word_chars[j] == merge.first && word_chars[j + 1] == merge.second) {
                    if (best_merge_idx == -1 || i < static_cast<size_t>(best_merge_idx)) {
                        best_merge_idx = i;
                        best_merge_pos = j;
                    }
                }
            }
        }
        
        if (best_merge_idx == -1) {
            break;
        }
        
        // Apply merge
        const auto& merge = bpe_merges_[best_merge_idx];
        word_chars[best_merge_pos] = merge.first + merge.second;
        word_chars.erase(word_chars.begin() + best_merge_pos + 1);
    }
    
    std::string result;
    for (const std::string& part : word_chars) {
        if (!result.empty()) result += " ";
        result += part;
    }
    
    return result;
}

std::vector<int32_t> HFTokenizer::encode(const std::string& text) const {
    std::vector<int32_t> token_ids;
    
    // Pre-tokenize
    std::vector<std::string> pretokens = preTokenize(text);
    
    for (const std::string& pretoken : pretokens) {
        // Apply BPE
        std::string bpe_result = applyBPE(pretoken);
        
        // Split by spaces and convert to token IDs
        std::stringstream ss(bpe_result);
        std::string token;
        
        while (ss >> token) {
            auto it = token_to_id_.find(token);
            if (it != token_to_id_.end()) {
                token_ids.push_back(it->second);
            } else if (unk_token_id_ != -1) {
                token_ids.push_back(unk_token_id_);
            }
        }
    }
    
    return token_ids;
}

std::string HFTokenizer::decode(const std::vector<int32_t>& token_ids) const {
    std::string result;
    
    for (int32_t token_id : token_ids) {
        auto it = id_to_token_.find(token_id);
        if (it != id_to_token_.end()) {
            std::string token = it->second;
            
            // Handle byte tokens (Ġ prefix indicates space)
            if (token.length() > 0 && token.substr(0, 1) == "Ġ") {
                result += " " + token.substr(1);
            } else {
                result += token;
            }
        }
    }
    
    return result;
}

bool HFTokenizer::isSpecialToken(int32_t token_id) const {
    for (const auto& pair : special_tokens_) {
        if (pair.second == token_id) {
            return true;
        }
    }
    return false;
}

bool HFTokenizer::isVisionToken(int32_t token_id) const {
    return token_id >= vision_start_token_ && token_id <= vision_end_token_;
}

std::string HFTokenizer::getTokenString(int32_t token_id) const {
    auto it = id_to_token_.find(token_id);
    return (it != id_to_token_.end()) ? it->second : "<unk>";
}

} // namespace duorou