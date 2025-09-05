#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <regex>

namespace duorou {

// HuggingFace tokenizer implementation
class HFTokenizer {
public:
    HFTokenizer();
    ~HFTokenizer();
    
    // Load tokenizer from directory
    bool loadFromDirectory(const std::string& model_dir);
    
    // Encode text to token IDs
    std::vector<int32_t> encode(const std::string& text) const;
    
    // Decode token IDs to text
    std::string decode(const std::vector<int32_t>& token_ids) const;
    
    // Get vocabulary size
    size_t getVocabSize() const { return vocab_size_; }
    
    // Get special token IDs
    int32_t getEosTokenId() const { return eos_token_id_; }
    int32_t getBosTokenId() const { return bos_token_id_; }
    int32_t getPadTokenId() const { return pad_token_id_; }
    int32_t getUnkTokenId() const { return unk_token_id_; }
    
    // Check if token is special
    bool isSpecialToken(int32_t token_id) const;
    
    // Check if token is vision-related
    bool isVisionToken(int32_t token_id) const;
    
    // Get token string
    std::string getTokenString(int32_t token_id) const;
    
private:
    std::string model_dir_;
    size_t vocab_size_;
    
    // Vocabulary mappings
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<int32_t, std::string> id_to_token_;
    
    // BPE merges
    std::vector<std::pair<std::string, std::string>> bpe_merges_;
    
    // Special tokens
    std::unordered_map<std::string, int32_t> special_tokens_;
    int32_t eos_token_id_;
    int32_t bos_token_id_;
    int32_t pad_token_id_;
    int32_t unk_token_id_;
    
    // Vision token range
    int32_t vision_start_token_;
    int32_t vision_end_token_;
    
    // Pre-tokenization regex
    std::regex pretokenize_regex_;
    
    // Load vocabulary from vocab.json
    bool loadVocabulary(const std::string& vocab_file);
    
    // Load merges from merges.txt
    bool loadMerges(const std::string& merges_file);
    
    // Load tokenizer config
    bool loadTokenizerConfig(const std::string& config_file);
    
    // BPE encoding
    std::vector<std::string> bpeEncode(const std::string& word) const;
    
    // Pre-tokenization
    std::vector<std::string> preTokenize(const std::string& text) const;
    
    // Apply BPE merges
    std::string applyBPE(const std::string& word) const;
};

} // namespace duorou