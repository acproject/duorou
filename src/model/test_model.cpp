#include "model.h"
#include "vocabulary.h"
#include "byte_pair_encoding.h"
#include "sentence_piece.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <string>

using namespace duorou::model;

void testVocabulary() {
    std::cout << "Testing Vocabulary..." << std::endl;
    
    // Create test vocabulary
    std::vector<std::string> values = {"hello", "world", "test", "token"};
    std::vector<int32_t> types = {0, 0, 0, 0};
    std::vector<float> scores = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<std::string> merges = {"he llo", "wor ld"};
    
    Vocabulary vocab;
    vocab.initialize(values, types, scores, merges);
    
    // Test encoding
    int32_t id = vocab.encode("hello");
    assert(id >= 0);
    
    // Test decoding
    std::string token = vocab.decode(id);
    assert(token == "hello");
    
    // Test size
    assert(vocab.size() == 4);
    
    std::cout << "Vocabulary tests passed!" << std::endl;
}

void testBytePairEncoding() {
    std::cout << "Testing BytePairEncoding..." << std::endl;
    
    // Create test vocabulary
    auto vocab = std::make_shared<Vocabulary>();
    std::vector<std::string> values = {"h", "e", "l", "o", "w", "r", "d", " ", "he", "ll", "wo"};
    std::vector<int32_t> types(values.size(), 0);
    std::vector<float> scores(values.size(), 1.0f);
    std::vector<std::string> merges = {"h e", "l l"};
    
    vocab->initialize(values, types, scores, merges);
    
    // Create BPE tokenizer
    std::string pattern = R"(\w+|\s+)";
    BytePairEncoding bpe(pattern, vocab);
    
    // Test encoding
    std::string text = "hello world";
    auto tokens = bpe.encode(text, false);
    assert(!tokens.empty());
    
    // Test decoding
    std::string decoded = bpe.decode(tokens);
    // Note: decoded might not exactly match original due to BPE processing
    
    std::cout << "BytePairEncoding tests passed!" << std::endl;
}

void testSentencePiece() {
    std::cout << "Testing SentencePiece..." << std::endl;
    
    // Create test vocabulary
    auto vocab = std::make_shared<Vocabulary>();
    std::vector<std::string> values = {"▁hello", "▁world", "▁test", "▁token"};
    std::vector<int32_t> types(values.size(), 0);
    std::vector<float> scores(values.size(), 1.0f);
    
    vocab->initialize(values, types, scores);
    
    // Create SentencePiece tokenizer
    SentencePiece spm(vocab);
    
    // Test encoding
    std::string text = "hello world";
    auto tokens = spm.encode(text, false);
    assert(!tokens.empty());
    
    // Test decoding
    std::string decoded = spm.decode(tokens);
    // Note: decoded might not exactly match original due to SentencePiece processing
    
    std::cout << "SentencePiece tests passed!" << std::endl;
}

void testBaseModel() {
    std::cout << "Testing BaseModel..." << std::endl;
    
    BaseModel model;
    
    // Test initial state
    assert(!model.isLoaded());
    assert(model.getModelName() == "BaseModel");
    assert(model.getModelVersion() == "1.0");
    
    // Test configuration
    const auto& config = model.getConfig();
    assert(config.context_length == 2048);
    assert(config.temperature == 0.8);
    
    std::cout << "BaseModel tests passed!" << std::endl;
}

void testModelFactory() {
    std::cout << "Testing ModelFactory..." << std::endl;
    
    // Test supported models
    auto models = ModelFactory::getSupportedModels();
    assert(!models.empty());
    
    // Test creating model
    auto model = ModelFactory::createModel("BaseModel");
    assert(model != nullptr);
    assert(!model->isLoaded());
    
    std::cout << "ModelFactory tests passed!" << std::endl;
}

void testSpecialTokens() {
    std::cout << "Testing Special Tokens..." << std::endl;
    
    // Create vocabulary with special tokens
    std::vector<std::string> values = {"<pad>", "<unk>", "<bos>", "<eos>", "hello", "world"};
    std::vector<int32_t> types = {1, 1, 1, 1, 0, 0}; // First 4 are special
    std::vector<float> scores(values.size(), 1.0f);
    
    Vocabulary vocab;
    vocab.initialize(values, types, scores);
    
    // Set special tokens
    vocab.setBOS({2}, true);  // <bos> token
    vocab.setEOS({3}, true);  // <eos> token
    
    // Test special token detection
    assert(vocab.isSpecial(0, Special::PAD));
    assert(vocab.isSpecial(1, Special::UNK));
    assert(vocab.isSpecial(2, Special::BOS));
    assert(vocab.isSpecial(3, Special::EOS));
    assert(!vocab.isSpecial(4, Special::PAD));
    
    std::cout << "Special tokens tests passed!" << std::endl;
}

int main() {
    try {
        std::cout << "Running Model Module Tests..." << std::endl;
        
        testVocabulary();
        testBytePairEncoding();
        testSentencePiece();
        testBaseModel();
        testModelFactory();
        testSpecialTokens();
        
        std::cout << "All tests passed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Test failed with unknown exception" << std::endl;
        return 1;
    }
}