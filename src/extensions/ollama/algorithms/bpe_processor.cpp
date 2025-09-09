#include "bpe_processor.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <queue>
#include <sstream>

namespace duorou {
namespace extensions {
namespace ollama {

BPEProcessor::BPEProcessor(const std::string &pre_tokenizer_regex,
                           std::shared_ptr<Vocabulary> vocab)
    : vocab_(vocab) {
  try {
    if (!pre_tokenizer_regex.empty()) {
      pre_tokenizer_ = std::regex(pre_tokenizer_regex);
    } else {
      // Qwen2.5VL specific regex pattern from ollama source
      pre_tokenizer_ = std::regex(
          R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)");
    }
  } catch (const std::exception &e) {
    std::cerr
        << "Warning: Invalid regex pattern, using simple whitespace split: "
        << e.what() << std::endl;
    // Fallback to simple pattern
    pre_tokenizer_ = std::regex(R"(\S+|\s+)");
  }
}

std::vector<int32_t> BPEProcessor::encode(const std::string &text,
                                          bool add_special) {
  // Process special tokens first
  std::vector<Fragment> fragments = processSpecialTokens(text);

  std::vector<int32_t> ids;
  for (const auto &frag : fragments) {
    if (!frag.ids.empty()) {
      ids.insert(ids.end(), frag.ids.begin(), frag.ids.end());
      continue;
    }

    // Split text using pre-tokenizer
    std::vector<std::string> splits = splitText(frag.value);

    for (const std::string &split : splits) {
      if (split.empty())
        continue;

      // Preprocess bytes
      std::string processed = preprocessBytes(split);

      // Try direct vocabulary lookup first
      int32_t direct_id = vocab_->encode(processed);
      if (direct_id >= 0) {
        ids.push_back(direct_id);
        continue;
      }

      // Tokenize using BPE algorithm
      std::vector<int32_t> fragment_ids = tokenizeFragment(processed);
      ids.insert(ids.end(), fragment_ids.begin(), fragment_ids.end());
    }
  }

  if (add_special && !ids.empty()) {
    ids = vocab_->addSpecials(ids);
  }

  return ids;
}

std::string BPEProcessor::decode(const std::vector<int32_t> &tokens) {
  std::stringstream result;
  bool first_token = true;

  for (int32_t token_id : tokens) {
    std::string token_str = vocab_->decode(token_id);

    // Skip special tokens during decoding
    if (vocab_->is(token_id, SPECIAL_BOS) ||
        vocab_->is(token_id, SPECIAL_EOS)) {
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

    // Post-process bytes
    token_str = postprocessBytes(token_str);

    if (!token_str.empty()) {
      result << token_str;
    }

    first_token = false;
  }

  return result.str();
}

bool BPEProcessor::is(int32_t token_id, Special special) const {
  return vocab_->is(token_id, special);
}

const Vocabulary *BPEProcessor::getVocabulary() const { return vocab_.get(); }

size_t BPEProcessor::getVocabSize() const { return vocab_->size(); }

std::vector<std::string>
BPEProcessor::splitText(const std::string &text) const {
  std::vector<std::string> results;

  try {
    std::sregex_iterator iter(text.begin(), text.end(), pre_tokenizer_);
    std::sregex_iterator end;

    for (; iter != end; ++iter) {
      results.push_back(iter->str());
    }
  } catch (const std::exception &e) {
    std::cerr << "Regex error, falling back to character split: " << e.what()
              << std::endl;
    // Fallback: split by characters
    for (char c : text) {
      results.push_back(std::string(1, c));
    }
  }

  return results;
}

std::vector<Fragment>
BPEProcessor::processSpecialTokens(const std::string &text) const {
  std::vector<Fragment> fragments = {Fragment(text)};

  // Process each special token
  for (const std::string &special : vocab_->getSpecialVocabulary()) {
    int32_t special_id = vocab_->encode(special);
    if (special_id < 0)
      continue;

    for (size_t i = 0; i < fragments.size(); ++i) {
      Fragment &frag = fragments[i];
      if (!frag.ids.empty())
        continue; // Skip already processed fragments

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

std::vector<int32_t>
BPEProcessor::tokenizeFragment(const std::string &text) const {
  std::vector<char32_t> runes = stringToRunes(text);
  if (runes.empty())
    return {};

  std::vector<BPEMerge> merges(runes.size());
  for (size_t i = 0; i < runes.size(); ++i) {
    merges[i].p = static_cast<int>(i) - 1;
    merges[i].n = static_cast<int>(i) + 1;
    merges[i].runes = {runes[i]};
  }

  // Priority queue for merge pairs
  std::priority_queue<std::shared_ptr<BPEPair>,
                      std::vector<std::shared_ptr<BPEPair>>, BPEPairComparator>
      queue;

  // Helper function to create merge pairs
  auto createPair = [&](int a, int b) -> std::shared_ptr<BPEPair> {
    if (a < 0 || b >= static_cast<int>(runes.size())) {
      return nullptr;
    }

    std::string left = runesToString(merges[a].runes);
    std::string right = runesToString(merges[b].runes);

    int rank = vocab_->merge(left, right);
    if (rank < 0) {
      return nullptr;
    }

    return std::make_shared<BPEPair>(a, b, rank, left + right);
  };

  // Initialize queue with adjacent pairs
  for (size_t i = 0; i < runes.size() - 1; ++i) {
    auto pair = createPair(static_cast<int>(i), static_cast<int>(i) + 1);
    if (pair) {
      queue.push(pair);
    }
  }

  // Process merges
  while (!queue.empty()) {
    auto pair = queue.top();
    queue.pop();

    BPEMerge &left = merges[pair->a];
    BPEMerge &right = merges[pair->b];

    // Check if merge is still valid
    if (left.runes.empty() || right.runes.empty() ||
        runesToString(left.runes) + runesToString(right.runes) != pair->value) {
      continue;
    }

    // Check if the merged token exists in vocabulary
    int32_t token_id = vocab_->encode(pair->value);
    if (token_id < 0) {
      continue;
    }

    // Perform merge
    left.runes.insert(left.runes.end(), right.runes.begin(), right.runes.end());
    right.runes.clear();
    left.n = right.n;

    if (right.n < static_cast<int>(merges.size())) {
      merges[right.n].p = pair->a;
    }

    // Add new pairs
    if (auto new_pair = createPair(left.p, pair->a)) {
      queue.push(new_pair);
    }
    if (auto new_pair = createPair(pair->a, left.n)) {
      queue.push(new_pair);
    }
  }

  // Collect final tokens
  std::vector<int32_t> result;
  for (const auto &merge : merges) {
    if (!merge.runes.empty()) {
      std::string token_str = runesToString(merge.runes);
      int32_t token_id = vocab_->encode(token_str);

      if (token_id >= 0) {
        result.push_back(token_id);
      }
    }
  }

  return result;
}

std::string BPEProcessor::preprocessBytes(const std::string &text) const {
  std::string result;

  // Convert each byte of the UTF-8 input to its mapped character
  for (unsigned char byte : text) {
    char32_t mapped = mapByte(byte);

    // Convert mapped character to UTF-8
    if (mapped < 0x80) {
      result += static_cast<char>(mapped);
    } else if (mapped < 0x800) {
      result += static_cast<char>(0xC0 | (mapped >> 6));
      result += static_cast<char>(0x80 | (mapped & 0x3F));
    } else if (mapped < 0x10000) {
      result += static_cast<char>(0xE0 | (mapped >> 12));
      result += static_cast<char>(0x80 | ((mapped >> 6) & 0x3F));
      result += static_cast<char>(0x80 | (mapped & 0x3F));
    } else {
      // Handle 4-byte UTF-8 sequences for private use area
      result += static_cast<char>(0xF0 | (mapped >> 18));
      result += static_cast<char>(0x80 | ((mapped >> 12) & 0x3F));
      result += static_cast<char>(0x80 | ((mapped >> 6) & 0x3F));
      result += static_cast<char>(0x80 | (mapped & 0x3F));
    }
  }

  return result;
}

std::string BPEProcessor::postprocessBytes(const std::string &text) const {
  std::string result;
  std::vector<char32_t> runes = stringToRunes(text);

  for (char32_t rune : runes) {
    unsigned char byte = unmapByte(rune);
    if (byte != 0 || rune == 256) { // 256 maps to byte 0
      // This is a mapped byte character
      result += static_cast<char>(byte);
    } else {
      // Not a mapped byte, this shouldn't happen in proper BPE
      // but handle it gracefully by converting back to UTF-8
      if (rune < 0x80) {
        result += static_cast<char>(rune);
      } else if (rune < 0x800) {
        result += static_cast<char>(0xC0 | (rune >> 6));
        result += static_cast<char>(0x80 | (rune & 0x3F));
      } else if (rune < 0x10000) {
        result += static_cast<char>(0xE0 | (rune >> 12));
        result += static_cast<char>(0x80 | ((rune >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (rune & 0x3F));
      } else {
        result += static_cast<char>(0xF0 | (rune >> 18));
        result += static_cast<char>(0x80 | ((rune >> 12) & 0x3F));
        result += static_cast<char>(0x80 | ((rune >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (rune & 0x3F));
      }
    }
  }

  return result;
}

char32_t BPEProcessor::mapByte(unsigned char byte) const {
  // GPT-2 byte to unicode mapping
  // Based on OpenAI's encoder.py bytes_to_unicode function
  static std::unordered_map<uint8_t, char32_t> byte_to_unicode;
  if (byte_to_unicode.empty()) {
    // Initialize the mapping exactly as in GPT-2
    std::vector<uint8_t> bs;

    // Add printable ASCII characters (33-126): ! to ~
    for (int i = 33; i <= 126; i++) {
      bs.push_back(i);
    }

    // Add Latin-1 supplement characters (161-172): ¡ to ¬
    for (int i = 161; i <= 172; i++) {
      bs.push_back(i);
    }

    // Add Latin-1 supplement characters (174-255): ® to ÿ
    for (int i = 174; i <= 255; i++) {
      bs.push_back(i);
    }

    std::vector<char32_t> cs;
    // Copy bs to cs first
    for (uint8_t b : bs) {
      cs.push_back(static_cast<char32_t>(b));
    }

    int n = 0;
    // Map remaining bytes to private use area starting at 256
    for (int b = 0; b < 256; b++) {
      if (std::find(bs.begin(), bs.end(), static_cast<uint8_t>(b)) ==
          bs.end()) {
        bs.push_back(static_cast<uint8_t>(b));
        cs.push_back(256 + n);
        n++;
      }
    }

    // Create the mapping
    for (size_t i = 0; i < bs.size(); i++) {
      byte_to_unicode[bs[i]] = cs[i];
    }
  }

  auto it = byte_to_unicode.find(byte);
  if (it != byte_to_unicode.end()) {
    return it->second;
  }

  // Fallback - should not happen with correct implementation
  return 0xE000 + byte;
}

unsigned char BPEProcessor::unmapByte(char32_t rune) const {
  // Reverse GPT-2 unicode-to-byte mapping
  // Create reverse mapping on first use
  static std::unordered_map<char32_t, uint8_t> unicode_to_byte;
  if (unicode_to_byte.empty()) {
    // Initialize the reverse mapping exactly as in GPT-2
    std::vector<uint8_t> bs;

    // Add printable ASCII characters (33-126): ! to ~
    for (int i = 33; i <= 126; i++) {
      bs.push_back(i);
    }

    // Add Latin-1 supplement characters (161-172): ¡ to ¬
    for (int i = 161; i <= 172; i++) {
      bs.push_back(i);
    }

    // Add Latin-1 supplement characters (174-255): ® to ÿ
    for (int i = 174; i <= 255; i++) {
      bs.push_back(i);
    }

    std::vector<char32_t> cs;
    // Copy bs to cs first
    for (uint8_t b : bs) {
      cs.push_back(static_cast<char32_t>(b));
    }

    int n = 0;
    // Map remaining bytes to private use area starting at 256
    for (int b = 0; b < 256; b++) {
      if (std::find(bs.begin(), bs.end(), static_cast<uint8_t>(b)) ==
          bs.end()) {
        bs.push_back(static_cast<uint8_t>(b));
        cs.push_back(256 + n);
        n++;
      }
    }

    // Create the reverse mapping
    for (size_t i = 0; i < bs.size(); i++) {
      unicode_to_byte[cs[i]] = bs[i];
    }
  }

  auto it = unicode_to_byte.find(rune);
  if (it != unicode_to_byte.end()) {
    return it->second;
  }

  // Fallback for invalid mapping
  return 0;
}

std::vector<char32_t>
BPEProcessor::stringToRunes(const std::string &text) const {
  std::vector<char32_t> runes;

  // Simple UTF-8 to UTF-32 conversion
  for (size_t i = 0; i < text.length();) {
    unsigned char c = text[i];
    char32_t codepoint;

    if (c < 0x80) {
      codepoint = c;
      i += 1;
    } else if ((c & 0xE0) == 0xC0) {
      if (i + 1 >= text.length())
        break;
      codepoint = ((c & 0x1F) << 6) | (text[i + 1] & 0x3F);
      i += 2;
    } else if ((c & 0xF0) == 0xE0) {
      if (i + 2 >= text.length())
        break;
      codepoint = ((c & 0x0F) << 12) | ((text[i + 1] & 0x3F) << 6) |
                  (text[i + 2] & 0x3F);
      i += 3;
    } else if ((c & 0xF8) == 0xF0) {
      if (i + 3 >= text.length())
        break;
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

std::string
BPEProcessor::runesToString(const std::vector<char32_t> &runes) const {
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