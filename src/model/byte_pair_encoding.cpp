#include "byte_pair_encoding.h"
#include <algorithm>
#include <codecvt>
#include <iomanip>
#include <iostream>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace duorou {
namespace model {

// Simple utility to replace all occurrences of a substring
static void replaceAll(std::string &s, const std::string &from,
                       const std::string &to) {
  if (from.empty())
    return;
  size_t pos = 0;
  while ((pos = s.find(from, pos)) != std::string::npos) {
    s.replace(pos, from.length(), to);
    pos += to.length();
  }
}

// Sanitize a Unicode-property regex into an ECMAScript-compatible approximation
// Notes:
// - std::regex (ECMAScript) does not support \p{..} classes.
// - We approximate:
//   \p{L} -> [A-Za-z\x80-\xFF] (treat non-ASCII bytes as letters to keep them
//   grouped)
//   \p{N} -> \d
//   [^\s\p{L}\p{N}] -> [^\sA-Za-z\d\x80-\xFF]
//   [^\r\n\p{L}\p{N}] -> [^\r\nA-Za-z\d\x80-\xFF]
// This is a pragmatic compromise to avoid over-fragmentation for UTF-8 text.
static std::string sanitizePatternForECMA(std::string pattern) {
  // First handle common negated classes to avoid double-replacing inner tokens
  // later
  replaceAll(pattern, "[^\\s\\p{L}\\p{N}]", "[^\\sA-Za-z\\d\\x80-\\xFF]");
  replaceAll(pattern, "[^\\r\\n\\p{L}\\p{N}]", "[^\\r\\nA-Za-z\\d\\x80-\\xFF]");

  // Replace Unicode property classes with ECMAScript-compatible approximations
  replaceAll(pattern, "\\p{L}", "[A-Za-z\\x80-\\xFF]");
  replaceAll(pattern, "\\p{N}", "\\d");
  // Some patterns might contain double-escaped forms from C++ string literals
  // or env strings
  replaceAll(pattern, "\\\\p{L}", "[A-Za-z\\x80-\\xFF]");
  replaceAll(pattern, "\\\\p{N}", "\\d");

  // Remove unsupported non-capturing groups (?:...) by turning them into normal
  // groups
  replaceAll(pattern, "(?:", "(");

  // Approximate unsupported negative lookahead used by GPT-2 pattern: \s+(?!\S)
  // -> \s+ This loses the end-of-string specificity but remains a safe
  // over-approximation for token splitting
  replaceAll(pattern, "\\s+(?!\\S)", "\\s+");

  return pattern;
}

BytePairEncoding::BytePairEncoding(const std::string &pattern,
                                   std::shared_ptr<Vocabulary> vocab)
    : vocab_(vocab) {
  try {
    // Try to compile given pattern with ECMAScript first
    std::string sanitized = sanitizePatternForECMA(pattern);
    preTokenizeRegex_ =
        std::regex(sanitized, std::regex::ECMAScript | std::regex::optimize);
  } catch (const std::regex_error &e) {
    // Fallback to a safe, ECMAScript-compatible pattern that roughly mimics
    // intended behavior
    std::cerr << "[WARN] Invalid BPE regex pattern for std::regex "
                 "(ECMAScript). Fallback to safe pattern. Reason: "
              << e.what() << std::endl;
    // This groups: ASCII letters, digits, runs of non-ASCII bytes (UTF-8),
    // punctuation runs, and whitespace runs
    const char *kSafeFallbackPattern =
        R"([A-Za-z]+|\d+|[\x80-\xFF]+|[^\sA-Za-z\d\x80-\xFF]+|\s+)";
    try {
      preTokenizeRegex_ = std::regex(
          kSafeFallbackPattern, std::regex::ECMAScript | std::regex::optimize);
    } catch (...) {
      // Ultimate guard: extremely simple splitter that always compiles
      preTokenizeRegex_ = std::regex(R"(\S+|\s+)", std::regex::ECMAScript);
    }
  }
}

std::vector<int32_t> BytePairEncoding::encode(const std::string &text,
                                              bool addSpecial) {
  // Start with a single fragment containing the entire text
  std::vector<Fragment> fragments = {Fragment(text)};

  // Process special tokens
  fragments = processSpecialTokens(fragments);

  std::vector<int32_t> result;

  for (const auto &fragment : fragments) {
    if (!fragment.ids.empty()) {
      // Fragment already has token IDs (special token)
      result.insert(result.end(), fragment.ids.begin(), fragment.ids.end());
      continue;
    }

    // Split the fragment using pre-tokenization regex
    auto splits = split(fragment.value);

    for (const auto &split : splits) {
      auto tokens = applyBPE(split);
      result.insert(result.end(), tokens.begin(), tokens.end());
    }
  }

  if (addSpecial && !result.empty()) {
    result = vocab_->addSpecials(result);
  }

  return result;
}

std::string BytePairEncoding::decode(const std::vector<int32_t> &ids) {
  // Reconstruct original byte stream by reversing GPT-2 byte->unicode mapping
  // and handling byte placeholder tokens. Then return as UTF-8.
  std::string out_bytes;
  out_bytes.reserve(ids.size() * 4);

  std::cout << "[DEBUG] BytePairEncoding::decode called with " << ids.size()
            << " token IDs" << std::endl;
  std::cout << "[DEBUG] Vocabulary size: " << vocab_->size() << std::endl;
  if (!ids.empty()) {
    std::cout << "[DEBUG] First 10 token IDs: ";
    for (size_t i = 0; i < std::min(static_cast<size_t>(10), ids.size()); ++i) {
      std::cout << ids[i] << " ";
    }
    std::cout << std::endl;
    if (ids.size() > 10) {
      std::cout << "[DEBUG] Last 10 token IDs: ";
      for (size_t i = std::max(static_cast<size_t>(0), ids.size() - 10);
           i < ids.size(); ++i) {
        std::cout << ids[i] << " ";
      }
      std::cout << std::endl;
    }
  }

  auto append_utf8 = [&](uint32_t cp) {
    if (cp <= 0x7F) {
      out_bytes.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
      out_bytes.push_back(static_cast<char>(0xC0 | (cp >> 6)));
      out_bytes.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
      out_bytes.push_back(static_cast<char>(0xE0 | (cp >> 12)));
      out_bytes.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out_bytes.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0x10FFFF) {
      out_bytes.push_back(static_cast<char>(0xF0 | (cp >> 18)));
      out_bytes.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
      out_bytes.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out_bytes.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
  };

  for (size_t i = 0; i < ids.size(); ++i) {
    int32_t id = ids[i];
    std::string token = vocab_->decode(id);
    if (i < 10 || token.find("<token_") != std::string::npos) {
      std::cout << "[DEBUG] Token[" << i << "] ID=" << id << " -> '" << token
                << "' (length: " << token.length();
      if (token.find("<token_") != std::string::npos) {
        std::cout << ") [PLACEHOLDER TOKEN DETECTED]";
      }
      std::cout << ")" << std::endl;
    }

    // Fast-path: if token is a single byte (common for byte-fallback tokens),
    // append directly
    if (token.size() == 1) {
      out_bytes.push_back(token[0]);
      continue;
    }

    // General path: iterate codepoints in token (UTF-8), reverse GPT-2 mapping
    // where applicable
    for (size_t j = 0; j < token.size();) {
      unsigned char b0 = static_cast<unsigned char>(token[j]);
      uint32_t cp = 0;
      size_t adv = 1;

      if (b0 < 0x80) {
        cp = b0;
        adv = 1;
      } else if ((b0 & 0xE0) == 0xC0 && j + 1 < token.size()) {
        unsigned char b1 = static_cast<unsigned char>(token[j + 1]);
        if ((b1 & 0xC0) == 0x80) {
          cp = ((b0 & 0x1F) << 6) | (b1 & 0x3F);
          adv = 2;
        } else {
          cp = b0; // invalid continuation, treat as single byte
          adv = 1;
        }
      } else if ((b0 & 0xF0) == 0xE0 && j + 2 < token.size()) {
        unsigned char b1 = static_cast<unsigned char>(token[j + 1]);
        unsigned char b2 = static_cast<unsigned char>(token[j + 2]);
        if ((b1 & 0xC0) == 0x80 && (b2 & 0xC0) == 0x80) {
          cp = ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F);
          adv = 3;
        } else {
          cp = b0; // invalid sequence, fallback to raw byte
          adv = 1;
        }
      } else if ((b0 & 0xF8) == 0xF0 && j + 3 < token.size()) {
        unsigned char b1 = static_cast<unsigned char>(token[j + 1]);
        unsigned char b2 = static_cast<unsigned char>(token[j + 2]);
        unsigned char b3 = static_cast<unsigned char>(token[j + 3]);
        if ((b1 & 0xC0) == 0x80 && (b2 & 0xC0) == 0x80 && (b3 & 0xC0) == 0x80) {
          cp = ((b0 & 0x07) << 18) | ((b1 & 0x3F) << 12) | ((b2 & 0x3F) << 6) |
               (b3 & 0x3F);
          adv = 4;
        } else {
          cp = b0;
          adv = 1;
        }
      } else {
        cp = b0;
        adv = 1;
      }

      // Reverse GPT-2 byte->unicode mapping where applicable
      // Mapped ranges per byteToUnicode():
      //  - 0x0100..0x0120 => bytes 0x00..0x20
      //  - 0x0121..0x0142 => bytes 0x7F..0xA0
      //  - 0x0143         => byte  0xAD
      //  - <= 0xFF        => identity mapping
      uint8_t mapped = unicodeToByte(cp);
      bool in_mapped_range = (cp == 0x0143) || (cp >= 0x0100 && cp <= 0x0120) ||
                             (cp >= 0x0121 && cp <= 0x0142) || (cp <= 0xFF);

      if (in_mapped_range) {
        out_bytes.push_back(static_cast<char>(mapped));
      } else {
        // Not a mapped codepoint: keep original character bytes (already UTF-8)
        append_utf8(cp);
      }

      j += adv;
    }
  }

  std::cout << "[DEBUG] Final decoded string length: " << out_bytes.length()
            << std::endl;
  std::cout << "[DEBUG] Final decoded string (first 100 chars): '"
            << out_bytes.substr(0, 100) << "'" << std::endl;

  return out_bytes;
}

bool BytePairEncoding::isSpecial(int32_t id, Special special) const {
  return vocab_->isSpecial(id, special);
}

const Vocabulary *BytePairEncoding::getVocabulary() const {
  return vocab_.get();
}

size_t BytePairEncoding::getVocabSize() const { return vocab_->size(); }

std::vector<std::string>
BytePairEncoding::split(const std::string &text) const {
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

std::vector<Fragment> BytePairEncoding::processSpecialTokens(
    const std::vector<Fragment> &fragments) const {
  std::vector<Fragment> result = fragments;

  auto specialTokens = vocab_->getSpecialVocabulary();

  for (const auto &special : specialTokens) {
    int32_t id = vocab_->encode(special);
    if (id < 0)
      continue;

    for (size_t i = 0; i < result.size(); ++i) {
      Fragment &frag = result[i];
      if (!frag.ids.empty())
        continue; // Already processed

      size_t pos = frag.value.find(special);
      if (pos == std::string::npos)
        continue;

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
      result.insert(result.begin() + i, newFragments.begin(),
                    newFragments.end());
      i += newFragments.size() - 1;
    }
  }

  return result;
}

std::vector<int32_t> BytePairEncoding::applyBPE(const std::string &text) const {
  // Map raw bytes to GPT-2 unicode domain per byteToUnicode(), then perform BPE
  // on mapped text
  std::string processedText = text;

  // Build mapped runes (each original byte -> one mapped Unicode codepoint
  // encoded as UTF-8)
  std::vector<std::string> runes;
  runes.reserve(processedText.size());

  auto encode_utf8 = [](uint32_t cp) {
    std::string s;
    if (cp <= 0x7F) {
      s.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
      s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
      s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
      s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
      s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
      s.push_back(static_cast<char>(0xF0 | (cp >> 18)));
      s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
      s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return s;
  };

  std::string mappedText;
  mappedText.reserve(processedText.size() * 2);

  for (size_t i = 0; i < processedText.size(); ++i) {
    uint8_t b = static_cast<uint8_t>(processedText[i]);
    uint32_t cp = byteToUnicode(b);
    std::string u = encode_utf8(cp);
    mappedText += u;
    runes.emplace_back(std::move(u));
  }

  // If the entire mapped text is a token, short-circuit
  int32_t id = vocab_->encode(mappedText);
  if (id >= 0) {
    return {id};
  }

  // Initialize merges
  struct MergeInfo {
    int prev = -1;
    int next = -1;
    std::vector<uint32_t> runes;
  };

  std::vector<MergeInfo> merges;
  merges.reserve(runes.size());

  for (size_t i = 0; i < runes.size(); ++i) {
    MergeInfo m;
    m.prev = (i == 0) ? -1 : static_cast<int>(i - 1);
    m.next = (i + 1 < runes.size()) ? static_cast<int>(i + 1) : -1;
    m.runes.push_back(static_cast<uint32_t>(i));
    merges.push_back(std::move(m));
  }

  // Priority queue of pairs by rank
  struct PairInfo {
    int a;
    int b;
    int rank;
    std::string value;
    bool operator>(const PairInfo &other) const { return rank > other.rank; }
  };

  auto getPairRank = [&](int leftIdx, int rightIdx) -> int {
    if (leftIdx < 0 || rightIdx < 0)
      return -1;
    std::string leftStr, rightStr;
    for (const auto &idx : merges[leftIdx].runes)
      leftStr += runes[idx];
    for (const auto &idx : merges[rightIdx].runes)
      rightStr += runes[idx];
    int rank = vocab_->getMergeRank(leftStr, rightStr);
    return rank;
  };

  std::priority_queue<PairInfo, std::vector<PairInfo>, std::greater<PairInfo>>
      pairs;
  for (size_t i = 0; i < merges.size(); ++i) {
    int rank = getPairRank(static_cast<int>(i), merges[i].next);
    if (rank >= 0) {
      std::string leftStr, rightStr;
      for (const auto &idx : merges[i].runes)
        leftStr += runes[idx];
      for (const auto &idx : merges[merges[i].next].runes)
        rightStr += runes[idx];
      pairs.push(
          {static_cast<int>(i), merges[i].next, rank, leftStr + rightStr});
    }
  }

  while (!pairs.empty()) {
    auto pair = pairs.top();
    pairs.pop();

    if (pair.a < 0 || pair.b < 0)
      continue;
    auto &left = merges[pair.a];
    auto &right = merges[pair.b];
    if (right.runes.empty())
      continue; // already merged

    std::string leftStr, rightStr;
    for (const auto &idx : left.runes)
      leftStr += runes[idx];
    for (const auto &idx : right.runes)
      rightStr += runes[idx];

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
      for (const auto &idx : merges[left.prev].runes)
        prevStr += runes[idx];
      int rank = vocab_->getMergeRank(prevStr, pair.value);
      if (rank >= 0) {
        pairs.push(PairInfo{left.prev, pair.a, rank, prevStr + pair.value});
      }
    }

    if (left.next >= 0) {
      std::string nextStr;
      for (const auto &idx : merges[left.next].runes)
        nextStr += runes[idx];
      int rank = vocab_->getMergeRank(pair.value, nextStr);
      if (rank >= 0) {
        pairs.push(PairInfo{pair.a, left.next, rank, pair.value + nextStr});
      }
    }
  }

  // Collect final tokens
  std::vector<int32_t> result;
  for (const auto &merge : merges) {
    if (!merge.runes.empty()) {
      std::string token;
      for (const auto &idx : merge.runes) {
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