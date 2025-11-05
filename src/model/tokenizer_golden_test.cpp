#include "../extensions/ollama/gguf_parser.h"
#include "../utils/string_utils.h"
#include "byte_pair_encoding.h"
#include "sentence_piece.h"
#include "text_processor.h"
#include "tokenizer_factory.h"
#include "vocabulary.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace duorou::model;
using duorou::extensions::ollama::GGUFParser;

static std::string getEnv(const char *key) {
  const char *v = std::getenv(key);
  return v ? std::string(v) : std::string();
}

static bool
readTSV(const std::string &path,
        std::vector<std::pair<std::string, std::vector<int32_t>>> &rows) {
  std::ifstream in(path);
  if (!in)
    return false;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    // Expect: text \t id1,id2,id3
    auto tabPos = line.find('\t');
    if (tabPos == std::string::npos)
      continue;
    std::string text = line.substr(0, tabPos);
    std::string idsStr = line.substr(tabPos + 1);
    std::vector<int32_t> ids;
    std::stringstream ss(idsStr);
    std::string item;
    while (std::getline(ss, item, ',')) {
      if (!item.empty())
        ids.push_back(static_cast<int32_t>(std::stoi(item)));
    }
    rows.emplace_back(text, ids);
  }
  return true;
}

int main() {
  // Required envs
  const std::string dir = getEnv("DUOROU_TOKENIZER_DIR");
  const std::string type = getEnv("DUOROU_TOKENIZER_TYPE");    // "bpe" or "spm"
  const std::string addSpecial = getEnv("DUOROU_ADD_SPECIAL"); // "0" or "1"
  const std::string encodePath = getEnv("DUOROU_GOLDEN_ENCODE");
  const std::string decodePath = getEnv("DUOROU_GOLDEN_DECODE");
  const std::string ggufFileEnv = getEnv("DUOROU_GGUF_FILE");

  if (dir.empty()) {
    std::cerr << "ENV DUOROU_TOKENIZER_DIR missing" << std::endl;
    return 2;
  }

  // Try GGUF in dir (optional). If found, create vocab/tokenizer from GGUF
  // metadata.
  std::shared_ptr<Vocabulary> vocab;
  std::unique_ptr<TextProcessor> tokenizer;

  // Attempt GGUF parse: find first .gguf file in dir or use explicit env path
  std::string ggufFile;
  if (!ggufFileEnv.empty()) {
    ggufFile = ggufFileEnv;
  } else {
    // Try common filenames
    std::vector<std::string> candidates = {"model.gguf", "tokenizer.gguf"};
    for (const auto &name : candidates) {
      std::string path = dir + "/" + name;
      std::ifstream f(path);
      if (f.good()) {
        ggufFile = path;
        break;
      }
    }
    // If still not found, scan for any .gguf file in directory
    if (ggufFile.empty()) {
      try {
        for (const auto &entry : std::filesystem::directory_iterator(dir)) {
          if (entry.is_regular_file()) {
            const auto &p = entry.path();
            if (p.extension() == ".gguf") {
              ggufFile = p.string();
              break;
            }
          }
        }
      } catch (...) {
        // ignore directory errors
      }
    }
  }

  if (!ggufFile.empty()) {
    GGUFParser parser(true);
    parser.setUseMmap(false);
    if (parser.parseFile(ggufFile)) {
      vocab = createVocabularyFromGGUF(parser);
      if (!vocab) {
        std::cerr << "Failed to create vocabulary from GGUF: " << ggufFile
                  << std::endl;
        return 3;
      }
      TokenizerFactoryOptions opts;
      opts.override_type = type;
      tokenizer = createTextProcessorFromGGUF(parser, vocab, opts);
    }
  }

  // Fallback: build empty vocab to allow decode of numeric tokens
  if (!tokenizer) {
    vocab = std::make_shared<Vocabulary>();
    // Minimal values: map id to placeholder token strings like "<token_ID>"
    const int maxId = 300000; // safe upper bound
    std::vector<std::string> values;
    std::vector<int32_t> types;
    values.reserve(maxId);
    types.reserve(maxId);
    for (int i = 0; i < maxId; ++i) {
      values.push_back(std::string("<token_") + std::to_string(i) + ">");
      types.push_back(TOKEN_TYPE_NORMAL);
    }
    vocab->initialize(values, types);
    TokenizerFactoryOptions opts;
    opts.override_type = type.empty() ? std::string("bpe") : type;
    tokenizer = createTextProcessorForArchitecture("qwen", vocab, opts);
  }

  if (!tokenizer) {
    std::cerr << "Failed to create tokenizer" << std::endl;
    return 4;
  }

  std::cout << "[INFO] Tokenizer ready. Vocab size="
            << tokenizer->getVocabSize() << std::endl;

  // Golden encode
  if (!encodePath.empty()) {
    std::vector<std::pair<std::string, std::vector<int32_t>>> rows;
    if (!readTSV(encodePath, rows)) {
      std::cerr << "Failed to read encode TSV: " << encodePath << std::endl;
    } else {
      size_t ok = 0, total = rows.size();
      bool addSp = (addSpecial == "1" || addSpecial == "true");
      for (const auto &r : rows) {
        auto got = tokenizer->encode(r.first, addSp);
        if (got == r.second)
          ++ok;
        else {
          std::cerr << "[ENCODE MISMATCH] text='" << r.first
                    << "'\n  expected=";
          for (size_t i = 0; i < r.second.size(); ++i)
            std::cerr << (i ? "," : "") << r.second[i];
          std::cerr << "\n  got=";
          for (size_t i = 0; i < got.size(); ++i)
            std::cerr << (i ? "," : "") << got[i];
          std::cerr << std::endl;
        }
      }
      std::cout << "[ENCODE] " << ok << "/" << total << " matched" << std::endl;
    }
  }

  // Golden decode
  if (!decodePath.empty()) {
    std::vector<std::pair<std::string, std::vector<int32_t>>> rows;
    if (!readTSV(decodePath, rows)) {
      std::cerr << "Failed to read decode TSV: " << decodePath << std::endl;
    } else {
      size_t ok = 0, total = rows.size();
      for (const auto &r : rows) {
        auto got = tokenizer->decode(r.second);
        if (got == r.first)
          ++ok;
        else {
          std::cerr << "[DECODE MISMATCH] ids=";
          for (size_t i = 0; i < r.second.size(); ++i)
            std::cerr << (i ? "," : "") << r.second[i];
          std::cerr << "\n  expected='" << r.first << "'\n  got='" << got << "'"
                    << std::endl;
        }
      }
      std::cout << "[DECODE] " << ok << "/" << total << " matched" << std::endl;
    }
  }

  // Simple manual test from env DUOROU_MANUAL_IDS (comma-separated)
  const std::string manualIds = getEnv("DUOROU_MANUAL_IDS");
  if (!manualIds.empty()) {
    std::vector<int32_t> ids;
    std::stringstream ss(manualIds);
    std::string item;
    while (std::getline(ss, item, ',')) {
      if (!item.empty())
        ids.push_back(static_cast<int32_t>(std::stoi(item)));
    }
    std::string text = tokenizer->decode(ids);
    std::cout << "[MANUAL DECODE] ids=" << manualIds << " => '\"" << text
              << "\"'" << std::endl;
  }

  // Add end-to-end roundtrip check using manual text
  const std::string manualText = getEnv("DUOROU_MANUAL_TEXT");
  if (!manualText.empty()) {
    bool addSp = (addSpecial == "1" || addSpecial == "true");
    auto ids = tokenizer->encode(manualText, addSp);
    std::cout << "[MANUAL ENCODE] text='" << manualText << "' => ";
    for (size_t i = 0; i < ids.size(); ++i) {
      std::cout << (i ? "," : "") << ids[i];
    }
    std::cout << std::endl;
    auto roundtrip = tokenizer->decode(ids);
    std::cout << "[ROUNDTRIP] decode(encode(text)) => '" << roundtrip << "'"
              << std::endl;
    if (roundtrip == manualText) {
      std::cout << "[ROUNDTRIP] OK" << std::endl;
    } else {
      std::cout << "[ROUNDTRIP] MISMATCH" << std::endl;
    }
  }

  return 0;
}