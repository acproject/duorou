#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>

#include "vocabulary.h"
#include "byte_pair_encoding.h"
#include "sentence_piece.h"

using duorou::model::Vocabulary;
using duorou::model::BytePairEncoding;
using duorou::model::SentencePiece;
using duorou::model::TextProcessor;

namespace fs = std::filesystem;

static std::string getEnv(const char* key) {
    const char* v = std::getenv(key);
    return v ? std::string(v) : std::string();
}

static bool readLines(const fs::path& file, std::vector<std::string>& out) {
    std::ifstream in(file);
    if (!in.is_open()) return false;
    std::string line;
    while (std::getline(in, line)) {
        // trim trailing CR if present (Windows line ending)
        if (!line.empty() && line.back() == '\r') line.pop_back();
        out.push_back(line);
    }
    return true;
}

static std::vector<int32_t> parseIdsCSV(const std::string& csv) {
    std::vector<int32_t> ids;
    std::stringstream ss(csv);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        try {
            ids.push_back(static_cast<int32_t>(std::stol(tok)));
        } catch (...) {
            // ignore parse errors
        }
    }
    return ids;
}

static int32_t parseTokenType(const std::string& tstr) {
    // Accept integer or names: normal|control|user(_defined)|unknown|unk|unused|byte
    try {
        size_t idx = 0;
        int val = std::stoi(tstr, &idx);
        if (idx == tstr.size()) return static_cast<int32_t>(val);
    } catch (...) {}
    std::string l = tstr;
    std::transform(l.begin(), l.end(), l.begin(), ::tolower);
    if (l == "normal") return duorou::model::TOKEN_TYPE_NORMAL;
    if (l == "control") return duorou::model::TOKEN_TYPE_CONTROL;
    if (l == "user" || l == "user_defined" || l == "user-defined") return duorou::model::TOKEN_TYPE_USER_DEFINED;
    if (l == "unknown" || l == "unk") return duorou::model::TOKEN_TYPE_UNKNOWN;
    if (l == "unused") return duorou::model::TOKEN_TYPE_UNUSED;
    if (l == "byte" || l == "byte_fallback" || l == "byte-fallback") return duorou::model::TOKEN_TYPE_BYTE;
    return duorou::model::TOKEN_TYPE_NORMAL;
}

static bool loadVocabularyFromDir(const fs::path& dir, const std::string& type, Vocabulary& vocab) {
    // Expect vocab.txt and merges.txt for bpe; for spm, expect vocab.txt with values/types/scores
    fs::path vocabTxt = dir / "vocab.txt";
    fs::path mergesTxt = dir / "merges.txt";

    std::vector<std::string> values;
    std::vector<int32_t> types;
    std::vector<float> scores;
    std::vector<std::string> merges;

    // Load vocab.txt: expect either "token type score" or just "token" per line
    if (fs::exists(vocabTxt)) {
        std::ifstream in(vocabTxt);
        if (!in.is_open()) {
            std::cerr << "Failed to open " << vocabTxt << std::endl;
            return false;
        }
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            std::stringstream ls(line);
            std::string token;
            int t = duorou::model::TOKEN_TYPE_NORMAL;
            float s = 0.0f;
            ls >> std::noskipws;
            // Token may contain spaces; assume it's quoted or tab-separated? Fallback: take to first tab
            // Here we support formats:
            // 1) token\ttype\tscore
            // 2) token
            // 3) token\ttype
            // We'll split by tab if present, else by space treating first as token when quoted with <> or []
            if (line.find('\t') != std::string::npos) {
                std::vector<std::string> parts;
                std::string part;
                std::stringstream ts(line);
                while (std::getline(ts, part, '\t')) parts.push_back(part);
                if (!parts.empty()) token = parts[0];
                if (parts.size() > 1) {
                    t = parseTokenType(parts[1]);
                }
                if (parts.size() > 2) {
                    try { s = std::stof(parts[2]); } catch (...) {}
                }
            } else {
                // space-separated fallback: token type score
                std::stringstream ws(line);
                if (!(ws >> token)) continue;
                std::string tstr;
                if (ws >> tstr) t = parseTokenType(tstr);
                if (!(ws >> s)) s = 0.0f;
            }
            values.push_back(token);
            types.push_back(t);
            scores.push_back(s);
        }
    } else {
        std::cerr << "Missing vocab.txt in " << dir << std::endl;
        return false;
    }

    if (type == "bpe") {
        if (fs::exists(mergesTxt)) {
            std::ifstream min(mergesTxt);
            std::string line;
            while (std::getline(min, line)) {
                if (!line.empty() && line[0] == '#') continue; // skip comments
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.empty()) continue;
                merges.push_back(line);
            }
        }
        vocab.initialize(values, types, scores, merges);
    } else {
        vocab.initialize(values, types, scores, {});
    }

    return true;
}

int main() {
    std::string dir = getEnv("DUOROU_TOKENIZER_DIR");
    std::string type = getEnv("DUOROU_TOKENIZER_TYPE");
    std::string encodeFile = getEnv("DUOROU_GOLDEN_ENCODE"); // TSV: text\tidsCSV
    std::string decodeFile = getEnv("DUOROU_GOLDEN_DECODE"); // TSV: idsCSV\ttext
    std::string addSpecialEnv = getEnv("DUOROU_ADD_SPECIAL");

    if (dir.empty() || type.empty()) {
        std::cout << "Tokenizer golden tests skipped" << std::endl;
        return 0;
    }

    auto vocabPtr = std::make_shared<Vocabulary>();
    if (!loadVocabularyFromDir(dir, type, *vocabPtr)) {
        std::cerr << "Failed to load vocabulary from: " << dir << std::endl;
        std::cout << "Tokenizer golden tests skipped" << std::endl;
        return 0;
    }

    std::unique_ptr<TextProcessor> tokenizer;
    if (type == "bpe") {
        std::string pattern = R"(\S+|\s+)";
        tokenizer = std::make_unique<BytePairEncoding>(pattern, vocabPtr);
    } else {
        tokenizer = std::make_unique<SentencePiece>(vocabPtr);
    }

    bool addSpecial = (addSpecialEnv == "1" || addSpecialEnv == "true" || addSpecialEnv == "TRUE");

    // Encode checks
    if (!encodeFile.empty()) {
        fs::path ef(encodeFile);
        if (!fs::exists(ef)) {
            std::cerr << "Encode golden file not found: " << ef << std::endl;
            std::cout << "Tokenizer golden tests skipped" << std::endl;
            return 0;
        }
        std::vector<std::string> lines;
        if (!readLines(ef, lines)) {
            std::cerr << "Failed to read encode golden file" << std::endl;
            return 1;
        }
        for (size_t i = 0; i < lines.size(); ++i) {
            const std::string& line = lines[i];
            if (line.empty()) continue;
            size_t tab = line.find('\t');
            if (tab == std::string::npos) {
                std::cerr << "Invalid encode golden format at line " << (i+1) << std::endl;
                return 1;
            }
            std::string text = line.substr(0, tab);
            std::string idsCSV = line.substr(tab + 1);
            std::vector<int32_t> expected = parseIdsCSV(idsCSV);
            std::vector<int32_t> got = tokenizer->encode(text, addSpecial);
            if (got != expected) {
                std::cerr << "Encode mismatch at line " << (i+1) << "\n";
                std::cerr << "Text: " << text << "\n";
                std::cerr << "Expected: ";
                for (size_t j = 0; j < expected.size(); ++j) {
                    if (j) std::cerr << ',';
                    std::cerr << expected[j];
                }
                std::cerr << "\nGot: ";
                for (size_t j = 0; j < got.size(); ++j) {
                    if (j) std::cerr << ',';
                    std::cerr << got[j];
                }
                std::cerr << "\n";
                return 1;
            }
        }
    }

    // Decode checks
    if (!decodeFile.empty()) {
        fs::path df(decodeFile);
        if (!fs::exists(df)) {
            std::cerr << "Decode golden file not found: " << df << std::endl;
            std::cout << "Tokenizer golden tests skipped" << std::endl;
            return 0;
        }
        std::vector<std::string> lines;
        if (!readLines(df, lines)) {
            std::cerr << "Failed to read decode golden file" << std::endl;
            return 1;
        }
        for (size_t i = 0; i < lines.size(); ++i) {
            const std::string& line = lines[i];
            if (line.empty()) continue;
            size_t tab = line.find('\t');
            if (tab == std::string::npos) {
                std::cerr << "Invalid decode golden format at line " << (i+1) << std::endl;
                return 1;
            }
            std::string idsCSV = line.substr(0, tab);
            std::string text = line.substr(tab + 1);
            std::vector<int32_t> ids = parseIdsCSV(idsCSV);
            std::string got = tokenizer->decode(ids);
            if (got != text) {
                std::cerr << "Decode mismatch at line " << (i+1) << "\n";
                std::cerr << "Ids: " << idsCSV << "\n";
                std::cerr << "Expected: " << text << "\n";
                std::cerr << "Got: " << got << "\n";
                return 1;
            }
        }
    }

    std::cout << "Tokenizer golden tests passed" << std::endl;
    return 0;
}