#include "vocabulary.h"
#include "byte_pair_encoding.h"
#include "sentence_piece.h"
#include "text_processor.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <algorithm>
#include <string>
#include <cstdlib>

using namespace duorou::model;

struct Args {
    std::string tokenizer_path;
    std::string type = "bpe"; // bpe | spm
    std::string mode = "encode"; // encode | decode
    std::string text;
    std::string ids;
    bool add_special = false;
    // New options
    std::string pattern; // BPE pre-tokenization regex
    std::string bos;     // BOS tokens or ids (comma-separated)
    std::string eos;     // EOS tokens or ids (comma-separated)
};

static void print_usage() {
    std::cout << "Usage: tokenizer_cli --tokenizer <path> --type <bpe|spm> --mode <encode|decode> [--text <str>] [--ids <comma-separated>] [--add-special] [--pattern <regex>] [--bos <csv tokens-or-ids>] [--eos <csv tokens-or-ids>]\n";
}

static bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--tokenizer" && i + 1 < argc) {
            args.tokenizer_path = argv[++i];
        } else if (a == "--type" && i + 1 < argc) {
            args.type = argv[++i];
        } else if (a == "--mode" && i + 1 < argc) {
            args.mode = argv[++i];
        } else if (a == "--text" && i + 1 < argc) {
            args.text = argv[++i];
        } else if (a == "--ids" && i + 1 < argc) {
            args.ids = argv[++i];
        } else if (a == "--add-special") {
            args.add_special = true;
        } else if (a == "--pattern" && i + 1 < argc) {
            args.pattern = argv[++i];
        } else if (a == "--bos" && i + 1 < argc) {
            args.bos = argv[++i];
        } else if (a == "--eos" && i + 1 < argc) {
            args.eos = argv[++i];
        } else if (a == "-h" || a == "--help") {
            print_usage();
            return false;
        } else {
            std::cerr << "Unknown argument: " << a << std::endl;
            return false;
        }
    }

    if (args.tokenizer_path.empty()) return false;
    if (args.mode != "encode" && args.mode != "decode") return false;
    if (args.type != "bpe" && args.type != "spm") return false;
    if (args.mode == "encode" && args.text.empty()) return false;
    if (args.mode == "decode" && args.ids.empty()) return false;
    return true;
}

// Robust vocab loader: supports
// - vocab.txt lines: token\ttype\tscore OR token\ttype OR token
// - merges.txt (optional), skipping comment lines starting with '#'
static std::shared_ptr<Vocabulary> load_vocab(const std::string& tokenizer_path) {
    auto vocab = std::make_shared<Vocabulary>();

    std::vector<std::string> values;
    std::vector<int32_t> types;
    std::vector<float> scores;
    std::vector<std::string> merges;

    auto trim_cr = [](std::string& s) {
        if (!s.empty() && s.back() == '\r') s.pop_back();
    };

    auto parse_type = [](const std::string& tstr) -> int32_t {
        // Accept integer or names: normal|control|user(_defined)|unknown|unk|unused|byte
        try {
            size_t idx = 0;
            int val = std::stoi(tstr, &idx);
            if (idx == tstr.size()) return static_cast<int32_t>(val);
        } catch (...) {}
        std::string l = tstr;
        std::transform(l.begin(), l.end(), l.begin(), ::tolower);
        if (l == "normal") return TOKEN_TYPE_NORMAL;
        if (l == "control") return TOKEN_TYPE_CONTROL;
        if (l == "user" || l == "user_defined" || l == "user-defined") return TOKEN_TYPE_USER_DEFINED;
        if (l == "unknown" || l == "unk") return TOKEN_TYPE_UNKNOWN;
        if (l == "unused") return TOKEN_TYPE_UNUSED;
        if (l == "byte" || l == "byte_fallback" || l == "byte-fallback") return TOKEN_TYPE_BYTE;
        return TOKEN_TYPE_NORMAL;
    };

    // Load vocab.txt
    std::ifstream vocabFile(tokenizer_path + "/vocab.txt");
    if (vocabFile.is_open()) {
        std::string line;
        while (std::getline(vocabFile, line)) {
            trim_cr(line);
            if (line.empty()) continue;

            std::string token;
            int32_t t = TOKEN_TYPE_NORMAL;
            float s = 0.0f;

            if (line.find('\t') != std::string::npos) {
                std::vector<std::string> parts;
                std::string part;
                std::stringstream ts(line);
                while (std::getline(ts, part, '\t')) parts.push_back(part);
                if (!parts.empty()) token = parts[0];
                if (parts.size() > 1) t = parse_type(parts[1]);
                if (parts.size() > 2) {
                    try { s = std::stof(parts[2]); } catch (...) { s = 0.0f; }
                }
            } else {
                // fallback: token type score (space separated)
                std::stringstream ws(line);
                if (!(ws >> token)) continue;
                std::string tstr;
                if (ws >> tstr) t = parse_type(tstr);
                if (!(ws >> s)) s = 0.0f;
            }
            values.push_back(token);
            types.push_back(t);
            scores.push_back(s);
        }
        vocabFile.close();
    } else {
        std::cerr << "Failed to open " << (tokenizer_path + "/vocab.txt") << std::endl;
    }

    // Load merges.txt (optional)
    std::ifstream mergesFile(tokenizer_path + "/merges.txt");
    if (mergesFile.is_open()) {
        std::string line;
        while (std::getline(mergesFile, line)) {
            trim_cr(line);
            if (line.empty()) continue;
            if (!line.empty() && line[0] == '#') continue;
            merges.push_back(line);
        }
        mergesFile.close();
    }

    vocab->initialize(values, types, scores, merges);
    return vocab;
}

static std::vector<int32_t> parse_ids(const std::string& s) {
    std::vector<int32_t> out;
    std::string num;
    std::istringstream iss(s);
    while (std::getline(iss, num, ',')) {
        if (!num.empty()) {
            out.push_back(static_cast<int32_t>(std::stoi(num)));
        }
    }
    return out;
}

static std::string getEnv(const char* key) {
    const char* v = std::getenv(key);
    return v ? std::string(v) : std::string();
}

static std::vector<int32_t> parse_ids_or_tokens_csv(const std::string& csv, const Vocabulary& vocab) {
    std::vector<int32_t> ids;
    std::stringstream ss(csv);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (part.empty()) continue;
        // trim spaces
        part.erase(part.begin(), std::find_if(part.begin(), part.end(), [](unsigned char ch){ return !std::isspace(ch); }));
        part.erase(std::find_if(part.rbegin(), part.rend(), [](unsigned char ch){ return !std::isspace(ch); }).base(), part.end());
        try {
            size_t idx = 0;
            long v = std::stol(part, &idx);
            if (idx == part.size()) {
                ids.push_back(static_cast<int32_t>(v));
                continue;
            }
        } catch (...) {}
        int32_t id = vocab.encode(part);
        if (id >= 0) {
            ids.push_back(id);
        } else {
            std::cerr << "Warning: token not found in vocab for BOS/EOS: " << part << std::endl;
        }
    }
    return ids;
}

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        print_usage();
        return 1;
    }

    auto vocab = load_vocab(args.tokenizer_path);
    if (!vocab || vocab->size() == 0) {
        std::cerr << "Vocabulary is empty or failed to load." << std::endl;
        return 2;
    }

    // Configure BOS/EOS via CLI or environment
    std::string bosStr = args.bos;
    std::string eosStr = args.eos;
    if (bosStr.empty()) bosStr = getEnv("DUOROU_BOS");
    if (eosStr.empty()) eosStr = getEnv("DUOROU_EOS");
    if (!bosStr.empty()) {
        auto bosIds = parse_ids_or_tokens_csv(bosStr, *vocab);
        if (!bosIds.empty()) vocab->setBOS(bosIds, args.add_special);
    }
    if (!eosStr.empty()) {
        auto eosIds = parse_ids_or_tokens_csv(eosStr, *vocab);
        if (!eosIds.empty()) vocab->setEOS(eosIds, args.add_special);
    }

    std::unique_ptr<TextProcessor> tokenizer;
    if (args.type == "bpe") {
        std::string pattern = !args.pattern.empty() ? args.pattern : R"(\S+|\s+)"; // default safe regex
        tokenizer = std::make_unique<BytePairEncoding>(pattern, vocab);
    } else {
        tokenizer = std::make_unique<SentencePiece>(vocab);
    }

    if (args.mode == "encode") {
        auto ids = tokenizer->encode(args.text, args.add_special);
        std::cout << "[";
        for (size_t i = 0; i < ids.size(); ++i) {
            std::cout << ids[i];
            if (i + 1 < ids.size()) std::cout << ",";
        }
        std::cout << "]\n";
    } else {
        auto ids = parse_ids(args.ids);
        auto text = tokenizer->decode(ids);
        std::cout << text << std::endl;
    }

    return 0;
}