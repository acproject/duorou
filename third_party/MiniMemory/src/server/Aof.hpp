#pragma once
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <sys/stat.h>
#include "RespParser.hpp"
#include "CommandHandler.hpp"

class AofWriter {
public:
    explicit AofWriter(const std::string& path)
        : path_(path) {
        openAppend();
    }

    ~AofWriter() {
        if (ofs_.is_open()) ofs_.close();
    }

    void append(const std::vector<std::string>& args) {
        if (!ofs_.is_open()) openAppend();
        if (!ofs_) return;
        writeResp(args);
        ofs_.flush();
    }

    static bool replay(const std::string& path, DataStore& store) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs) return false;
        CommandHandler handler(store);
        std::string buf;
        buf.reserve(1 << 20);
        store.setLoading(true);
        try {
            std::string chunk(8192, '\0');
            while (ifs) {
                ifs.read(&chunk[0], static_cast<std::streamsize>(chunk.size()));
                std::streamsize got = ifs.gcount();
                if (got > 0) buf.append(chunk.data(), static_cast<size_t>(got));
                while (true) {
                    auto args = RespParser::parse(buf);
                    if (args.empty()) break;
                    handler.handle_command(args);
                }
            }
            store.setLoading(false);
            return true;
        } catch (...) {
            store.setLoading(false);
            return false;
        }
    }

    static bool rewritePlainResp(DataStore& store, const std::string& outPath) {
        // 在重写期间抑制 on_apply 以避免污染现有 AOF
        bool prevLoading = store.isLoading();
        store.setLoading(true);

        std::ofstream ofs(outPath, std::ios::binary | std::ios::trunc);
        if (!ofs) {
            store.setLoading(prevLoading);
            return false;
        }
        auto writeResp = [&](const std::vector<std::string>& args){
            ofs << "*" << args.size() << "\r\n";
            for (const auto& a : args) {
                ofs << "$" << a.size() << "\r\n";
                ofs.write(a.data(), static_cast<std::streamsize>(a.size()));
                ofs << "\r\n";
            }
        };
        std::string info = store.info();
        int dbs = 16;
        int current = 0;
        {
            std::istringstream is(info);
            std::string line;
            while (std::getline(is, line)) {
                if (line.rfind("databases:", 0) == 0) {
                    try { dbs = std::stoi(line.substr(10)); } catch (...) {}
                } else if (line.rfind("current_db:", 0) == 0) {
                    try { current = std::stoi(line.substr(11)); } catch (...) {}
                }
            }
        }
        for (int i = 0; i < dbs; ++i) {
            if (!store.select(i)) break;
            writeResp({"SELECT", std::to_string(i)});
            auto keys = store.keys("*");
            for (const auto& k : keys) {
                auto v = store.get(k);
                if (v != "(nil)") writeResp({"SET", k, v});
            }
        }
        store.select(current);
        ofs.flush();
        // 恢复 loading 状态
        store.setLoading(prevLoading);
        return true;
    }

private:
    std::string path_;
    std::ofstream ofs_;

    void openAppend() {
        ofs_.open(path_, std::ios::binary | std::ios::app);
    }

    void writeResp(const std::vector<std::string>& args) {
        ofs_ << "*" << args.size() << "\r\n";
        for (const auto& a : args) {
            ofs_ << "$" << a.size() << "\r\n";
            ofs_.write(a.data(), static_cast<std::streamsize>(a.size()));
            ofs_ << "\r\n";
        }
    }
};