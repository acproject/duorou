#include "CommandHandler.hpp"
#include <string>
#include <cstdio>
#include "Aof.hpp"
#include <unordered_set>

// 基础命令
std::string CommandHandler::set_command(DataStore& store, const std::vector<std::string>& args) {
    if(args.size() < 3) return protocol_error("Wrong number of arguments");
    store.set(args[1], args[2]);
    return "+OK\r\n";
}

std::string CommandHandler::get_command(DataStore& store, const std::vector<std::string>& args) {
    if (args.size() != 2) {
        return "-ERR wrong number of arguments for 'get' command\r\n";
    }
    std::string value = store.get(args[1]);  // 使用传入的 store 而不是 dataStore
    if (value == "(nil)") {
        return "$-1\r\n";  // 返回 nil 值
    }
    return "$" + std::to_string(value.length()) + "\r\n" + value + "\r\n";
}

std::string CommandHandler::del_command(DataStore& store, const std::vector<std::string>& args) {
    if(args.size() < 2) return protocol_error("Wrong number of arguments");
    int deleted = 0;
    for(size_t i = 1; i < args.size(); ++i) {
        deleted += store.del(args[i]);
    }
    return ":" + std::to_string(deleted) + "\r\n";
}

// 事务命令
std::string CommandHandler::multi_command(DataStore& store, const std::vector<std::string>& args) {
    store.multi();
    return "+OK\r\n";
}

std::string CommandHandler::exec_command(DataStore& store, const std::vector<std::string>& args) {
    auto results = store.exec();
    if (!results) {
        return "-ERR Transaction failed\r\n";
    }else {
        return "+OK\r\n";
    }
}

std::string CommandHandler::discard_command(DataStore& store, const std::vector<std::string>& args) {
    store.discard();
    return "+OK\r\n";
}

std::string CommandHandler::watch_command(DataStore& store, const std::vector<std::string>& args) {
    if(args.size() != 2) return protocol_error("Wrong number of arguments");
    if (store.watch(args[1])) {
        return "+OK\r\n";
    }
    return "-ERR Watch failed\r\n";
}

std::string CommandHandler::unwatch_command(DataStore& store, const std::vector<std::string>& args) {
    store.unwatch();
    return "+OK\r\n";
}

// 数据库管理命令
std::string CommandHandler::select_command(DataStore& store, const std::vector<std::string>& args) {
    if(args.size() != 2) return protocol_error("Wrong number of arguments");
    int index = std::stoi(args[1]);
    if (store.select(index)) {
        return "+OK\r\n";
    }
    return protocol_error("Invalid DB index");
}

std::string CommandHandler::flushdb_command(DataStore& store, const std::vector<std::string>& args) {
    if (args.size() != 1) {
        return protocol_error("Wrong number of arguments");
    }
    
    store.flushdb();
    return "+OK\r\n";
}

std::string CommandHandler::flushall_command(DataStore& store, const std::vector<std::string>& args) {
    store.flushall();
    return "+OK\r\n";
}

// AOF 重写：BGREWRITEAOF [path]
// 若未提供 path，默认使用 "appendonly.aof"
std::string CommandHandler::bgrewriteaof_command(DataStore& store, const std::vector<std::string>& args) {
    std::string target = "appendonly.aof";
    if (args.size() > 2) {
        return protocol_error("Wrong number of arguments");
    }
    if (args.size() == 2) {
        target = args[1];
    }

    std::string tmp = target + ".tmp";
    if (!AofWriter::rewritePlainResp(store, tmp)) {
        return "-ERR AOF rewrite failed\r\n";
    }

    // 尝试替换目标文件
    std::remove(target.c_str());
    if (std::rename(tmp.c_str(), target.c_str()) != 0) {
        return "-ERR failed to replace AOF file\r\n";
    }
    return "+OK\r\n";
}

// 键空间命令
std::string CommandHandler::exists_command(DataStore& store, const std::vector<std::string>& args) {
    if(args.size() < 2) return protocol_error("Wrong number of arguments");
    int count = 0;
    for(size_t i = 1; i < args.size(); ++i) {
        count += store.exists(args[i]);
    }
    return ":" + std::to_string(count) + "\r\n";
}

std::string CommandHandler::keys_command(DataStore& store, const std::vector<std::string>& args) {
    if(args.size() != 2) return protocol_error("Wrong number of arguments");
    auto keys = store.keys(args[1]);
    std::string result = "*" + std::to_string(keys.size()) + "\r\n";
    for(const auto& key : keys) {
        result += "$" + std::to_string(key.length()) + "\r\n" + key + "\r\n";
    }
    return result;
}

std::string CommandHandler::scan_command(DataStore& store, const std::vector<std::string>& args) {
    if(args.size() < 2) return protocol_error("Wrong number of arguments");
    size_t count = args.size() > 2 ? std::stoul(args[2]) : 10;
    auto keys = store.scan(args[1], count);
    std::string result = "*" + std::to_string(keys.size()) + "\r\n";
    for(const auto& key : keys) {
        result += "$" + std::to_string(key.length()) + "\r\n" + key + "\r\n";
    }
    return result;
}

// 其他命令
std::string CommandHandler::ping_command(DataStore& store, const std::vector<std::string>& args) {
    return "+PONG\r\n";
}

std::string CommandHandler::save_command(DataStore& store, const std::vector<std::string>& args) {
    if(store.saveMCDB("dump.mcdb")) {
        return "+OK\r\n";
    } else {
        return "-ERR Failed to save RDB\r\n";
    }
}

std::string CommandHandler::info_command(DataStore& store, const std::vector<std::string>& args) {
    std::string info = store.info();
    return "$" + std::to_string(info.length()) + "\r\n" + info + "\r\n";
}

// 过期命令
std::string CommandHandler::pexpire_command(DataStore& store, const std::vector<std::string>& args) {
    if(args.size() != 3) return protocol_error("Wrong number of arguments");
    // 先检查键是否存在
    if (store.exists(args[1]) == 0) {
        return ":0\r\n"; // 键不存在，返回0
    }
    store.pexpire(args[1], std::stoll(args[2]));
    return ":1\r\n";
}

std::string CommandHandler::pttl_command(DataStore& store, const std::vector<std::string>& args) {
    if(args.size() != 2) return protocol_error("Wrong number of arguments");
    return ":" + std::to_string(store.pttl(args[1])) + "\r\n";
}

// 数值操作命令
std::string CommandHandler::incr_command(DataStore& store, const std::vector<std::string>& args) {
    if(args.size() != 2) return protocol_error("Wrong number of arguments");
    std::string value = store.get(args[1]);
    int num = value == "(nil)" ? 0 : std::stoi(value);
    store.set(args[1], std::to_string(num + 1));
    return ":" + std::to_string(num + 1) + "\r\n";
}

std::string CommandHandler::set_numeric_command(DataStore& store, const std::vector<std::string>& args) {
     if(args.size() < 3) return protocol_error("Wrong number of arguments for SETNX");
    
    try {
        std::vector<float> values;
        for(size_t i = 2; i < args.size(); ++i) {
            values.push_back(std::stof(args[i]));
        }
        
        if(store.set_numeric(args[1], values)) {
            return "+OK\r\n";
        } else {
            return "-ERR Failed to set numeric values\r\n";
        }
    } catch(const std::exception& e) {
        return "-ERR " + std::string(e.what()) + "\r\n";
    }
}

std::string CommandHandler::get_numeric_command(DataStore& store, const std::vector<std::string>& args) {
    if(args.size() != 2) return protocol_error("Wrong number of arguments for GETNX");
    
    std::vector<float> values = store.get_numeric(args[1]);
    if(values.empty()) {
        return "*0\r\n";
    }
    
    std::string result = "*" + std::to_string(values.size()) + "\r\n";
    for(const auto& val : values) {
        std::string str_val = std::to_string(val);
        // 移除尾部多余的0
        str_val.erase(str_val.find_last_not_of('0') + 1, std::string::npos);
        if(str_val.back() == '.') str_val.pop_back();
        
        result += "$" + std::to_string(str_val.length()) + "\r\n" + str_val + "\r\n";
    }
    return result;
}

// 键重命名命令
std::string CommandHandler::rename_command(DataStore& store, const std::vector<std::string>& args) {
    if(args.size() != 3) return protocol_error("Wrong number of arguments");
    if (store.rename(args[1], args[2])) {
        return "+OK\r\n";
    }
    return protocol_error("No such key");
}

// --- 扩展：元数据与冷热标记 ---
// METASET key field value -> 存储为 __meta:<key>:<field>
std::string CommandHandler::metaset_command(DataStore& store, const std::vector<std::string>& args) {
    if (args.size() != 4) return protocol_error("Wrong number of arguments for METASET");
    const std::string& key = args[1];
    const std::string& field = args[2];
    const std::string& value = args[3];
    std::string metaKey = std::string("__meta:") + key + ":" + field;
    store.set(metaKey, value);
    return "+OK\r\n";
}

// METAGET key [field]
// 无 field: 以 RESP 数组返回 [field, value, ...]
std::string CommandHandler::metaget_command(DataStore& store, const std::vector<std::string>& args) {
    if (args.size() < 2 || args.size() > 3) return protocol_error("Wrong number of arguments for METAGET");
    const std::string& key = args[1];
    if (args.size() == 3) {
        std::string metaKey = std::string("__meta:") + key + ":" + args[2];
        std::string val = store.get(metaKey);
        if (val == "(nil)") return "$-1\r\n";
        return "$" + std::to_string(val.size()) + "\r\n" + val + "\r\n";
    }
    // 汇总已存在的字段
    auto keys = store.keys(std::string("__meta:") + key + ":*");
    if (keys.empty()) return "*0\r\n";
    std::string resp = "*" + std::to_string(keys.size() * 2) + "\r\n";
    for (const auto& mk : keys) {
        // 字段名为最后一个冒号后的片段
        size_t pos = mk.rfind(':');
        std::string field = pos == std::string::npos ? mk : mk.substr(pos + 1);
        std::string val = store.get(mk);
        if (val == "(nil)") val = "";
        resp += "$" + std::to_string(field.size()) + "\r\n" + field + "\r\n";
        resp += "$" + std::to_string(val.size()) + "\r\n" + val + "\r\n";
    }
    return resp;
}

// TAGADD key tag1 tag2 ... -> 去重追加，存储为 __meta:<key>:tags 逗号分隔
std::string CommandHandler::tagadd_command(DataStore& store, const std::vector<std::string>& args) {
    if (args.size() < 3) return protocol_error("Wrong number of arguments for TAGADD");
    const std::string& key = args[1];
    std::string tagKey = std::string("__meta:") + key + ":tags";
    std::string existing = store.get(tagKey);
    if (existing == "(nil)") existing.clear();
    std::unordered_set<std::string> tags;
    // 解析已有
    size_t start = 0;
    while (start < existing.size()) {
        size_t comma = existing.find(',', start);
        std::string t = existing.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!t.empty()) tags.insert(t);
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    // 添加新标签
    for (size_t i = 2; i < args.size(); ++i) {
        if (!args[i].empty()) tags.insert(args[i]);
    }
    // 重建字符串
    std::string out;
    for (const auto& t : tags) {
        if (!out.empty()) out.push_back(',');
        out += t;
    }
    store.set(tagKey, out);
    return "+OK\r\n";
}

// HOTSET key score -> 设置 __meta:<key>:hot_score 与 __meta:<key>:hot
std::string CommandHandler::hotset_command(DataStore& store, const std::vector<std::string>& args) {
    if (args.size() != 3) return protocol_error("Wrong number of arguments for HOTSET");
    const std::string& key = args[1];
    const std::string& score = args[2];
    std::string scoreKey = std::string("__meta:") + key + ":hot_score";
    store.set(scoreKey, score);
    // 简单阈值：>=5 视为热点
    bool is_hot = false;
    try { is_hot = std::stof(score) >= 5.0f; } catch (...) { is_hot = false; }
    std::string hotKey = std::string("__meta:") + key + ":hot";
    store.set(hotKey, is_hot ? "1" : "0");
    return "+OK\r\n";
}

// OBJSET key mime data -> __obj:<key>:data, __obj:<key>:mime
std::string CommandHandler::objset_command(DataStore& store, const std::vector<std::string>& args) {
    if (args.size() != 4) return protocol_error("Wrong number of arguments for OBJSET");
    const std::string& key = args[1];
    const std::string& mime = args[2];
    const std::string& data = args[3];
    std::string dataKey = std::string("__obj:") + key + ":data";
    std::string mimeKey = std::string("__obj:") + key + ":mime";
    store.set(dataKey, data);
    store.set(mimeKey, mime);
    return "+OK\r\n";
}

// OBJGET key -> [mime, data]
std::string CommandHandler::objget_command(DataStore& store, const std::vector<std::string>& args) {
    if (args.size() != 2) return protocol_error("Wrong number of arguments for OBJGET");
    const std::string& key = args[1];
    std::string dataKey = std::string("__obj:") + key + ":data";
    std::string mimeKey = std::string("__obj:") + key + ":mime";
    std::string data = store.get(dataKey);
    std::string mime = store.get(mimeKey);
    if (data == "(nil)") return "*0\r\n";
    if (mime == "(nil)") mime.clear();
    std::string resp;
    resp.reserve(32 + mime.size() + data.size());
    resp += "*2\r\n";
    resp += "$" + std::to_string(mime.size()) + "\r\n" + mime + "\r\n";
    resp += "$" + std::to_string(data.size()) + "\r\n" + data + "\r\n";
    return resp;
}

// GRAPH.ADDEDGE from relation to -> 追加到 __graph:adj:<from>
std::string CommandHandler::graph_addedge_command(DataStore& store, const std::vector<std::string>& args) {
    if (args.size() != 4) return protocol_error("Wrong number of arguments for GRAPH.ADDEDGE");
    const std::string& from = args[1];
    const std::string& rel  = args[2];
    const std::string& to   = args[3];
    std::string adjKey = std::string("__graph:adj:") + from;
    std::string cur = store.get(adjKey);
    if (cur == "(nil)") cur.clear();
    if (!cur.empty()) cur.push_back(',');
    cur += rel + ":" + to;
    store.set(adjKey, cur);
    return "+OK\r\n";
}

// GRAPH.NEIGHBORS id -> 数组返回 relation:to 列表
std::string CommandHandler::graph_neighbors_command(DataStore& store, const std::vector<std::string>& args) {
    if (args.size() != 2) return protocol_error("Wrong number of arguments for GRAPH.NEIGHBORS");
    const std::string& node = args[1];
    std::string adjKey = std::string("__graph:adj:") + node;
    std::string cur = store.get(adjKey);
    if (cur == "(nil)" || cur.empty()) return "*0\r\n";
    // 统计元素数量
    size_t count = 1;
    for (char c : cur) if (c == ',') ++count;
    std::string resp = "*" + std::to_string(count) + "\r\n";
    size_t start = 0;
    while (start <= cur.size()) {
        size_t comma = cur.find(',', start);
        std::string item = cur.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        resp += "$" + std::to_string(item.size()) + "\r\n" + item + "\r\n";
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return resp;
}

std::string CommandHandler::handle_command(const std::vector<std::string>& args) {
    if(args.empty()) return protocol_error("Empty command");
    
    const std::string& cmd = args[0];
    
    // 直接处理一些常用命令，提高性能
    if (cmd == "GET" || cmd == "get") {
        if (args.size() != 2) {
            return "-ERR wrong number of arguments for 'get' command\r\n";
        }
        std::string value = data_store.get(args[1]);
        if (value == "(nil)") {
            return "$-1\r\n";  // 返回 nil 值
        }
        return "$" + std::to_string(value.length()) + "\r\n" + value + "\r\n";
    }
    
    auto it = command_map.find(to_uppercase(cmd));
    
    if(it == command_map.end()) {
        return protocol_error("Unknown command");
    }
    
    try {
        return it->second(data_store, args);
    } catch(const std::exception& e) {
        return protocol_error(e.what());
    }
}

// 删除 to_uppercase 函数的实现，因为它已经在头文件中定义