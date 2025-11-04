// src/server/CommandHandler.hpp
#pragma once
#include "DataStore.hpp"
#include <algorithm>
#include <functional>
#include <unordered_map>
#include <vector>
#include <string>

class CommandHandler {
public:
    using CommandFunction = std::function<std::string(DataStore&, const std::vector<std::string>&)>;
    
    explicit CommandHandler(DataStore& store) : data_store(store) {
        // 注册所有命令
        command_map["MULTI"] = multi_command;
        command_map["EXEC"] = exec_command;
        command_map["DISCARD"] = discard_command;
        command_map["WATCH"] = watch_command;
        command_map["UNWATCH"] = unwatch_command;
        command_map["RENAME"] = rename_command;
        command_map["SCAN"] = scan_command;
        command_map["PING"] = ping_command;
        command_map["SET"] = set_command;
        command_map["GET"] = get_command;
        command_map["DEL"] = del_command;
        command_map["EXISTS"] = exists_command;
        command_map["INCR"] = incr_command;
        command_map["SETNX"] = set_numeric_command;
        command_map["GETNX"] = get_numeric_command;
        // 向量读写别名，便于直观使用
        command_map["VSET"] = set_numeric_command;
        command_map["VGET"] = get_numeric_command;
        command_map["SELECT"] = select_command;
        command_map["PEXPIRE"] = pexpire_command;
        command_map["PTTL"] = pttl_command;
        command_map["SAVE"] = save_command;
        command_map["INFO"] = info_command;
        command_map["KEYS"] = keys_command;
        command_map["FLUSHDB"] = flushdb_command;
        command_map["FLUSHALL"] = flushall_command;
        command_map["BGREWRITEAOF"] = bgrewriteaof_command;

        // 元数据与冷热标记
        command_map["METASET"] = metaset_command;
        command_map["METAGET"] = metaget_command;
        command_map["TAGADD"] = tagadd_command;
        command_map["HOTSET"] = hotset_command;

        // 二进制对象
        command_map["OBJSET"] = objset_command;
        command_map["OBJGET"] = objget_command;

        // 图抽象（简单邻接）
        command_map["GRAPH.ADDEDGE"] = graph_addedge_command;
        command_map["GRAPH.NEIGHBORS"] = graph_neighbors_command;
    }

    // 将 handle_command 移到 public 部分
    std::string handle_command(const std::vector<std::string>& args);

private:
    DataStore& data_store;
    std::unordered_map<std::string, CommandFunction> command_map;

    // 工具方法
    static std::string protocol_error(const std::string& msg) {
        return "-ERR " + msg + "\r\n";
    }

    static std::string to_uppercase(std::string str) {
        std::transform(str.begin(), str.end(), str.begin(), ::toupper);
        return str;
    }

    // 声明所有命令处理函数
    static std::string multi_command(DataStore& store, const std::vector<std::string>& args);
    static std::string exec_command(DataStore& store, const std::vector<std::string>& args);
    static std::string discard_command(DataStore& store, const std::vector<std::string>& args);
    static std::string watch_command(DataStore& store, const std::vector<std::string>& args);
    static std::string unwatch_command(DataStore& store, const std::vector<std::string>& args);
    static std::string rename_command(DataStore& store, const std::vector<std::string>& args);
    static std::string scan_command(DataStore& store, const std::vector<std::string>& args);
    static std::string ping_command(DataStore& store, const std::vector<std::string>& args);
    static std::string set_command(DataStore& store, const std::vector<std::string>& args);
    static std::string get_command(DataStore& store, const std::vector<std::string>& args);
    static std::string del_command(DataStore& store, const std::vector<std::string>& args);
    static std::string exists_command(DataStore& store, const std::vector<std::string>& args);
    static std::string incr_command(DataStore& store, const std::vector<std::string>& args);
    static std::string set_numeric_command(DataStore& store, const std::vector<std::string>& args);
    static std::string get_numeric_command(DataStore& store, const std::vector<std::string>& args);
    static std::string select_command(DataStore& store, const std::vector<std::string>& args);
    static std::string pexpire_command(DataStore& store, const std::vector<std::string>& args);
    static std::string pttl_command(DataStore& store, const std::vector<std::string>& args);
    static std::string save_command(DataStore& store, const std::vector<std::string>& args);
    static std::string info_command(DataStore& store, const std::vector<std::string>& args);
    static std::string keys_command(DataStore& store, const std::vector<std::string>& args);
    static std::string flushdb_command(DataStore& store, const std::vector<std::string>& args);
    static std::string flushall_command(DataStore& store, const std::vector<std::string>& args);
    static std::string bgrewriteaof_command(DataStore& store, const std::vector<std::string>& args);

    // 扩展：元数据/冷热标记
    static std::string metaset_command(DataStore& store, const std::vector<std::string>& args);
    static std::string metaget_command(DataStore& store, const std::vector<std::string>& args);
    static std::string tagadd_command(DataStore& store, const std::vector<std::string>& args);
    static std::string hotset_command(DataStore& store, const std::vector<std::string>& args);

    // 扩展：对象化二进制
    static std::string objset_command(DataStore& store, const std::vector<std::string>& args);
    static std::string objget_command(DataStore& store, const std::vector<std::string>& args);

    // 扩展：图抽象
    static std::string graph_addedge_command(DataStore& store, const std::vector<std::string>& args);
    static std::string graph_neighbors_command(DataStore& store, const std::vector<std::string>& args);
};