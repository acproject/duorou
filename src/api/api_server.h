#pragma once

#include <string>
#include <memory>
#include <functional>
#include <thread>
#include <atomic>
#include <map>
#include "../../third_party/llama.cpp/vendor/nlohmann/json.hpp"

namespace duorou {

namespace core {
    class ModelManager;
    class Logger;
}

class ModelManager;
class Logger;

// HTTP request structure
struct HttpRequest {
    std::string method;
    std::string path;
    std::map<std::string, std::string> headers;
    std::string body;
    std::map<std::string, std::string> query_params;
};

// HTTP response structure
struct HttpResponse {
    int status_code = 200;
    std::map<std::string, std::string> headers;
    std::string body;
    
    void setJson(const nlohmann::json& json) {
        headers["Content-Type"] = "application/json";
        body = json.dump();
    }
    
    void setError(int code, const std::string& message, const std::string& type = "error") {
        status_code = code;
        nlohmann::json error_json = {
            {"error", {
                {"code", code},
                {"message", message},
                {"type", type}
            }}
        };
        setJson(error_json);
    }
};

// Route handler function type
using RouteHandler = std::function<HttpResponse(const HttpRequest&)>;

// API Server class
class ApiServer {
public:
    ApiServer(std::shared_ptr<core::ModelManager> model_manager, 
              std::shared_ptr<core::Logger> logger,
              int port = 8080);
    ~ApiServer();
    
    // Server control
    bool start();
    void stop();
    bool isRunning() const { return running_; }
    
    // Route registration
    void addRoute(const std::string& method, const std::string& path, RouteHandler handler);
    
    // Get server info
    int getPort() const { return port_; }
    std::string getAddress() const { return address_; }
    
private:
    // Core server functionality
    void serverLoop();
    HttpResponse handleRequest(const HttpRequest& request);
    
    // Route matching
    RouteHandler findHandler(const std::string& method, const std::string& path);
    
    // Built-in API endpoints
    void setupRoutes();
    
    // Health and info endpoints
    HttpResponse handleHealth(const HttpRequest& request);
    HttpResponse handleInfo(const HttpRequest& request);
    
    // Model management endpoints
    HttpResponse handleListModels(const HttpRequest& request);
    HttpResponse handleLoadModel(const HttpRequest& request);
    HttpResponse handleUnloadModel(const HttpRequest& request);
    HttpResponse handleModelInfo(const HttpRequest& request);
    
    // Text generation endpoints (OpenAI compatible)
    HttpResponse handleChatCompletions(const HttpRequest& request);
    HttpResponse handleCompletions(const HttpRequest& request);
    
    // Image generation endpoints
    HttpResponse handleImageGeneration(const HttpRequest& request);
    
    // Utility methods
    std::string extractModelId(const std::string& path);
    nlohmann::json parseRequestBody(const HttpRequest& request);
    
private:
    std::shared_ptr<core::ModelManager> model_manager_;
    std::shared_ptr<core::Logger> logger_;
    
    int port_;
    std::string address_;
    std::atomic<bool> running_;
    std::thread server_thread_;
    
    // Route storage
    std::map<std::string, std::map<std::string, RouteHandler>> routes_; // method -> path -> handler
    
    // Server socket (implementation specific)
    int server_socket_;
};

} // namespace duorou