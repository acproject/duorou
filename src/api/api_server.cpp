#include "api_server.h"
#include "../core/model_manager.h"
#include "../core/logger.h"

#include <iostream>
#include <sstream>
#include <regex>
#include <chrono>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>

namespace duorou {

ApiServer::ApiServer(std::shared_ptr<core::ModelManager> model_manager, 
                     std::shared_ptr<core::Logger> logger,
                     int port)
    : model_manager_(model_manager)
    , logger_(logger)
    , port_(port)
    , address_("127.0.0.1")
    , running_(false)
    , server_socket_(-1) {
    setupRoutes();
}

ApiServer::~ApiServer() {
    stop();
}

bool ApiServer::start() {
    if (running_) {
        return true;
    }
    
    // Create socket
    server_socket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket_ < 0) {
        logger_->error("Failed to create socket");
        return false;
    }
    
    // Set socket options
    int opt = 1;
    if (setsockopt(server_socket_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        logger_->error("Failed to set socket options");
        close(server_socket_);
        return false;
    }
    
    // Bind socket
    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = inet_addr(address_.c_str());
    address.sin_port = htons(port_);
    
    if (bind(server_socket_, (struct sockaddr*)&address, sizeof(address)) < 0) {
        logger_->error("Failed to bind socket to port " + std::to_string(port_));
        close(server_socket_);
        return false;
    }
    
    // Listen for connections
    if (listen(server_socket_, 10) < 0) {
        logger_->error("Failed to listen on socket");
        close(server_socket_);
        return false;
    }
    
    running_ = true;
    server_thread_ = std::thread(&ApiServer::serverLoop, this);
    
    logger_->info("API Server started on " + address_ + ":" + std::to_string(port_));
    return true;
}

void ApiServer::stop() {
    if (!running_) {
        return;
    }
    
    running_ = false;
    
    if (server_socket_ >= 0) {
        close(server_socket_);
        server_socket_ = -1;
    }
    
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
    
    logger_->info("API Server stopped");
}

void ApiServer::addRoute(const std::string& method, const std::string& path, RouteHandler handler) {
    routes_[method][path] = handler;
}

void ApiServer::serverLoop() {
    while (running_) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        
        int client_socket = accept(server_socket_, (struct sockaddr*)&client_addr, &client_len);
        if (client_socket < 0) {
            if (running_) {
                logger_->error("Failed to accept client connection");
            }
            continue;
        }
        
        // Handle request in a separate thread for better concurrency
        std::thread([this, client_socket]() {
            // Read request
            char buffer[4096];
            ssize_t bytes_read = recv(client_socket, buffer, sizeof(buffer) - 1, 0);
            
            if (bytes_read > 0) {
                buffer[bytes_read] = '\0';
                
                // Parse HTTP request
                HttpRequest request;
                std::string request_str(buffer);
                
                // Parse request line
                std::istringstream iss(request_str);
                std::string line;
                if (std::getline(iss, line)) {
                    std::istringstream line_stream(line);
                    line_stream >> request.method >> request.path;
                }
                
                // Parse headers
                while (std::getline(iss, line) && line != "\r") {
                    size_t colon_pos = line.find(':');
                    if (colon_pos != std::string::npos) {
                        std::string key = line.substr(0, colon_pos);
                        std::string value = line.substr(colon_pos + 1);
                        // Trim whitespace
                        value.erase(0, value.find_first_not_of(" \t"));
                        value.erase(value.find_last_not_of(" \r") + 1);
                        request.headers[key] = value;
                    }
                }
                
                // Read body if present
                auto content_length_it = request.headers.find("Content-Length");
                if (content_length_it != request.headers.end()) {
                    int content_length = std::stoi(content_length_it->second);
                    if (content_length > 0) {
                        request.body.resize(content_length);
                        recv(client_socket, &request.body[0], content_length, 0);
                    }
                }
                
                // Handle request
                HttpResponse response = handleRequest(request);
                
                // Send response
                std::ostringstream response_stream;
                response_stream << "HTTP/1.1 " << response.status_code << " OK\r\n";
                
                for (const auto& header : response.headers) {
                    response_stream << header.first << ": " << header.second << "\r\n";
                }
                
                response_stream << "Content-Length: " << response.body.length() << "\r\n";
                response_stream << "\r\n";
                response_stream << response.body;
                
                std::string response_str = response_stream.str();
                send(client_socket, response_str.c_str(), response_str.length(), 0);
            }
            
            close(client_socket);
        }).detach();
    }
}

HttpResponse ApiServer::handleRequest(const HttpRequest& request) {
    logger_->info("API Request: " + request.method + " " + request.path);
    
    // Find handler
    RouteHandler handler = findHandler(request.method, request.path);
    if (handler) {
        try {
            return handler(request);
        } catch (const std::exception& e) {
            HttpResponse response;
            response.setError(500, "Internal server error: " + std::string(e.what()), "internal_error");
            return response;
        }
    }
    
    // Route not found
    HttpResponse response;
    response.setError(404, "Route not found: " + request.method + " " + request.path, "not_found_error");
    return response;
}

RouteHandler ApiServer::findHandler(const std::string& method, const std::string& path) {
    auto method_it = routes_.find(method);
    if (method_it != routes_.end()) {
        auto path_it = method_it->second.find(path);
        if (path_it != method_it->second.end()) {
            return path_it->second;
        }
        
        // Try pattern matching for parameterized routes
        for (const auto& route : method_it->second) {
            if (route.first.find("{") != std::string::npos) {
                // Simple pattern matching - can be enhanced
                // For now, just do simple string comparison
                if (route.first == path) {
                    return route.second;
                }
            }
        }
    }
    
    return nullptr;
}

void ApiServer::setupRoutes() {
    // Health and info endpoints
    addRoute("GET", "/health", [this](const HttpRequest& req) { return handleHealth(req); });
    addRoute("GET", "/info", [this](const HttpRequest& req) { return handleInfo(req); });
    
    // Model management endpoints
    addRoute("GET", "/v1/models", [this](const HttpRequest& req) { return handleListModels(req); });
    addRoute("POST", "/v1/models/load", [this](const HttpRequest& req) { return handleLoadModel(req); });
    addRoute("POST", "/v1/models/unload", [this](const HttpRequest& req) { return handleUnloadModel(req); });
    addRoute("GET", "/v1/models/info", [this](const HttpRequest& req) { return handleModelInfo(req); });
    
    // OpenAI compatible endpoints
    addRoute("POST", "/v1/chat/completions", [this](const HttpRequest& req) { return handleChatCompletions(req); });
    addRoute("POST", "/v1/completions", [this](const HttpRequest& req) { return handleCompletions(req); });
    
    // Image generation endpoints
    addRoute("POST", "/v1/images/generations", [this](const HttpRequest& req) { return handleImageGeneration(req); });
}

HttpResponse ApiServer::handleHealth(const HttpRequest& request) {
    HttpResponse response;
    nlohmann::json health_json = {
        {"status", "healthy"},
        {"timestamp", std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count()},
        {"version", "1.0.0"}
    };
    response.setJson(health_json);
    return response;
}

HttpResponse ApiServer::handleInfo(const HttpRequest& request) {
    HttpResponse response;
    nlohmann::json info_json = {
        {"name", "Duorou AI Server"},
        {"version", "1.0.0"},
        {"description", "Multi-modal AI inference server"},
        {"supported_models", nlohmann::json::array({"language", "diffusion"})},
        {"endpoints", nlohmann::json::array({
            "/health", "/info", "/v1/models", "/v1/chat/completions", "/v1/images/generations"
        })}
    };
    response.setJson(info_json);
    return response;
}

HttpResponse ApiServer::handleListModels(const HttpRequest& request) {
    HttpResponse response;
    
    auto models = model_manager_->getAllModels();
    nlohmann::json models_json = nlohmann::json::array();
    
    for (const auto& model : models) {
        nlohmann::json model_json = {
            {"id", model.name},
            {"object", "model"},
            {"created", std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count()},
            {"owned_by", "duorou"},
            {"type", model.type == core::ModelType::LANGUAGE_MODEL ? "language" : "diffusion"},
            {"status", model.status == core::ModelStatus::LOADED ? "loaded" : "not_loaded"},
            {"path", model.path},
            {"memory_usage", model.memory_usage}
        };
        models_json.push_back(model_json);
    }
    
    nlohmann::json result = {
        {"object", "list"},
        {"data", models_json}
    };
    
    response.setJson(result);
    return response;
}

HttpResponse ApiServer::handleLoadModel(const HttpRequest& request) {
    HttpResponse response;
    
    try {
        nlohmann::json req_json = parseRequestBody(request);
        
        if (!req_json.contains("path") || !req_json.contains("type")) {
            response.setError(400, "Missing required fields: path, type", "invalid_request_error");
            return response;
        }
        
        std::string path = req_json["path"];
        std::string type_str = req_json["type"];
        
        core::ModelType type = (type_str == "language") ? core::ModelType::LANGUAGE_MODEL : core::ModelType::DIFFUSION_MODEL;
        
        // Note: ModelManager::loadModel expects model_id, not path
        // For now, use path as model_id - this should be improved
        bool success = model_manager_->loadModel(path);
        
        if (success) {
            nlohmann::json result = {
                {"success", true},
                {"message", "Model loaded successfully"},
                {"path", path},
                {"type", type_str}
            };
            response.setJson(result);
        } else {
            response.setError(500, "Failed to load model", "model_load_error");
        }
        
    } catch (const std::exception& e) {
        response.setError(400, "Invalid JSON request: " + std::string(e.what()), "invalid_request_error");
    }
    
    return response;
}

HttpResponse ApiServer::handleUnloadModel(const HttpRequest& request) {
    HttpResponse response;
    
    try {
        nlohmann::json req_json = parseRequestBody(request);
        
        if (!req_json.contains("path")) {
            response.setError(400, "Missing required field: path", "invalid_request_error");
            return response;
        }
        
        std::string path = req_json["path"];
        // Note: ModelManager::unloadModel expects model_id, not path
        // For now, use path as model_id - this should be improved
        model_manager_->unloadModel(path);
        
        nlohmann::json result = {
            {"success", true},
            {"message", "Model unloaded successfully"},
            {"path", path}
        };
        response.setJson(result);
        
    } catch (const std::exception& e) {
        response.setError(400, "Invalid JSON request: " + std::string(e.what()), "invalid_request_error");
    }
    
    return response;
}

HttpResponse ApiServer::handleModelInfo(const HttpRequest& request) {
    HttpResponse response;
    
    auto models = model_manager_->getAllModels();
    nlohmann::json info_json = {
        {"total_models", models.size()},
        {"loaded_models", std::count_if(models.begin(), models.end(), 
            [](const core::ModelInfo& m) { return m.status == core::ModelStatus::LOADED; })},
        {"total_memory_usage", std::accumulate(models.begin(), models.end(), 0ULL,
            [](size_t sum, const core::ModelInfo& m) { return sum + m.memory_usage; })}
    };
    
    response.setJson(info_json);
    return response;
}

HttpResponse ApiServer::handleChatCompletions(const HttpRequest& request) {
    HttpResponse response;
    
    try {
        nlohmann::json req_json = parseRequestBody(request);
        
        if (!req_json.contains("messages")) {
            response.setError(400, "Missing required field: messages", "invalid_request_error");
            return response;
        }
        
        // Extract parameters
        auto messages = req_json["messages"];
        std::string model = req_json.value("model", "default");
        int max_tokens = req_json.value("max_tokens", 100);
        float temperature = req_json.value("temperature", 0.7f);
        
        // Build prompt from messages
        std::string prompt;
        for (const auto& message : messages) {
            std::string role = message["role"];
            std::string content = message["content"];
            prompt += role + ": " + content + "\n";
        }
        prompt += "assistant: ";
        
        // Generate response using model manager
        // Note: This is a placeholder - actual text generation would need to be implemented
        std::string generated_text = "Generated response for: " + prompt.substr(0, 50) + "...";
        
        // Format OpenAI-compatible response
        nlohmann::json result = {
            {"id", "chatcmpl-" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count())},
            {"object", "chat.completion"},
            {"created", std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count()},
            {"model", model},
            {"choices", nlohmann::json::array({
                {
                    {"index", 0},
                    {"message", {
                        {"role", "assistant"},
                        {"content", generated_text}
                    }},
                    {"finish_reason", "stop"}
                }
            })},
            {"usage", {
                {"prompt_tokens", prompt.length() / 4}, // Rough estimate
                {"completion_tokens", generated_text.length() / 4},
                {"total_tokens", (prompt.length() + generated_text.length()) / 4}
            }}
        };
        
        response.setJson(result);
        
    } catch (const std::exception& e) {
        response.setError(400, "Invalid request: " + std::string(e.what()), "invalid_request_error");
    }
    
    return response;
}

HttpResponse ApiServer::handleCompletions(const HttpRequest& request) {
    HttpResponse response;
    
    try {
        nlohmann::json req_json = parseRequestBody(request);
        
        if (!req_json.contains("prompt")) {
            response.setError(400, "Missing required field: prompt", "invalid_request_error");
            return response;
        }
        
        std::string prompt = req_json["prompt"];
        std::string model = req_json.value("model", "default");
        int max_tokens = req_json.value("max_tokens", 100);
        float temperature = req_json.value("temperature", 0.7f);
        
        // Generate response
        // Note: This is a placeholder - actual text generation would need to be implemented
        std::string generated_text = "Generated response for: " + prompt.substr(0, 50) + "...";
        
        // Format OpenAI-compatible response
        nlohmann::json result = {
            {"id", "cmpl-" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count())},
            {"object", "text_completion"},
            {"created", std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count()},
            {"model", model},
            {"choices", nlohmann::json::array({
                {
                    {"text", generated_text},
                    {"index", 0},
                    {"finish_reason", "stop"}
                }
            })},
            {"usage", {
                {"prompt_tokens", prompt.length() / 4},
                {"completion_tokens", generated_text.length() / 4},
                {"total_tokens", (prompt.length() + generated_text.length()) / 4}
            }}
        };
        
        response.setJson(result);
        
    } catch (const std::exception& e) {
        response.setError(400, "Invalid request: " + std::string(e.what()), "invalid_request_error");
    }
    
    return response;
}

HttpResponse ApiServer::handleImageGeneration(const HttpRequest& request) {
    HttpResponse response;
    
    try {
        nlohmann::json req_json = parseRequestBody(request);
        
        if (!req_json.contains("prompt")) {
            response.setError(400, "Missing required field: prompt", "invalid_request_error");
            return response;
        }
        
        std::string prompt = req_json["prompt"];
        std::string model = req_json.value("model", "default");
        int width = req_json.value("width", 512);
        int height = req_json.value("height", 512);
        int steps = req_json.value("steps", 20);
        float cfg_scale = req_json.value("cfg_scale", 7.5f);
        
        // Generate image using model manager
        // Note: This is a placeholder - actual image generation would need to be implemented
        std::string image_data = ""; // Empty for now
        
        if (image_data.empty()) {
            response.setError(500, "Failed to generate image", "image_generation_error");
            return response;
        }
        
        // Format response
        nlohmann::json result = {
            {"created", std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count()},
            {"data", nlohmann::json::array({
                {
                    {"url", "data:image/png;base64," + image_data}, // Base64 encoded image
                    {"width", width},
                    {"height", height}
                }
            })}
        };
        
        response.setJson(result);
        
    } catch (const std::exception& e) {
        response.setError(400, "Invalid request: " + std::string(e.what()), "invalid_request_error");
    }
    
    return response;
}

std::string ApiServer::extractModelId(const std::string& path) {
    size_t pos = path.find_last_of('/');
    if (pos != std::string::npos) {
        return path.substr(pos + 1);
    }
    return path;
}

nlohmann::json ApiServer::parseRequestBody(const HttpRequest& request) {
    if (request.body.empty()) {
        return nlohmann::json::object();
    }
    
    return nlohmann::json::parse(request.body);
}

} // namespace duorou