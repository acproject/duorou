#include "model_downloader.h"
#include "model_path_manager.h"

using namespace duorou::core;
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <chrono>
#include <filesystem>
#include <regex>
#include <algorithm>
#include <curl/curl.h>
#if __has_include(<nlohmann/json.hpp>)
#  include <nlohmann/json.hpp>
#elif __has_include("../../third_party/llama.cpp/vendor/nlohmann/json.hpp")
#  include "../../third_party/llama.cpp/vendor/nlohmann/json.hpp"
#else
#  warning "nlohmann/json.hpp not found; JSON-dependent features will be disabled"
#  define DUOROU_NO_JSON 1
#endif
#include <openssl/sha.h>

// Ensure filesystem is available
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#error "Filesystem library not available"
#endif

namespace duorou {

#ifndef DUOROU_NO_JSON
using json = nlohmann::json;
#endif

/**
 * @brief HTTP response data structure
 */
struct HttpResponse {
    std::string data;
    long response_code = 0;
    std::string error_message;
};

/**
 * @brief Download context
 */
struct DownloadContext {
    DownloadProgressCallback callback;
    size_t total_size = 0;
    size_t downloaded_size = 0;
    std::chrono::steady_clock::time_point start_time;
    std::ofstream* file = nullptr;
};

/**
 * @brief CURL write callback function
 */
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* userp) {
    size_t total_size = size * nmemb;
    userp->append((char*)contents, total_size);
    return total_size;
}

/**
 * @brief CURL file write callback function
 */
static size_t WriteFileCallback(void* contents, size_t size, size_t nmemb, DownloadContext* ctx) {
    size_t total_size = size * nmemb;
    
    if (ctx->file) {
        ctx->file->write((char*)contents, total_size);
        ctx->downloaded_size += total_size;
        
        // Call progress callback
        if (ctx->callback && ctx->total_size > 0) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - ctx->start_time).count();
            double speed = elapsed > 0 ? (ctx->downloaded_size * 1000.0 / elapsed) : 0.0;
            ctx->callback(ctx->downloaded_size, ctx->total_size, speed);
        }
    }
    
    return total_size;
}

/**
 * @brief CURL progress callback function
 */
static int ProgressCallback(void* clientp, curl_off_t dltotal, curl_off_t dlnow, curl_off_t /*ultotal*/, curl_off_t /*ulnow*/) {
    DownloadContext* ctx = static_cast<DownloadContext*>(clientp);
    
    if (dltotal > 0) {
        ctx->total_size = dltotal;
        ctx->downloaded_size = dlnow;
        
        if (ctx->callback) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - ctx->start_time).count();
            double speed = elapsed > 0 ? (dlnow * 1000.0 / elapsed) : 0.0;
            ctx->callback(dlnow, dltotal, speed);
        }
    }
    
    return 0;
}

/**
 * @brief ModelDownloader implementation class
 */
class ModelDownloader::Impl {
public:
    std::string base_url_;
    std::string model_dir_;
    std::unique_ptr<ModelPathManager> path_manager_;
    DownloadProgressCallback progress_callback_;
    size_t max_cache_size_;
    
    Impl(const std::string& base_url, const std::string& model_dir)
        : base_url_(base_url), model_dir_(expandPath(model_dir)), max_cache_size_(10ULL * 1024 * 1024 * 1024) { // 10GB default
        
        // Initialize CURL
        curl_global_init(CURL_GLOBAL_DEFAULT);
        
        // Create model directory
        fs::create_directories(model_dir_);
        
        // Initialize path manager
        path_manager_ = std::make_unique<ModelPathManager>(model_dir_);
    }
    
    ~Impl() {
        curl_global_cleanup();
    }
    
    /**
     * @brief Expand ~ symbol in path
     */
    std::string expandPath(const std::string& path) {
        if (path.empty() || path[0] != '~') {
            return path;
        }
        
        const char* home = getenv("HOME");
        if (!home) {
            return path;
        }
        
        return std::string(home) + path.substr(1);
    }
    
    /**
     * @brief Execute HTTP GET request
     */
    HttpResponse httpGet(const std::string& url) {
        HttpResponse response;
        
        CURL* curl = curl_easy_init();
        if (!curl) {
            response.error_message = "Failed to initialize CURL";
            return response;
        }
        
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response.data);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
        
        CURLcode res = curl_easy_perform(curl);
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response.response_code);
        
        if (res != CURLE_OK) {
            response.error_message = curl_easy_strerror(res);
        }
        
        curl_easy_cleanup(curl);
        return response;
    }
    
    /**
     * @brief Download file
     */
    DownloadResult downloadFile(const std::string& url, const std::string& local_path, DownloadContext& ctx) {
        DownloadResult result;
        
        // Create directory
        fs::create_directories(fs::path(local_path).parent_path());
        
        std::ofstream file(local_path, std::ios::binary);
        if (!file.is_open()) {
            result.error_message = "Failed to create local file: " + local_path;
            return result;
        }
        
        ctx.file = &file;
        ctx.start_time = std::chrono::steady_clock::now();
        
        CURL* curl = curl_easy_init();
        if (!curl) {
            result.error_message = "Failed to initialize CURL";
            return result;
        }
        
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteFileCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ctx);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, ProgressCallback);
        curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &ctx);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 3600L); // 1 hour timeout
        
        auto start_time = std::chrono::steady_clock::now();
        CURLcode res = curl_easy_perform(curl);
        auto end_time = std::chrono::steady_clock::now();
        
        long response_code;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
        curl_easy_cleanup(curl);
        
        file.close();
        
        if (res != CURLE_OK) {
            result.error_message = curl_easy_strerror(res);
            fs::remove(local_path);
            return result;
        }
        
        if (response_code != 200) {
            result.error_message = "HTTP error: " + std::to_string(response_code);
            fs::remove(local_path);
            return result;
        }
        
        result.success = true;
        result.local_path = local_path;
        result.downloaded_bytes = ctx.downloaded_size;
        result.download_time = std::chrono::duration<double>(end_time - start_time).count();
        
        return result;
    }
    
    /**
     * @brief Parse model name
     */
    core::ModelPath parseModelName(const std::string& model_name) {
        core::ModelPath model_path;
        model_path.parseFromString(model_name);
        return model_path;
    }
    
    /**
     * @brief Fetch model manifest
     */
    core::ModelManifest fetchModelManifest(const core::ModelPath& model_path) {
#ifdef DUOROU_NO_JSON
        throw std::runtime_error("nlohmann/json is not available; cannot parse manifest.");
#else
        std::string url = base_url_ + "/v2/" + model_path.repository + "/manifests/" + model_path.tag;
        
        HttpResponse response = httpGet(url);
        if (response.response_code != 200) {
            throw std::runtime_error("Failed to fetch manifest: HTTP " + std::to_string(response.response_code));
        }
        
        try {
            json manifest_json = json::parse(response.data, nullptr, /*allow_exceptions=*/false);
            if (manifest_json.is_discarded()) {
                throw std::runtime_error("Failed to parse manifest JSON: discarded");
            }
             core::ModelManifest manifest;
             
             manifest.schema_version = manifest_json.value("schemaVersion", 2);
             manifest.media_type = manifest_json.value("mediaType", "");
             
             if (manifest_json.contains("config")) {
                 auto config = manifest_json["config"];
                 manifest.config.media_type = config.value("mediaType", "");
                 manifest.config.digest = config.value("digest", "");
                 manifest.config.size = config.value("size", 0);
             }
             
             if (manifest_json.contains("layers")) {
                 for (const auto& layer_json : manifest_json["layers"]) {
                     core::ModelLayer layer;
                     layer.media_type = layer_json.value("mediaType", "");
                     layer.digest = layer_json.value("digest", "");
                     layer.size = layer_json.value("size", 0);
                     manifest.layers.push_back(layer);
                 }
             }
             
             return manifest;
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Failed to parse manifest JSON: ") + e.what());
        }
#endif
    }
    
    /**
     * @brief Download blob
     */
    DownloadResult downloadBlob(const core::ModelPath& model_path, const std::string& digest, DownloadContext& ctx) {
        std::string url = base_url_ + "/v2/" + model_path.repository + "/blobs/" + digest;
        std::string local_path = path_manager_->getBlobFilePath(digest);
        
        // Check if blob already exists
        if (path_manager_->blobExists(digest)) {
            DownloadResult result;
            result.success = true;
            result.local_path = local_path;
            result.downloaded_bytes = path_manager_->getBlobSize(digest);
            return result;
        }
        
        return downloadFile(url, local_path, ctx);
    }
};

// ModelDownloader实现

ModelDownloader::ModelDownloader(const std::string& base_url, const std::string& model_dir)
    : pImpl_(std::make_unique<Impl>(base_url, model_dir)) {
}

ModelDownloader::~ModelDownloader() = default;

void ModelDownloader::setProgressCallback(DownloadProgressCallback callback) {
    pImpl_->progress_callback_ = callback;
}

std::future<DownloadResult> ModelDownloader::downloadModel(const std::string& model_name) {
    return std::async(std::launch::async, [this, model_name]() {
        return downloadModelSync(model_name);
    });
}

DownloadResult ModelDownloader::downloadModelSync(const std::string& model_name) {
    DownloadResult result;
    
    try {
        // 解析模型名称
        ModelPath model_path = pImpl_->parseModelName(model_name);
        
        // 获取模型清单
        ModelManifest manifest = pImpl_->fetchModelManifest(model_path);
        
        // 保存清单
        pImpl_->path_manager_->writeManifest(model_path, manifest);
        
        // 下载配置blob
        if (!manifest.config.digest.empty()) {
            DownloadContext ctx;
            ctx.callback = pImpl_->progress_callback_;
            
            DownloadResult config_result = pImpl_->downloadBlob(model_path, manifest.config.digest, ctx);
            if (!config_result.success) {
                result.error_message = "Failed to download config: " + config_result.error_message;
                return result;
            }
        }
        
        // 下载所有层
        for (const auto& layer : manifest.layers) {
            DownloadContext ctx;
            if (pImpl_->progress_callback_) {
                ctx.callback = [this](size_t downloaded, size_t total, double speed) {
                    // 计算总体进度
                    if (pImpl_->progress_callback_) {
                        pImpl_->progress_callback_(downloaded, total, speed);
                    }
                };
            }
            
            DownloadResult layer_result = pImpl_->downloadBlob(model_path, layer.digest, ctx);
            if (!layer_result.success) {
                result.error_message = "Failed to download layer " + layer.digest + ": " + layer_result.error_message;
                return result;
            }
            
            result.downloaded_bytes += layer_result.downloaded_bytes;
        }
        
        result.success = true;
        result.local_path = pImpl_->path_manager_->getManifestFilePath(model_path);
        
    } catch (const std::exception& e) {
        result.error_message = e.what();
    }
    
    return result;
}

ModelInfo ModelDownloader::getModelInfo(const std::string& model_name) {
    ModelInfo info;
    
    try {
        core::ModelPath model_path = pImpl_->parseModelName(model_name);
        info.name = model_path.repository;
        info.tag = model_path.tag;
        
        // 如果本地有清单，读取信息
        if (isModelDownloaded(model_name)) {
            core::ModelManifest manifest;
            pImpl_->path_manager_->readManifest(model_path, manifest);
            
            // 计算总大小
            info.size = manifest.config.size;
            for (const auto& layer : manifest.layers) {
                info.size += layer.size;
            }
            
            info.digest = manifest.config.digest;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error getting model info: " << e.what() << std::endl;
    }
    
    return info;
}

bool ModelDownloader::isModelDownloaded(const std::string& model_name) {
    try {
        core::ModelPath model_path = pImpl_->parseModelName(model_name);
        return fs::exists(pImpl_->path_manager_->getManifestFilePath(model_path));
    } catch (const std::exception&) {
        return false;
    }
}

bool ModelDownloader::isOllamaModel(const std::string& model_name) {
    // 检查模型名称是否符合Ollama模型的命名规范
    // Ollama模型通常格式为: model_name[:tag] 或 namespace/model_name[:tag]
    
    // 简单的启发式检查：
    // 1. 包含冒号的可能是Ollama模型（如 llama2:7b, qwen2.5:latest）
    // 2. 包含斜杠的可能是命名空间模型（如 microsoft/phi-3）
    // 3. 常见的Ollama模型名称模式
    
    std::regex ollama_pattern(R"(^[a-zA-Z0-9._-]+(/[a-zA-Z0-9._-]+)?(:[a-zA-Z0-9._-]+)?$)");
    
    if (!std::regex_match(model_name, ollama_pattern)) {
        return false;
    }
    
    // 检查是否为已知的Ollama模型前缀
    std::vector<std::string> ollama_prefixes = {
        "llama", "qwen", "gemma", "mistral", "phi", "codellama", 
        "vicuna", "alpaca", "orca", "wizard", "dolphin", "neural",
        "tinyllama", "deepseek", "yi", "baichuan", "chatglm"
    };
    
    std::string model_base = model_name;
    // 移除标签部分（冒号后的内容）
    size_t colon_pos = model_base.find(':');
    if (colon_pos != std::string::npos) {
        model_base = model_base.substr(0, colon_pos);
    }
    
    // 移除命名空间部分（斜杠前的内容）
    size_t slash_pos = model_base.find('/');
    if (slash_pos != std::string::npos) {
        model_base = model_base.substr(slash_pos + 1);
    }
    
    // 转换为小写进行比较
    std::transform(model_base.begin(), model_base.end(), model_base.begin(), ::tolower);
    
    // 检查是否匹配已知前缀
    for (const auto& prefix : ollama_prefixes) {
        if (model_base.find(prefix) == 0) {
            return true;
        }
    }
    
    // 如果模型已经在本地下载，也认为是Ollama模型
    return isModelDownloaded(model_name);
}

std::vector<std::string> ModelDownloader::getLocalModels() {
    auto manifests = pImpl_->path_manager_->enumerateManifests();
    std::vector<std::string> model_names;

    // 过滤规则：仅保留文本模型，排除包含 "vl" 或明显视觉/多模态标记的仓库名
    auto is_vision_like = [](const std::string &repository) {
        std::string s = repository;
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);

        // 视觉/多模态模型常见关键词与约定
        const std::vector<std::string> keywords = {
            // 通用标识
            "-vl", "vl", "vision", "multimodal",
            // 常见模型族
            "llava", "bakllava", "glm-4v", "4v", "phi-3-vision",
            "moondream", "minicpm", "cogvlm"
        };

        for (const auto &k : keywords) {
            if (s.find(k) != std::string::npos) return true;
        }
        return false;
    };

    for (const auto &pair : manifests) {
        // pair.first 形如: registry.ollama.ai/library/qwen3:8b 或 registry.ollama.ai/library/qwen3-vl:8b
        core::ModelPath path;
        if (!path.parseFromString(pair.first)) {
            // 不可解析则原样加入（保守）
            model_names.push_back(pair.first);
            continue;
        }

        if (is_vision_like(path.repository)) {
            // 跳过视觉/多模态模型
            continue;
        }

        model_names.push_back(pair.first);
    }

    // 排序并去重，保证稳定输出
    std::sort(model_names.begin(), model_names.end());
    model_names.erase(std::unique(model_names.begin(), model_names.end()), model_names.end());
    return model_names;
}

bool ModelDownloader::deleteModel(const std::string& model_name) {
    try {
        core::ModelPath model_path = pImpl_->parseModelName(model_name);
        std::string manifest_path = pImpl_->path_manager_->getManifestFilePath(model_path);
        
        if (fs::exists(manifest_path)) {
            fs::remove(manifest_path);
            return true;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error deleting model: " << e.what() << std::endl;
    }
    
    return false;
}

std::string ModelDownloader::getModelPath(const std::string& model_name) {
    try {
        core::ModelPath model_path = pImpl_->parseModelName(model_name);
        return pImpl_->path_manager_->getManifestFilePath(model_path);
    } catch (const std::exception&) {
        return "";
    }
}

bool ModelDownloader::verifyModel(const std::string& model_name) {
    try {
        core::ModelPath model_path = pImpl_->parseModelName(model_name);
        core::ModelManifest manifest;
        if (!pImpl_->path_manager_->readManifest(model_path, manifest)) {
            return false;
        }
        
        // 验证配置blob
        if (!manifest.config.digest.empty()) {
            if (!pImpl_->path_manager_->verifyBlob(manifest.config.digest)) {
                return false;
            }
        }
        
        // 验证所有层
        for (const auto& layer : manifest.layers) {
            if (!pImpl_->path_manager_->verifyBlob(layer.digest)) {
                return false;
            }
        }
        
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

size_t ModelDownloader::cleanupUnusedBlobs() {
    return pImpl_->path_manager_->pruneLayers();
}

size_t ModelDownloader::getCacheSize() {
    size_t total_size = 0;
    
    try {
        std::string blobs_dir = pImpl_->model_dir_ + "/blobs";
        if (fs::exists(blobs_dir)) {
            for (const auto& entry : fs::recursive_directory_iterator(blobs_dir)) {
                if (entry.is_regular_file()) {
                    total_size += entry.file_size();
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error calculating cache size: " << e.what() << std::endl;
    }
    
    return total_size;
}

void ModelDownloader::setMaxCacheSize(size_t max_size) {
    pImpl_->max_cache_size_ = max_size;
}

void ModelDownloader::setModelDirectory(const std::string& model_dir) {
    // 展开 ~ 并更新内部路径
    std::string expanded = pImpl_->expandPath(model_dir);
    pImpl_->model_dir_ = expanded;

    // 确保目录存在
    try {
        fs::create_directories(expanded);
    } catch (const std::exception& e) {
        std::cerr << "Error creating model directory: " << e.what() << std::endl;
    }

    // 更新并重新初始化路径管理器
    if (!pImpl_->path_manager_) {
        pImpl_->path_manager_ = std::make_unique<ModelPathManager>(expanded);
    } else {
        pImpl_->path_manager_->setBasePath(expanded);
    }

    // 重新初始化以确保新的目录结构可用
    pImpl_->path_manager_->initialize();
}

// ModelDownloaderFactory实现

std::unique_ptr<ModelDownloader> ModelDownloaderFactory::create(const std::string& base_url, const std::string& model_dir) {
    return std::make_unique<ModelDownloader>(base_url, model_dir);
}

} // namespace duorou