#include "ggml_wrapper.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstring>

namespace duorou {
namespace extensions {
namespace ollama {
namespace ggml {

// TensorInfo实现
bool TensorInfo::valid() const {
    return !name.empty() && numBytes() > 0;
}

int64_t TensorInfo::numValues() const {
    int64_t numItems = 1;
    for (uint64_t dim : shape) {
        numItems *= static_cast<int64_t>(dim);
    }
    return numItems;
}

int64_t TensorInfo::numBytes() const {
    // 简化的字节数计算，实际应该根据type计算
    return numValues() * 4; // 假设每个值4字节
}

// KVHelper实现
std::string KVHelper::architecture() const {
    return getString("general.architecture", "unknown");
}

std::string KVHelper::kind() const {
    return getString("general.type", "unknown");
}

uint64_t KVHelper::parameterCount() const {
    return getUint64("general.parameter_count", 0);
}

FileType KVHelper::fileType() const {
    uint32_t t = getUint("general.file_type", 0);
    if (t > 0) {
        return static_cast<FileType>(t);
    }
    return FileType::UNKNOWN;
}

uint64_t KVHelper::blockCount() const {
    return getUint64("block_count", 0);
}

uint64_t KVHelper::embeddingLength() const {
    return getUint64("embedding_length", 0);
}

uint64_t KVHelper::headCountMax() const {
    return static_cast<uint64_t>(getUintOrMaxArrayValue("attention.head_count", 1));
}

uint64_t KVHelper::headCountMin() const {
    return static_cast<uint64_t>(getUintOrMinArrayValue("attention.head_count", 1));
}

uint64_t KVHelper::headCountKVMax() const {
    return static_cast<uint64_t>(getUintOrMaxArrayValue("attention.head_count_kv", 1));
}

uint64_t KVHelper::headCountKVMin() const {
    return static_cast<uint64_t>(getUintOrMinArrayValue("attention.head_count_kv", 1));
}

uint64_t KVHelper::embeddingHeadCountMax() const {
    uint64_t heads = headCountMin();
    if (heads > 0) {
        return embeddingLength() / heads;
    }
    return 0;
}

uint64_t KVHelper::embeddingHeadCountK() const {
    return static_cast<uint64_t>(getUint("attention.key_length", static_cast<uint32_t>(embeddingHeadCountMax())));
}

uint64_t KVHelper::embeddingHeadCountV() const {
    return static_cast<uint64_t>(getUint("attention.value_length", static_cast<uint32_t>(embeddingHeadCountMax())));
}

uint64_t KVHelper::contextLength() const {
    return getUint64("context_length", 0);
}

std::string KVHelper::chatTemplate() const {
    return getString("tokenizer.chat_template", "");
}

std::string KVHelper::getString(const std::string& key, const std::string& defaultValue) const {
    auto it = kv_.find(key);
    if (it != kv_.end()) {
        if (std::holds_alternative<std::string>(it->second)) {
            return std::get<std::string>(it->second);
        }
    }
    return defaultValue;
}

uint32_t KVHelper::getUint(const std::string& key, uint32_t defaultValue) const {
    auto it = kv_.find(key);
    if (it != kv_.end()) {
        if (std::holds_alternative<uint64_t>(it->second)) {
            return static_cast<uint32_t>(std::get<uint64_t>(it->second));
        }
        if (std::holds_alternative<int64_t>(it->second)) {
            return static_cast<uint32_t>(std::get<int64_t>(it->second));
        }
    }
    return defaultValue;
}

uint64_t KVHelper::getUint64(const std::string& key, uint64_t defaultValue) const {
    auto it = kv_.find(key);
    if (it != kv_.end()) {
        if (std::holds_alternative<uint64_t>(it->second)) {
            return std::get<uint64_t>(it->second);
        }
        if (std::holds_alternative<int64_t>(it->second)) {
            return static_cast<uint64_t>(std::get<int64_t>(it->second));
        }
    }
    return defaultValue;
}

int64_t KVHelper::getInt64(const std::string& key, int64_t defaultValue) const {
    auto it = kv_.find(key);
    if (it != kv_.end()) {
        if (std::holds_alternative<int64_t>(it->second)) {
            return std::get<int64_t>(it->second);
        }
        if (std::holds_alternative<uint64_t>(it->second)) {
            return static_cast<int64_t>(std::get<uint64_t>(it->second));
        }
    }
    return defaultValue;
}

double KVHelper::getFloat(const std::string& key, double defaultValue) const {
    auto it = kv_.find(key);
    if (it != kv_.end()) {
        if (std::holds_alternative<double>(it->second)) {
            return std::get<double>(it->second);
        }
    }
    return defaultValue;
}

bool KVHelper::getBool(const std::string& key, bool defaultValue) const {
    auto it = kv_.find(key);
    if (it != kv_.end()) {
        if (std::holds_alternative<bool>(it->second)) {
            return std::get<bool>(it->second);
        }
    }
    return defaultValue;
}

std::vector<std::string> KVHelper::getStrings(const std::string& key) const {
    auto it = kv_.find(key);
    if (it != kv_.end()) {
        if (std::holds_alternative<std::vector<std::string>>(it->second)) {
            return std::get<std::vector<std::string>>(it->second);
        }
    }
    return {};
}

std::vector<int64_t> KVHelper::getInts(const std::string& key) const {
    auto it = kv_.find(key);
    if (it != kv_.end()) {
        if (std::holds_alternative<std::vector<int64_t>>(it->second)) {
            return std::get<std::vector<int64_t>>(it->second);
        }
    }
    return {};
}

std::vector<uint64_t> KVHelper::getUints(const std::string& key) const {
    auto it = kv_.find(key);
    if (it != kv_.end()) {
        if (std::holds_alternative<std::vector<uint64_t>>(it->second)) {
            return std::get<std::vector<uint64_t>>(it->second);
        }
    }
    return {};
}

std::vector<double> KVHelper::getFloats(const std::string& key) const {
    auto it = kv_.find(key);
    if (it != kv_.end()) {
        if (std::holds_alternative<std::vector<double>>(it->second)) {
            return std::get<std::vector<double>>(it->second);
        }
    }
    return {};
}

std::vector<bool> KVHelper::getBools(const std::string& key) const {
    auto it = kv_.find(key);
    if (it != kv_.end()) {
        if (std::holds_alternative<std::vector<bool>>(it->second)) {
            return std::get<std::vector<bool>>(it->second);
        }
    }
    return {};
}

uint32_t KVHelper::getUintOrMaxArrayValue(const std::string& key, uint32_t defaultValue) const {
    // 首先尝试获取单个值
    uint32_t singleValue = getUint(key, 0);
    if (singleValue > 0) {
        return singleValue;
    }
    
    // 然后尝试获取数组的最大值
    std::vector<uint64_t> values = getUints(key);
    if (!values.empty()) {
        auto maxIt = std::max_element(values.begin(), values.end());
        return static_cast<uint32_t>(*maxIt);
    }
    
    return defaultValue;
}

uint32_t KVHelper::getUintOrMinArrayValue(const std::string& key, uint32_t defaultValue) const {
    // 首先尝试获取单个值
    uint32_t singleValue = getUint(key, 0);
    if (singleValue > 0) {
        return singleValue;
    }
    
    // 然后尝试获取数组的最小值
    std::vector<uint64_t> values = getUints(key);
    if (!values.empty()) {
        auto minIt = std::min_element(values.begin(), values.end());
        return static_cast<uint32_t>(*minIt);
    }
    
    return defaultValue;
}

// 简单的模型实现
class SimpleModel : public Model {
public:
    SimpleModel(KV kv, Tensors tensors) : kv_(std::move(kv)), tensors_(std::move(tensors)) {}
    
    const KV& getKV() const override { return kv_; }
    const Tensors& getTensors() const override { return tensors_; }

private:
    KV kv_;
    Tensors tensors_;
};

// GGML::Impl实现
class GGML::Impl {
public:
    std::unique_ptr<Model> model_;
    int64_t length_ = 0;
    bool loaded_ = false;
    
    bool load(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << path << std::endl;
            return false;
        }
        
        // 获取文件大小
        file.seekg(0, std::ios::end);
        length_ = file.tellg();
        file.seekg(0, std::ios::beg);
        
        // 这里应该实现实际的GGML文件解析
        // 目前只是一个占位符实现
        KV kv;
        kv["general.architecture"] = std::string("llama");
        kv["general.parameter_count"] = uint64_t(7000000000);
        kv["general.file_type"] = uint64_t(static_cast<uint32_t>(FileType::Q4_K_M));
        
        Tensors tensors;
        // 添加一些示例张量
        TensorInfo tensor;
        tensor.name = "token_embd.weight";
        tensor.shape = {32000, 4096};
        tensor.type = static_cast<uint32_t>(FileType::F16);
        tensor.offset = 0;
        tensors.push_back(tensor);
        
        model_ = std::make_unique<SimpleModel>(std::move(kv), std::move(tensors));
        loaded_ = true;
        
        return true;
    }
    
    void unload() {
        model_.reset();
        length_ = 0;
        loaded_ = false;
    }
};

// GGML实现
GGML::GGML() : impl_(std::make_unique<Impl>()) {}

GGML::~GGML() = default;

GGML::GGML(GGML&& other) noexcept : impl_(std::move(other.impl_)) {}

GGML& GGML::operator=(GGML&& other) noexcept {
    if (this != &other) {
        impl_ = std::move(other.impl_);
    }
    return *this;
}

bool GGML::load(const std::string& path) {
    return impl_->load(path);
}

const Model* GGML::getModel() const {
    return impl_->model_.get();
}

KVHelper GGML::getKVHelper() const {
    if (impl_->model_) {
        return KVHelper(impl_->model_->getKV());
    }
    return KVHelper(KV{});
}

int64_t GGML::getLength() const {
    return impl_->length_;
}

bool GGML::isLoaded() const {
    return impl_->loaded_;
}

void GGML::unload() {
    impl_->unload();
}

// 文件类型解析和转换
FileType parseFileType(const std::string& str) {
    if (str == "F32") return FileType::F32;
    if (str == "F16") return FileType::F16;
    if (str == "Q8_0") return FileType::Q8_0;
    if (str == "Q4_K_S") return FileType::Q4_K_S;
    if (str == "Q4_K_M" || str == "Q4_K") return FileType::Q4_K_M;
    if (str == "BF16") return FileType::BF16;
    return FileType::UNKNOWN;
}

std::string fileTypeToString(FileType type) {
    switch (type) {
        case FileType::F32: return "F32";
        case FileType::F16: return "F16";
        case FileType::Q4_0: return "Q4_0";
        case FileType::Q4_1: return "Q4_1";
        case FileType::Q8_0: return "Q8_0";
        case FileType::Q5_0: return "Q5_0";
        case FileType::Q5_1: return "Q5_1";
        case FileType::Q2_K: return "Q2_K";
        case FileType::Q3_K_S: return "Q3_K_S";
        case FileType::Q3_K_M: return "Q3_K_M";
        case FileType::Q3_K_L: return "Q3_K_L";
        case FileType::Q4_K_S: return "Q4_K_S";
        case FileType::Q4_K_M: return "Q4_K_M";
        case FileType::Q5_K_S: return "Q5_K_S";
        case FileType::Q5_K_M: return "Q5_K_M";
        case FileType::Q6_K: return "Q6_K";
        case FileType::BF16: return "BF16";
        default: return "UNKNOWN";
    }
}

} // namespace ggml
} // namespace ollama
} // namespace extensions
} // namespace duorou