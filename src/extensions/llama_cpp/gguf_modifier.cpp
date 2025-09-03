#include "gguf_modifier.h"
#include "ggml_incremental_extension.h"
#include "model_config_manager.h"
#include "vision_model_handler.h"
#include "attention_handler.h"
#include "../../../third_party/llama.cpp/include/llama.h"
#include "../../../third_party/llama.cpp/ggml/include/gguf.h"

#include <fstream>
#include <iostream>
#include <cstring>
#include <filesystem>
#include <algorithm>

bool GGUFModifier::modifyArchitectureIfNeeded(const std::string& gguf_path) {
    std::string current_arch = getGGUFArchitecture(gguf_path);
    if (current_arch.empty()) {
        std::cerr << "Failed to read architecture from GGUF file: " << gguf_path << std::endl;
        return false;
    }
    
    bool success = true;
    
    // Check if architecture mapping is needed
    if (needsArchitectureModification(gguf_path)) {
        std::string mapped_arch = duorou::extensions::GGMLIncrementalExtension::getBaseArchitecture(current_arch);
        if (mapped_arch != current_arch) {
            std::cout << "Mapping architecture '" << current_arch << "' to '" << mapped_arch << "'" << std::endl;
            success &= modifyArchitectureField(gguf_path, mapped_arch);
            current_arch = mapped_arch; // Update for subsequent operations
        }
    }
    
    // Perform model-specific modifications
    success &= performModelSpecificModifications(gguf_path, current_arch);
    
    return success;
}

// createModifiedGGUF function removed - no longer needed
// Model architecture mapping is now handled through kv_override mechanism

bool GGUFModifier::needsArchitectureModification(const std::string& gguf_path) {
    std::string arch = getGGUFArchitecture(gguf_path);
    return !arch.empty() && duorou::extensions::GGMLIncrementalExtension::isArchitectureSupported(arch);
}

std::string GGUFModifier::getGGUFArchitecture(const std::string& gguf_path) {
    // Use llama.cpp's GGUF reading functionality
    struct gguf_init_params params = {false, nullptr};
    struct gguf_context* ctx = gguf_init_from_file(gguf_path.c_str(), params);
    if (!ctx) {
        return "";
    }
    
    // Find the architecture key
    int key_idx = gguf_find_key(ctx, "general.architecture");
    if (key_idx < 0) {
        gguf_free(ctx);
        return "";
    }
    
    // Get the architecture value
    const char* arch_str = gguf_get_val_str(ctx, key_idx);
    std::string result = arch_str ? arch_str : "";
    
    gguf_free(ctx);
    return result;
}

bool GGUFModifier::modifyArchitectureField(const std::string& gguf_path, const std::string& new_arch) {
    std::string current_arch = getGGUFArchitecture(gguf_path);
    if (current_arch.empty()) {
        std::cerr << "Failed to read current architecture from GGUF file" << std::endl;
        return false;
    }
    
    // If the new architecture is longer than the current one, we can't do in-place replacement
    if (new_arch.length() > current_arch.length()) {
        std::cerr << "New architecture name is longer than current one, cannot modify in-place" << std::endl;
        return false;
    }
    
    // Read the entire file
    std::ifstream file(gguf_path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open GGUF file for reading: " << gguf_path << std::endl;
        return false;
    }
    
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> buffer(file_size);
    file.read(buffer.data(), file_size);
    file.close();
    
    // Search for the architecture string in the GGUF metadata section
    // GGUF strings are stored with a length prefix (uint64_t) followed by the string data
    bool found = false;
    size_t search_limit = std::min(file_size, static_cast<size_t>(1024 * 1024)); // Search only first 1MB
    
    for (size_t i = 0; i < search_limit - current_arch.length() - 8; ++i) {
        // Check if we found the architecture string
        if (std::memcmp(buffer.data() + i, current_arch.c_str(), current_arch.length()) == 0) {
            // Verify this is likely a GGUF string by checking the length prefix
            if (i >= 8) {
                uint64_t* len_ptr = reinterpret_cast<uint64_t*>(buffer.data() + i - 8);
                if (*len_ptr == current_arch.length()) {
                    // Replace the architecture string
                    std::memset(buffer.data() + i, 0, current_arch.length());
                    std::memcpy(buffer.data() + i, new_arch.c_str(), new_arch.length());
                    
                    // Update the length prefix if the new string is shorter
                    *len_ptr = new_arch.length();
                    
                    found = true;
                    break;
                }
            }
        }
    }
    
    if (!found) {
        std::cerr << "Failed to find architecture string in GGUF metadata" << std::endl;
        return false;
    }
    
    // Write the modified buffer back to file
    std::ofstream out_file(gguf_path, std::ios::binary);
    if (!out_file) {
        std::cerr << "Failed to open GGUF file for writing: " << gguf_path << std::endl;
        return false;
    }
    
    out_file.write(buffer.data(), buffer.size());
    out_file.close();
    
    std::cout << "Successfully modified architecture in GGUF file" << std::endl;
    return true;
}

bool GGUFModifier::addMissingQwen25VLKeys(const std::string& gguf_path) {
    std::string arch = getGGUFArchitecture(gguf_path);
    if (arch != "qwen25vl" && arch != "qwen2vl") {
        return true;  // Not a qwen2.5vl model, no modification needed
    }
    
    // Check if dimension_sections key is missing
    if (!hasKey(gguf_path, "qwen2vl.rope.dimension_sections")) {
        std::cout << "Missing qwen2vl.rope.dimension_sections key detected" << std::endl;
        
        // Create a temporary modified GGUF file with the missing keys
        std::string temp_path = gguf_path + ".temp_modified";
        
        try {
            // Copy original file to temporary location
            std::filesystem::copy_file(gguf_path, temp_path, std::filesystem::copy_options::overwrite_existing);
            
            // Use GGUF writer to add missing keys
            struct gguf_init_params params = {false, nullptr};
            struct gguf_context* ctx = gguf_init_from_file(temp_path.c_str(), params);
            if (!ctx) {
                std::cerr << "Failed to open GGUF file for modification" << std::endl;
                std::filesystem::remove(temp_path);
                return false;
            }
            
            // 检查是否存在qwen25vl.rope.mrope_section，如果存在则读取并映射到qwen2vl.rope.dimension_sections
            int mrope_index = gguf_find_key(ctx, "qwen25vl.rope.mrope_section");
            std::vector<uint32_t> dimension_sections;
            
            if (mrope_index >= 0) {
                // 读取原始的mrope_section数组
                enum gguf_type mrope_type = gguf_get_kv_type(ctx, mrope_index);
                if (mrope_type == GGUF_TYPE_ARRAY) {
                    enum gguf_type arr_type = gguf_get_arr_type(ctx, mrope_index);
                    uint64_t arr_size = gguf_get_arr_n(ctx, mrope_index);
                    
                    if (arr_type == GGUF_TYPE_INT32 && arr_size == 3) {
                        const int32_t* mrope_data = (const int32_t*)gguf_get_arr_data(ctx, mrope_index);
                        if (mrope_data != nullptr) {
                            // 将int32转换为uint32
                            for (uint64_t i = 0; i < arr_size; i++) {
                                dimension_sections.push_back(static_cast<uint32_t>(mrope_data[i]));
                            }
                            std::cout << "Mapped mrope_section [" << mrope_data[0] << ", " << mrope_data[1] << ", " << mrope_data[2] << "] to dimension_sections" << std::endl;
                        }
                    }
                }
            }
            
            // 如果没有找到mrope_section或读取失败，使用默认值
            if (dimension_sections.empty()) {
                dimension_sections = {64, 64, 64};  // qwen2vl的默认值
                std::cout << "Using default dimension_sections [64, 64, 64]" << std::endl;
            }
            
            // 只有当键不存在时才添加dimension_sections数组
            if (!hasKey(gguf_path, "qwen2vl.rope.dimension_sections")) {
                gguf_set_arr_data(ctx, "qwen2vl.rope.dimension_sections", GGUF_TYPE_UINT32, 
                                 dimension_sections.data(), dimension_sections.size());
                std::cout << "Added qwen2vl.rope.dimension_sections" << std::endl;
            }
            
            // Add layer_norm_rms_epsilon if missing
            if (!hasKey(gguf_path, "qwen2vl.attention.layer_norm_rms_epsilon")) {
                float layer_norm_eps = 1e-6f;
                gguf_set_val_f32(ctx, "qwen2vl.attention.layer_norm_rms_epsilon", layer_norm_eps);
            }
            
            // Write the modified GGUF file
            if (!gguf_write_to_file(ctx, gguf_path.c_str(), false)) {
                std::cerr << "Failed to write modified GGUF file" << std::endl;
                gguf_free(ctx);
                return false;
            }
            
            gguf_free(ctx);
            
            std::cout << "Successfully added missing keys to GGUF file" << std::endl;
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Error modifying GGUF file: " << e.what() << std::endl;
            if (std::filesystem::exists(temp_path)) {
                std::filesystem::remove(temp_path);
            }
            return false;
        }
    }
    
    return true;  // No modification needed
}

bool GGUFModifier::hasKey(const std::string& gguf_path, const std::string& key_name) {
    struct gguf_init_params params = {false, nullptr};
    struct gguf_context* ctx = gguf_init_from_file(gguf_path.c_str(), params);
    if (!ctx) {
        return false;
    }
    
    int key_idx = gguf_find_key(ctx, key_name.c_str());
    bool exists = (key_idx >= 0);
    
    gguf_free(ctx);
    return exists;
}

bool GGUFModifier::addMissingGemma3Keys(const std::string& gguf_path) {
    std::string arch = getGGUFArchitecture(gguf_path);
    if (arch != "gemma3") {
        return true; // Not a Gemma3 model
    }
    
    bool modified = false;
    
    // Add attention logit softcapping
    if (!hasKey(gguf_path, "gemma3.attention.logit_softcap")) {
        setFloatValue(gguf_path, "gemma3.attention.logit_softcap", 50.0f);
        modified = true;
    }
    
    // Add final logit softcapping
    if (!hasKey(gguf_path, "gemma3.final_logit_softcap")) {
        setFloatValue(gguf_path, "gemma3.final_logit_softcap", 30.0f);
        modified = true;
    }
    
    // Add sliding window size
    if (!hasKey(gguf_path, "gemma3.attention.sliding_window")) {
        setIntValue(gguf_path, "gemma3.attention.sliding_window", 4096);
        modified = true;
    }
    
    if (modified) {
        std::cout << "Added missing Gemma3 keys to GGUF file" << std::endl;
    }
    
    return true;
}

bool GGUFModifier::addMissingMistral3Keys(const std::string& gguf_path) {
    std::string arch = getGGUFArchitecture(gguf_path);
    if (arch != "mistral3") {
        return true; // Not a Mistral3 model
    }
    
    bool modified = false;
    
    // Add sliding window size
    if (!hasKey(gguf_path, "mistral3.attention.sliding_window")) {
        setIntValue(gguf_path, "mistral3.attention.sliding_window", 131072);
        modified = true;
    }
    
    // Add RoPE base
    if (!hasKey(gguf_path, "mistral3.rope.freq_base")) {
        setFloatValue(gguf_path, "mistral3.rope.freq_base", 1000000.0f);
        modified = true;
    }
    
    if (modified) {
        std::cout << "Added missing Mistral3 keys to GGUF file" << std::endl;
    }
    
    return true;
}

bool GGUFModifier::addMissingGptossKeys(const std::string& gguf_path) {
    std::string arch = getGGUFArchitecture(gguf_path);
    if (arch != "gptoss" && arch != "gpt-oss") {
        return true; // Not a GPT-OSS model
    }
    
    bool modified = false;
    
    // Add RoPE type
    if (!hasKey(gguf_path, "gptoss.rope.type")) {
        setStringValue(gguf_path, "gptoss.rope.type", "neox");
        modified = true;
    }
    
    // Add attention mechanism
    if (!hasKey(gguf_path, "gptoss.attention.type")) {
        setStringValue(gguf_path, "gptoss.attention.type", "standard");
        modified = true;
    }
    
    if (modified) {
        std::cout << "Added missing GPT-OSS keys to GGUF file" << std::endl;
    }
    
    return true;
}

bool GGUFModifier::addVisionMetadata(const std::string& gguf_path, const std::string& architecture) {
    if (!VisionModelHandler::hasVisionSupport(architecture)) {
        return true; // Not a vision model
    }
    
    auto visionConfig = VisionModelHandler::getVisionConfig(architecture);
    if (!visionConfig) {
        return true;
    }
    
    bool modified = false;
    
    // Add vision processor metadata
    std::string visionPrefix = architecture + ".vision";
    
    if (!hasKey(gguf_path, visionPrefix + ".image_size")) {
        setIntValue(gguf_path, visionPrefix + ".image_size", visionConfig->imageSize);
        modified = true;
    }
    
    if (!hasKey(gguf_path, visionPrefix + ".patch_size")) {
        setIntValue(gguf_path, visionPrefix + ".patch_size", visionConfig->patchSize);
        modified = true;
    }
    
    if (!hasKey(gguf_path, visionPrefix + ".tokens_per_image")) {
        setIntValue(gguf_path, visionPrefix + ".tokens_per_image", visionConfig->tokensPerImage);
        modified = true;
    }
    
    if (modified) {
        std::cout << "Added vision metadata for " << architecture << " model" << std::endl;
    }
    
    return true;
}

bool GGUFModifier::addAttentionMetadata(const std::string& gguf_path, const std::string& architecture) {
    auto attentionConfig = AttentionHandler::getAttentionConfig(architecture);
    if (!attentionConfig) {
        return true;
    }
    
    bool modified = false;
    std::string attentionPrefix = architecture + ".attention";
    
    // Add sliding window metadata
    if (attentionConfig->hasSlidingWindow && !hasKey(gguf_path, attentionPrefix + ".sliding_window")) {
        setIntValue(gguf_path, attentionPrefix + ".sliding_window", attentionConfig->slidingWindowSize);
        modified = true;
    }
    
    // Add softcapping metadata
    if (attentionConfig->hasSoftcapping) {
        if (!hasKey(gguf_path, attentionPrefix + ".logit_softcap")) {
            setFloatValue(gguf_path, attentionPrefix + ".logit_softcap", attentionConfig->attentionLogitSoftcap);
            modified = true;
        }
        
        if (!hasKey(gguf_path, architecture + ".final_logit_softcap")) {
            setFloatValue(gguf_path, architecture + ".final_logit_softcap", attentionConfig->finalLogitSoftcap);
            modified = true;
        }
    }
    
    // Add RoPE metadata
    std::string ropePrefix = architecture + ".rope";
    if (!hasKey(gguf_path, ropePrefix + ".freq_base")) {
        setFloatValue(gguf_path, ropePrefix + ".freq_base", attentionConfig->ropeBase);
        modified = true;
    }
    
    if (modified) {
        std::cout << "Added attention metadata for " << architecture << " model" << std::endl;
    }
    
    return true;
}

bool GGUFModifier::performModelSpecificModifications(const std::string& gguf_path, const std::string& architecture) {
    bool success = true;
    
    // Add model-specific missing keys
    if (architecture == "qwen25vl" || architecture == "qwen2vl") {
        success &= addMissingQwen25VLKeys(gguf_path);
    } else if (architecture == "gemma3") {
        success &= addMissingGemma3Keys(gguf_path);
    } else if (architecture == "mistral3") {
        success &= addMissingMistral3Keys(gguf_path);
    } else if (architecture == "gptoss" || architecture == "gpt-oss") {
        success &= addMissingGptossKeys(gguf_path);
    }
    
    // Add vision metadata for multimodal models
    success &= addVisionMetadata(gguf_path, architecture);
    
    // Add attention metadata
    success &= addAttentionMetadata(gguf_path, architecture);
    
    return success;
}

std::unordered_map<std::string, std::string> GGUFModifier::getAllMetadata(const std::string& gguf_path) {
    std::unordered_map<std::string, std::string> metadata;
    
    struct gguf_init_params params = {false, nullptr};
    struct gguf_context* ctx = gguf_init_from_file(gguf_path.c_str(), params);
    if (!ctx) {
        return metadata;
    }
    
    int n_kv = gguf_get_n_kv(ctx);
    for (int i = 0; i < n_kv; ++i) {
        const char* key = gguf_get_key(ctx, i);
        enum gguf_type type = gguf_get_kv_type(ctx, i);
        
        if (type == GGUF_TYPE_STRING) {
            const char* value = gguf_get_val_str(ctx, i);
            metadata[key] = value ? value : "";
        }
    }
    
    gguf_free(ctx);
    return metadata;
}

bool GGUFModifier::setStringValue(const std::string& gguf_path, const std::string& key_name, const std::string& value) {
    // This is a simplified implementation - in practice, you'd need to properly modify the GGUF file
    // For now, we'll use a temporary approach
    std::string temp_path = gguf_path + ".temp_string_mod";
    
    try {
        std::filesystem::copy_file(gguf_path, temp_path, std::filesystem::copy_options::overwrite_existing);
        
        struct gguf_init_params params = {false, nullptr};
        struct gguf_context* ctx = gguf_init_from_file(temp_path.c_str(), params);
        if (!ctx) {
            std::filesystem::remove(temp_path);
            return false;
        }
        
        gguf_set_val_str(ctx, key_name.c_str(), value.c_str());
        
        if (!gguf_write_to_file(ctx, temp_path.c_str(), false)) {
            gguf_free(ctx);
            std::filesystem::remove(temp_path);
            return false;
        }
        
        gguf_free(ctx);
        std::filesystem::rename(temp_path, gguf_path);
        return true;
        
    } catch (const std::exception& e) {
        if (std::filesystem::exists(temp_path)) {
            std::filesystem::remove(temp_path);
        }
        return false;
    }
}

bool GGUFModifier::setFloatValue(const std::string& gguf_path, const std::string& key_name, float value) {
    std::string temp_path = gguf_path + ".temp_float_mod";
    
    try {
        std::filesystem::copy_file(gguf_path, temp_path, std::filesystem::copy_options::overwrite_existing);
        
        struct gguf_init_params params = {false, nullptr};
        struct gguf_context* ctx = gguf_init_from_file(temp_path.c_str(), params);
        if (!ctx) {
            std::filesystem::remove(temp_path);
            return false;
        }
        
        gguf_set_val_f32(ctx, key_name.c_str(), value);
        
        if (!gguf_write_to_file(ctx, temp_path.c_str(), false)) {
            gguf_free(ctx);
            std::filesystem::remove(temp_path);
            return false;
        }
        
        gguf_free(ctx);
        std::filesystem::rename(temp_path, gguf_path);
        return true;
        
    } catch (const std::exception& e) {
        if (std::filesystem::exists(temp_path)) {
            std::filesystem::remove(temp_path);
        }
        return false;
    }
}

bool GGUFModifier::setIntValue(const std::string& gguf_path, const std::string& key_name, int value) {
    std::string temp_path = gguf_path + ".temp_int_mod";
    
    try {
        std::filesystem::copy_file(gguf_path, temp_path, std::filesystem::copy_options::overwrite_existing);
        
        struct gguf_init_params params = {false, nullptr};
        struct gguf_context* ctx = gguf_init_from_file(temp_path.c_str(), params);
        if (!ctx) {
            std::filesystem::remove(temp_path);
            return false;
        }
        
        gguf_set_val_u32(ctx, key_name.c_str(), static_cast<uint32_t>(value));
        
        if (!gguf_write_to_file(ctx, temp_path.c_str(), false)) {
            gguf_free(ctx);
            std::filesystem::remove(temp_path);
            return false;
        }
        
        gguf_free(ctx);
        std::filesystem::rename(temp_path, gguf_path);
        return true;
        
    } catch (const std::exception& e) {
        if (std::filesystem::exists(temp_path)) {
            std::filesystem::remove(temp_path);
        }
        return false;
    }
}

bool GGUFModifier::setArrayValue(const std::string& gguf_path, const std::string& key_name, const std::vector<uint32_t>& values) {
    std::string temp_path = gguf_path + ".temp_array_mod";
    
    try {
        std::filesystem::copy_file(gguf_path, temp_path, std::filesystem::copy_options::overwrite_existing);
        
        struct gguf_init_params params = {false, nullptr};
        struct gguf_context* ctx = gguf_init_from_file(temp_path.c_str(), params);
        if (!ctx) {
            std::filesystem::remove(temp_path);
            return false;
        }
        
        gguf_set_arr_data(ctx, key_name.c_str(), GGUF_TYPE_UINT32, values.data(), values.size());
        
        if (!gguf_write_to_file(ctx, temp_path.c_str(), false)) {
            gguf_free(ctx);
            std::filesystem::remove(temp_path);
            return false;
        }
        
        gguf_free(ctx);
        std::filesystem::rename(temp_path, gguf_path);
        return true;
        
    } catch (const std::exception& e) {
        if (std::filesystem::exists(temp_path)) {
            std::filesystem::remove(temp_path);
        }
        return false;
    }
}