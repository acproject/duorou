#include "gguf_modifier.h"
#include "arch_mapping.h"
#include "../../../third_party/llama.cpp/include/llama.h"
#include "../../../third_party/llama.cpp/ggml/include/gguf.h"

#include <fstream>
#include <iostream>
#include <cstring>
#include <filesystem>

bool GGUFModifier::modifyArchitectureIfNeeded(const std::string& gguf_path) {
    // First check if modification is needed
    if (!needsArchitectureModification(gguf_path)) {
        return true;  // No modification needed
    }
    
    std::string current_arch = getGGUFArchitecture(gguf_path);
    if (current_arch.empty()) {
        std::cerr << "Failed to read architecture from GGUF file: " << gguf_path << std::endl;
        return false;
    }
    
    std::string mapped_arch = ArchMapping::getMappedArchitecture(current_arch);
    if (mapped_arch == current_arch) {
        return true;  // No mapping needed
    }
    
    std::cout << "Mapping architecture '" << current_arch << "' to '" << mapped_arch << "'" << std::endl;
    
    return modifyArchitectureField(gguf_path, mapped_arch);
}

bool GGUFModifier::createModifiedGGUF(const std::string& original_path, const std::string& temp_path) {
    // Copy original file to temporary location
    try {
        std::filesystem::copy_file(original_path, temp_path, std::filesystem::copy_options::overwrite_existing);
    } catch (const std::exception& e) {
        std::cerr << "Failed to copy GGUF file: " << e.what() << std::endl;
        return false;
    }
    
    // Modify the temporary file
    return modifyArchitectureIfNeeded(temp_path);
}

bool GGUFModifier::needsArchitectureModification(const std::string& gguf_path) {
    std::string arch = getGGUFArchitecture(gguf_path);
    return !arch.empty() && ArchMapping::needsMapping(arch);
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