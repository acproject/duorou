#include "compatibility_checker.h"
#include "ggml_incremental_extension.h"
#include "model_config_manager.h"
#include "vision_model_handler.h"
#include "attention_handler.h"
#include <algorithm>
#include <sstream>

// Static member initialization
std::unordered_map<std::string, std::shared_ptr<CompatibilityChecker::ModelRequirements>> CompatibilityChecker::modelRequirements;
std::unordered_set<std::string> CompatibilityChecker::llamaCppNativeArchitectures;
std::unordered_set<std::string> CompatibilityChecker::ollamaRequiredArchitectures;
bool CompatibilityChecker::initialized = false;

CompatibilityChecker::CompatibilityResult CompatibilityChecker::checkCompatibility(
    const std::string& architecture,
    const std::string& modelPath
) {
    if (!initialized) {
        initialize();
    }
    
    CompatibilityResult result;
    result.originalArchitecture = architecture;
    
    std::string normalizedArch = normalizeArchitectureName(architecture);
    auto requirements = getModelRequirements(normalizedArch);
    
    if (!requirements) {
        result.level = CompatibilityLevel::NOT_SUPPORTED;
        result.errors.push_back("Unknown architecture: " + architecture);
        return result;
    }
    
    // Check if architecture needs incremental extension
    using namespace duorou::extensions;
    if (GGMLIncrementalExtension::isArchitectureSupported(normalizedArch)) {
        result.mappedArchitecture = GGMLIncrementalExtension::getBaseArchitecture(normalizedArch);
        result.level = CompatibilityLevel::NEEDS_MAPPING;
        result.requiredModifications.push_back("GGML incremental extension: " + normalizedArch + " -> " + result.mappedArchitecture);
    } else {
        result.mappedArchitecture = normalizedArch;
    }
    
    // Determine compatibility level
    result.level = determineCompatibilityLevel(normalizedArch, *requirements);
    
    // Check for special requirements
    result.needsOllamaEngine = (ollamaRequiredArchitectures.find(normalizedArch) != ollamaRequiredArchitectures.end());
    result.hasVisionSupport = VisionModelHandler::hasVisionSupport(normalizedArch);
    result.hasAdvancedAttention = AttentionHandler::hasAdvancedAttention(normalizedArch);
    
    // Collect warnings and errors
    result.warnings = checkArchitectureWarnings(normalizedArch);
    result.errors = checkArchitectureErrors(normalizedArch);
    
    // Add recommendations
    auto modifications = getRecommendedModifications(normalizedArch);
    for (const auto& mod : modifications) {
        result.requiredModifications.push_back(mod);
    }
    
    // Add specific recommendations
    if (result.hasVisionSupport) {
        result.recommendations["vision"] = "Model supports vision processing. Ensure image preprocessing is available.";
    }
    
    if (result.hasAdvancedAttention) {
        result.recommendations["attention"] = "Model uses advanced attention mechanisms. Performance may vary.";
    }
    
    if (result.needsOllamaEngine) {
        result.recommendations["engine"] = "Model requires Ollama-specific processing. Consider using Ollama runtime.";
    }
    
    return result;
}

CompatibilityChecker::CompatibilityResult CompatibilityChecker::checkCompatibilityFromMetadata(
    const std::unordered_map<std::string, std::string>& metadata,
    const std::vector<std::string>& tensorNames
) {
    CompatibilityResult result;
    
    // Extract architecture from metadata
    auto archIt = metadata.find("general.architecture");
    if (archIt == metadata.end()) {
        result.level = CompatibilityLevel::NOT_SUPPORTED;
        result.errors.push_back("No architecture information found in metadata");
        return result;
    }
    
    std::string architecture = archIt->second;
    result = checkCompatibility(architecture);
    
    // Validate tensors
    auto requirements = getModelRequirements(normalizeArchitectureName(architecture));
    if (requirements) {
        auto [missingTensors, extraTensors] = validateTensors(architecture, tensorNames);
        
        for (const auto& missing : missingTensors) {
            result.errors.push_back("Missing required tensor: " + missing);
        }
        
        for (const auto& extra : extraTensors) {
            result.warnings.push_back("Unexpected tensor found: " + extra);
        }
        
        if (!missingTensors.empty()) {
            result.level = CompatibilityLevel::NOT_SUPPORTED;
        }
    }
    
    return result;
}

std::shared_ptr<CompatibilityChecker::ModelRequirements> CompatibilityChecker::getModelRequirements(const std::string& architecture) {
    if (!initialized) {
        initialize();
    }
    
    std::string normalizedArch = normalizeArchitectureName(architecture);
    auto it = modelRequirements.find(normalizedArch);
    if (it != modelRequirements.end()) {
        return it->second;
    }
    
    return nullptr;
}

bool CompatibilityChecker::isArchitectureSupported(const std::string& architecture) {
    if (!initialized) {
        initialize();
    }
    
    std::string normalizedArch = normalizeArchitectureName(architecture);
    
    // Check if directly supported
    if (llamaCppNativeArchitectures.find(normalizedArch) != llamaCppNativeArchitectures.end()) {
        return true;
    }
    
    // Check if can be mapped
    return duorou::extensions::GGMLIncrementalExtension::isArchitectureSupported(normalizedArch);
}

std::vector<std::string> CompatibilityChecker::getSupportedArchitectures() {
    if (!initialized) {
        initialize();
    }
    
    std::vector<std::string> supported;
    for (const auto& arch : llamaCppNativeArchitectures) {
        supported.push_back(arch);
    }
    
    // Add mapped architectures
    for (const auto& [arch, req] : modelRequirements) {
        if (std::find(supported.begin(), supported.end(), arch) == supported.end()) {
            supported.push_back(arch);
        }
    }
    
    return supported;
}

std::vector<std::string> CompatibilityChecker::getOllamaRequiredArchitectures() {
    if (!initialized) {
        initialize();
    }
    
    std::vector<std::string> required;
    for (const auto& arch : ollamaRequiredArchitectures) {
        required.push_back(arch);
    }
    
    return required;
}

std::pair<std::vector<std::string>, std::vector<std::string>> CompatibilityChecker::validateTensors(
    const std::string& architecture,
    const std::vector<std::string>& tensorNames
) {
    std::vector<std::string> missingTensors;
    std::vector<std::string> extraTensors;
    
    auto requirements = getModelRequirements(architecture);
    if (!requirements) {
        return {missingTensors, extraTensors};
    }
    
    // Check for missing required tensors
    for (const auto& required : requirements->requiredTensors) {
        if (std::find(tensorNames.begin(), tensorNames.end(), required) == tensorNames.end()) {
            missingTensors.push_back(required);
        }
    }
    
    // Check for extra tensors (not required or optional)
    for (const auto& tensor : tensorNames) {
        if (requirements->requiredTensors.find(tensor) == requirements->requiredTensors.end() &&
            requirements->optionalTensors.find(tensor) == requirements->optionalTensors.end()) {
            extraTensors.push_back(tensor);
        }
    }
    
    return {missingTensors, extraTensors};
}

bool CompatibilityChecker::isQuantizationSupported(
    const std::string& architecture,
    const std::string& quantization
) {
    auto requirements = getModelRequirements(architecture);
    if (!requirements) {
        return false;
    }
    
    return std::find(requirements->supportedQuantizations.begin(),
                    requirements->supportedQuantizations.end(),
                    quantization) != requirements->supportedQuantizations.end();
}

std::vector<std::string> CompatibilityChecker::getRecommendedModifications(const std::string& architecture) {
    std::vector<std::string> modifications;
    std::string normalizedArch = normalizeArchitectureName(architecture);
    
    // Check if needs incremental extension
    if (duorou::extensions::GGMLIncrementalExtension::isArchitectureSupported(normalizedArch)) {
        modifications.push_back("Update architecture metadata to: " + duorou::extensions::GGMLIncrementalExtension::getBaseArchitecture(normalizedArch));
    }
    
    // Check for vision models
    if (VisionModelHandler::hasVisionSupport(normalizedArch)) {
        modifications.push_back("Ensure vision processor tensors are properly formatted");
        modifications.push_back("Verify image preprocessing parameters");
    }
    
    // Check for advanced attention
    if (AttentionHandler::hasAdvancedAttention(normalizedArch)) {
        modifications.push_back("Configure attention mechanism parameters");
        if (AttentionHandler::usesSlidingWindow(normalizedArch)) {
            modifications.push_back("Set sliding window attention parameters");
        }
    }
    
    return modifications;
}

bool CompatibilityChecker::needsSpecialPreprocessing(const std::string& architecture) {
    std::string normalizedArch = normalizeArchitectureName(architecture);
    
    return VisionModelHandler::hasVisionSupport(normalizedArch) ||
           AttentionHandler::hasAdvancedAttention(normalizedArch) ||
           (ollamaRequiredArchitectures.find(normalizedArch) != ollamaRequiredArchitectures.end());
}

int CompatibilityChecker::getCompatibilityScore(const std::string& architecture) {
    auto result = checkCompatibility(architecture);
    
    switch (result.level) {
        case CompatibilityLevel::FULLY_COMPATIBLE:
            return 100;
        case CompatibilityLevel::NEEDS_MAPPING:
            return 85;
        case CompatibilityLevel::NEEDS_MODIFICATION:
            return 70;
        case CompatibilityLevel::PARTIALLY_SUPPORTED:
            return 50;
        case CompatibilityLevel::NOT_SUPPORTED:
        default:
            return 0;
    }
}

void CompatibilityChecker::initialize() {
    if (initialized) return;
    
    // Initialize native llama.cpp architectures
    llamaCppNativeArchitectures = {
        "llama", "qwen2", "gemma2", "qwen2vl", "phi3", "mistral", "mixtral"
    };
    
    // Initialize Ollama-required architectures
    ollamaRequiredArchitectures = {
        "qwen25vl", "gemma3", "mistral3", "gptoss"
    };
    
    // Create model requirements
    createLlamaRequirements();
    createQwen2Requirements();
    createQwen3Requirements();
    createQwen25vlRequirements();
    createGemma2Requirements();
    createGemma3Requirements();
    createGemma3nRequirements();
    createMistral3Requirements();
    createGptossRequirements();
    
    initialized = true;
}

// Implementation of create*Requirements methods
void CompatibilityChecker::createLlamaRequirements() {
    auto req = std::make_shared<ModelRequirements>();
    req->architecture = "llama";
    req->requiredTensors = {
        "token_embd.weight", "output_norm.weight", "output.weight"
    };
    req->optionalTensors = {
        "pos_embd.weight", "rope.freqs"
    };
    req->supportedQuantizations = {
        "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K", "F16", "F32"
    };
    req->minContextLength = 512;
    req->maxContextLength = 32768;
    
    modelRequirements["llama"] = req;
}

void CompatibilityChecker::createQwen2Requirements() {
    auto req = std::make_shared<ModelRequirements>();
    req->architecture = "qwen2";
    req->requiredTensors = {
        "token_embd.weight", "output_norm.weight", "output.weight"
    };
    req->supportedQuantizations = {
        "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K", "F16", "F32"
    };
    req->minContextLength = 512;
    req->maxContextLength = 131072;
    
    modelRequirements["qwen2"] = req;
}

void CompatibilityChecker::createQwen3Requirements() {
    auto req = std::make_shared<ModelRequirements>();
    req->architecture = "qwen3";
    req->requiredTensors = {
        "token_embd.weight", "output_norm.weight", "output.weight"
    };
    req->supportedQuantizations = {
        "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K", "F16", "F32"
    };
    req->minContextLength = 512;
    req->maxContextLength = 131072;
    
    modelRequirements["qwen3"] = req;
}

void CompatibilityChecker::createQwen25vlRequirements() {
    auto req = std::make_shared<ModelRequirements>();
    req->architecture = "qwen25vl";
    req->requiredTensors = {
        "token_embd.weight", "output_norm.weight", "output.weight",
        "vision.patch_embed.proj.weight", "vision.patch_embed.proj.bias"
    };
    req->optionalTensors = {
        "vision.pos_embed", "vision.temporal_embed"
    };
    req->supportedQuantizations = {
        "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "F16", "F32"
    };
    req->requiresVisionProcessor = true;
    req->minContextLength = 512;
    req->maxContextLength = 131072;
    
    modelRequirements["qwen25vl"] = req;
    modelRequirements["qwen2.5vl"] = req;
}

void CompatibilityChecker::createGemma2Requirements() {
    auto req = std::make_shared<ModelRequirements>();
    req->architecture = "gemma2";
    req->requiredTensors = {
        "token_embd.weight", "output_norm.weight", "output.weight"
    };
    req->supportedQuantizations = {
        "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K", "F16", "F32"
    };
    req->minContextLength = 512;
    req->maxContextLength = 8192;
    
    modelRequirements["gemma2"] = req;
}

void CompatibilityChecker::createGemma3Requirements() {
    auto req = std::make_shared<ModelRequirements>();
    req->architecture = "gemma3";
    req->requiredTensors = {
        "token_embd.weight", "output_norm.weight", "output.weight",
        "vision.patch_embed.proj.weight", "vision.patch_embed.proj.bias"
    };
    req->supportedQuantizations = {
        "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "F16", "F32"
    };
    req->requiresVisionProcessor = true;
    req->minContextLength = 512;
    req->maxContextLength = 8192;
    
    modelRequirements["gemma3"] = req;
}

void CompatibilityChecker::createGemma3nRequirements() {
    auto req = std::make_shared<ModelRequirements>();
    req->architecture = "gemma3n";
    req->requiredTensors = {
        "token_embd.weight", "output_norm.weight", "output.weight"
    };
    req->supportedQuantizations = {
        "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K", "F16", "F32"
    };
    req->minContextLength = 512;
    req->maxContextLength = 8192;
    
    modelRequirements["gemma3n"] = req;
}

void CompatibilityChecker::createMistral3Requirements() {
    auto req = std::make_shared<ModelRequirements>();
    req->architecture = "mistral3";
    req->requiredTensors = {
        "token_embd.weight", "output_norm.weight", "output.weight",
        "vision.patch_embed.proj.weight", "vision.patch_embed.proj.bias"
    };
    req->supportedQuantizations = {
        "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "F16", "F32"
    };
    req->requiresVisionProcessor = true;
    req->minContextLength = 512;
    req->maxContextLength = 131072;
    
    modelRequirements["mistral3"] = req;
}

void CompatibilityChecker::createGptossRequirements() {
    auto req = std::make_shared<ModelRequirements>();
    req->architecture = "gptoss";
    req->requiredTensors = {
        "token_embd.weight", "output_norm.weight", "output.weight"
    };
    req->supportedQuantizations = {
        "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K", "F16", "F32"
    };
    req->requiresSpecialTokenizer = true;
    req->minContextLength = 512;
    req->maxContextLength = 4096;
    
    modelRequirements["gptoss"] = req;
    modelRequirements["gpt-oss"] = req;
}

// Helper function implementations
CompatibilityChecker::CompatibilityLevel CompatibilityChecker::determineCompatibilityLevel(
    const std::string& architecture,
    const ModelRequirements& requirements
) {
    std::string normalizedArch = normalizeArchitectureName(architecture);
    
    // Check if natively supported
    if (llamaCppNativeArchitectures.find(normalizedArch) != llamaCppNativeArchitectures.end()) {
        return CompatibilityLevel::FULLY_COMPATIBLE;
    }
    
    // Check if needs incremental extension
    if (duorou::extensions::GGMLIncrementalExtension::isArchitectureSupported(normalizedArch)) {
        return CompatibilityLevel::NEEDS_MAPPING;
    }
    
    // Check if needs Ollama engine
    if (ollamaRequiredArchitectures.find(normalizedArch) != ollamaRequiredArchitectures.end()) {
        return CompatibilityLevel::PARTIALLY_SUPPORTED;
    }
    
    // Check if has special requirements
    if (requirements.requiresVisionProcessor || requirements.requiresSpecialTokenizer) {
        return CompatibilityLevel::NEEDS_MODIFICATION;
    }
    
    return CompatibilityLevel::NOT_SUPPORTED;
}

std::vector<std::string> CompatibilityChecker::checkArchitectureWarnings(const std::string& architecture) {
    std::vector<std::string> warnings;
    std::string normalizedArch = normalizeArchitectureName(architecture);
    
    if (VisionModelHandler::hasVisionSupport(normalizedArch)) {
        warnings.push_back("Vision models may have limited support in some llama.cpp versions");
    }
    
    if (AttentionHandler::hasAdvancedAttention(normalizedArch)) {
        warnings.push_back("Advanced attention mechanisms may impact performance");
    }
    
    if (ollamaRequiredArchitectures.find(normalizedArch) != ollamaRequiredArchitectures.end()) {
        warnings.push_back("Model may require Ollama-specific processing for full functionality");
    }
    
    return warnings;
}

std::vector<std::string> CompatibilityChecker::checkArchitectureErrors(const std::string& architecture) {
    std::vector<std::string> errors;
    std::string normalizedArch = normalizeArchitectureName(architecture);
    
    if (!isArchitectureSupported(normalizedArch)) {
        errors.push_back("Architecture not supported by llama.cpp");
    }
    
    return errors;
}

bool CompatibilityChecker::hasRequiredTensors(
    const std::vector<std::string>& tensorNames,
    const std::unordered_set<std::string>& requiredTensors
) {
    for (const auto& required : requiredTensors) {
        if (std::find(tensorNames.begin(), tensorNames.end(), required) == tensorNames.end()) {
            return false;
        }
    }
    return true;
}

std::string CompatibilityChecker::normalizeArchitectureName(const std::string& architecture) {
    std::string normalized = architecture;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);
    
    // Handle alternative names
    if (normalized == "qwen2.5vl") {
        normalized = "qwen25vl";
    } else if (normalized == "gpt-oss") {
        normalized = "gptoss";
    }
    
    return normalized;
}