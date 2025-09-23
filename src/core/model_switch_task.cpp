#include "model_switch_task.h"
#include <iostream>
#include <thread>
#include <sstream>

namespace duorou {
namespace core {

// ModelSwitchTask implementation
ModelSwitchTask::ModelSwitchTask(const std::string& id, const std::string& name, 
                               const std::string& target_model, TaskPriority priority)
    : BaseTask(id, name, priority)
    , target_model_(target_model)
    , simulated_duration_(std::chrono::milliseconds(1000)) {
}

TaskResult ModelSwitchTask::execute() {
    TaskResult result;
    
    try {
        std::cout << "[ModelSwitchTask] Starting model switch to: " << target_model_ << std::endl;
        
        // Check if cancelled
        if (isCancelled()) {
            result.success = false;
            result.message = "Task was cancelled before execution";
            return result;
        }
        
        // Simulate model switching process
        auto start_time = std::chrono::steady_clock::now();
        auto end_time = start_time + simulated_duration_;
        
        while (std::chrono::steady_clock::now() < end_time) {
            // Check cancellation status
            if (isCancelled()) {
                result.success = false;
                result.message = "Task was cancelled during execution";
                return result;
            }
            
            // Simulate work progress
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        std::cout << "[ModelSwitchTask] Model switch completed: " << target_model_ << std::endl;
        
        result.success = true;
        result.message = "Model switched to " + target_model_;
        result.output_data = "target_model: " + target_model_;
        
    } catch (const std::exception& e) {
        result.success = false;
        result.message = std::string("Exception during model switch: ") + e.what();
        std::cerr << "[ModelSwitchTask] Exception: " << e.what() << std::endl;
    } catch (...) {
        result.success = false;
        result.message = "Unknown exception during model switch";
        std::cerr << "[ModelSwitchTask] Unknown exception" << std::endl;
    }
    
    return result;
}

// TextGenerationTask implementation
TextGenerationTask::TextGenerationTask(const std::string& id, const std::string& prompt, TaskPriority priority)
    : BaseTask(id, "TextGeneration_" + id, priority)
    , prompt_(prompt)
    , simulated_duration_(std::chrono::milliseconds(2000)) {
}

TaskResult TextGenerationTask::execute() {
    TaskResult result;
    
    try {
        std::cout << "[TextGenerationTask] Starting text generation with prompt: " << prompt_ << std::endl;
        
        // Check if cancelled
        if (isCancelled()) {
            result.success = false;
            result.message = "Task was cancelled before execution";
            return result;
        }
        
        // Simulate text generation process
        auto start_time = std::chrono::steady_clock::now();
        auto end_time = start_time + simulated_duration_;
        
        std::ostringstream generated_text;
        generated_text << "Generated response for: \"" << prompt_ << "\"";
        
        while (std::chrono::steady_clock::now() < end_time) {
            // Check cancellation status
            if (isCancelled()) {
                result.success = false;
                result.message = "Task was cancelled during execution";
                return result;
            }
            
            // Simulate generation progress
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        
        std::cout << "[TextGenerationTask] Text generation completed" << std::endl;
        
        result.success = true;
        result.message = "Text generation completed successfully";
        result.output_data = "prompt: " + prompt_ + ", generated_text: " + generated_text.str() + ", model_used: llama_model";
        
    } catch (const std::exception& e) {
        result.success = false;
        result.message = std::string("Exception during text generation: ") + e.what();
        std::cerr << "[TextGenerationTask] Exception: " << e.what() << std::endl;
    } catch (...) {
        result.success = false;
        result.message = "Unknown exception during text generation";
        std::cerr << "[TextGenerationTask] Unknown exception" << std::endl;
    }
    
    return result;
}

// ImageGenerationTask implementation
ImageGenerationTask::ImageGenerationTask(const std::string& id, const std::string& prompt, TaskPriority priority)
    : BaseTask(id, "ImageGeneration_" + id, priority)
    , prompt_(prompt)
    , simulated_duration_(std::chrono::milliseconds(5000)) {
}

TaskResult ImageGenerationTask::execute() {
    TaskResult result;
    
    try {
        std::cout << "[ImageGenerationTask] Starting image generation with prompt: " << prompt_ << std::endl;
        
        // Check if cancelled
        if (isCancelled()) {
            result.success = false;
            result.message = "Task was cancelled before execution";
            return result;
        }
        
        // Simulate image generation process
        auto start_time = std::chrono::steady_clock::now();
        auto end_time = start_time + simulated_duration_;
        
        std::ostringstream image_path;
        image_path << "generated_image_" << getId() << ".png";
        
        while (std::chrono::steady_clock::now() < end_time) {
            // Check cancellation status
            if (isCancelled()) {
                result.success = false;
                result.message = "Task was cancelled during execution";
                return result;
            }
            
            // Simulate generation progress
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        
        std::cout << "[ImageGenerationTask] Image generation completed" << std::endl;
        
        result.success = true;
        result.message = "Image generation completed successfully";
        result.output_data = "prompt: " + prompt_ + ", image_path: " + image_path.str() + ", model_used: stable_diffusion_model, image_size: 512x512";
        
    } catch (const std::exception& e) {
        result.success = false;
        result.message = std::string("Exception during image generation: ") + e.what();
        std::cerr << "[ImageGenerationTask] Exception: " << e.what() << std::endl;
    } catch (...) {
        result.success = false;
        result.message = "Unknown exception during image generation";
        std::cerr << "[ImageGenerationTask] Unknown exception" << std::endl;
    }
    
    return result;
}

} // namespace core
} // namespace duorou