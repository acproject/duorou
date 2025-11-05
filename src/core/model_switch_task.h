#pragma once

#include "workflow_engine.h"
#include <string>
#include <chrono>

namespace duorou {
namespace core {

/**
 * @brief Model switching task class
 * 
 * This class demonstrates how to use resource locking and model switching optimization features
 */
class ModelSwitchTask : public BaseTask {
public:
    /**
     * @brief Constructor
     * @param id Task ID
     * @param name Task name
     * @param target_model Target model name
     * @param priority Task priority
     */
    ModelSwitchTask(const std::string& id, const std::string& name, 
                   const std::string& target_model, TaskPriority priority = TaskPriority::NORMAL);
    
    /**
     * @brief Destructor
     */
    virtual ~ModelSwitchTask() = default;
    
    /**
     * @brief Execute task
     * @return Task execution result
     */
    TaskResult execute() override;
    
    /**
     * @brief Get required model for task
     * @return Model name
     */
    std::string getRequiredModel() const override { return target_model_; }
    
    /**
     * @brief Get target model name
     * @return Target model name
     */
    const std::string& getTargetModel() const { return target_model_; }
    
    /**
     * @brief Set simulated execution time
     * @param duration Execution time (milliseconds)
     */
    void setSimulatedDuration(std::chrono::milliseconds duration) { simulated_duration_ = duration; }

private:
    std::string target_model_;                          ///< Target model name
    std::chrono::milliseconds simulated_duration_;     ///< Simulated execution time
};

/**
 * @brief Text generation task class
 * 
 * Demonstrates tasks that require llama model
 */
class TextGenerationTask : public BaseTask {
public:
    TextGenerationTask(const std::string& id, const std::string& prompt, TaskPriority priority = TaskPriority::NORMAL);
    virtual ~TextGenerationTask() = default;
    
    TaskResult execute() override;
    std::string getRequiredModel() const override { return "llama_model"; }
    
    const std::string& getPrompt() const { return prompt_; }
    void setSimulatedDuration(std::chrono::milliseconds duration) { simulated_duration_ = duration; }

private:
    std::string prompt_;                                ///< Input prompt
    std::chrono::milliseconds simulated_duration_;     ///< Simulated execution time
};

/**
 * @brief Image generation task class
 * 
 * Demonstrates tasks that require stable diffusion model
 */
class ImageGenerationTask : public BaseTask {
public:
    ImageGenerationTask(const std::string& id, const std::string& prompt, TaskPriority priority = TaskPriority::NORMAL);
    virtual ~ImageGenerationTask() = default;
    
    TaskResult execute() override;
    std::string getRequiredModel() const override { return "stable_diffusion_model"; }
    
    const std::string& getPrompt() const { return prompt_; }
    void setSimulatedDuration(std::chrono::milliseconds duration) { simulated_duration_ = duration; }

private:
    std::string prompt_;                                ///< Input prompt
    std::chrono::milliseconds simulated_duration_;     ///< Simulated execution time
};

} // namespace core
} // namespace duorou