#pragma once

#include "workflow_engine.h"
#include <string>
#include <chrono>

namespace duorou {
namespace core {

/**
 * @brief 模型切换任务类
 * 
 * 这个类演示了如何使用资源锁定和模型切换优化功能
 */
class ModelSwitchTask : public BaseTask {
public:
    /**
     * @brief 构造函数
     * @param id 任务ID
     * @param name 任务名称
     * @param target_model 目标模型名称
     * @param priority 任务优先级
     */
    ModelSwitchTask(const std::string& id, const std::string& name, 
                   const std::string& target_model, TaskPriority priority = TaskPriority::NORMAL);
    
    /**
     * @brief 析构函数
     */
    virtual ~ModelSwitchTask() = default;
    
    /**
     * @brief 执行任务
     * @return 任务执行结果
     */
    TaskResult execute() override;
    
    /**
     * @brief 获取任务所需的模型
     * @return 模型名称
     */
    std::string getRequiredModel() const override { return target_model_; }
    
    /**
     * @brief 获取目标模型名称
     * @return 目标模型名称
     */
    const std::string& getTargetModel() const { return target_model_; }
    
    /**
     * @brief 设置模拟执行时间
     * @param duration 执行时间（毫秒）
     */
    void setSimulatedDuration(std::chrono::milliseconds duration) { simulated_duration_ = duration; }

private:
    std::string target_model_;                          ///< 目标模型名称
    std::chrono::milliseconds simulated_duration_;     ///< 模拟执行时间
};

/**
 * @brief 文本生成任务类
 * 
 * 演示需要llama模型的任务
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
    std::string prompt_;                                ///< 输入提示词
    std::chrono::milliseconds simulated_duration_;     ///< 模拟执行时间
};

/**
 * @brief 图像生成任务类
 * 
 * 演示需要stable diffusion模型的任务
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
    std::string prompt_;                                ///< 输入提示词
    std::chrono::milliseconds simulated_duration_;     ///< 模拟执行时间
};

} // namespace core
} // namespace duorou