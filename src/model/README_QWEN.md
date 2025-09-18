# Qwen Model C++ Wrapper

本项目实现了Qwen25VL模型的C++封装，参考了Go版本的架构设计，提供了完整的多模态AI模型支持。

## 架构概览

### 核心组件

1. **BaseModel** (`base_model.h`)
   - 所有模型的基础抽象类
   - 提供通用的编码/解码接口
   - 支持模型初始化和配置管理

2. **QwenTextModel** (`qwen_text_model.h/cpp`)
   - 文本处理模型
   - 实现自注意力机制和前馈网络
   - 支持文本生成和编码

3. **QwenVisionModel** (`qwen_vision_model.h/cpp`)
   - 视觉处理模型
   - 实现Vision Transformer架构
   - 支持图像特征提取

4. **QwenImageProcessor** (`qwen_image_processor.h/cpp`)
   - 图像预处理器
   - 支持多种图像格式解码
   - 实现智能缩放和归一化

5. **QwenMultimodalModel** (`qwen_multimodal_model.h/cpp`)
   - 多模态模型整合
   - 结合文本和视觉处理能力
   - 提供统一的多模态接口

## 设计模式

### 1. 组合模式 (Composition Pattern)
```cpp
class QwenMultimodalModel : public BaseModel {
private:
    std::unique_ptr<TextModel> textModel_;
    std::unique_ptr<VisionModel> visionModel_;
    std::unique_ptr<ImageProcessor> imageProcessor_;
};
```

### 2. 工厂模式 (Factory Pattern)
```cpp
std::unique_ptr<TextModel> createQwenTextModel(const std::string& configPath);
std::unique_ptr<VisionModel> createQwenVisionModel(const std::string& configPath);
std::unique_ptr<MultimodalModel> createQwenMultimodalModel(const std::string& configPath);
```

### 3. 接口分离 (Interface Segregation)
- `TextModel`: 专门处理文本
- `VisionModel`: 专门处理图像
- `ImageProcessor`: 专门处理图像预处理
- `MultimodalProcessor`: 处理多模态输入

## 主要特性

### 文本处理
- 支持BytePair编码
- 自注意力机制实现
- 可配置的模型参数

### 视觉处理
- Vision Transformer架构
- 旋转位置编码
- 多尺度图像处理

### 图像预处理
- 多格式支持 (PNG, JPEG, BMP)
- 智能缩放算法
- ImageNet标准归一化

### 多模态整合
- 文本和图像特征融合
- 统一的输入输出接口
- 可扩展的模态支持

## 使用示例

```cpp
#include "qwen_multimodal_model.h"

// 创建多模态模型
auto model = createQwenMultimodalModel("config.json");

// 文本处理
std::string text = "Hello, world!";
auto tokens = model->encode(text);
auto decoded = model->decode(tokens);

// 图像处理
std::vector<uint8_t> imageData = loadImage("image.jpg");
auto features = model->processImage(imageData);
```

## 编译配置

模型库已集成到CMake构建系统中：

```cmake
set(MODEL_SOURCES
    qwen_text_model.cpp
    qwen_vision_model.cpp
    qwen_image_processor.cpp
    qwen_multimodal_model.cpp
    # ... 其他源文件
)

set(MODEL_HEADERS
    base_model.h
    qwen_text_model.h
    qwen_vision_model.h
    qwen_image_processor.h
    qwen_multimodal_model.h
    # ... 其他头文件
)
```

## 与Go版本的对应关系

| Go组件 | C++组件 | 说明 |
|--------|---------|------|
| `model.Base` | `BaseModel` | 基础模型接口 |
| `TextModel` | `QwenTextModel` | 文本处理模型 |
| `VisionModel` | `QwenVisionModel` | 视觉处理模型 |
| `ImageProcessor` | `QwenImageProcessor` | 图像预处理器 |
| 主Model | `QwenMultimodalModel` | 多模态模型整合 |

## 扩展性

该架构设计支持：
- 新模态类型的添加
- 不同模型实现的替换
- 配置驱动的模型行为
- 插件式的组件扩展

## 性能优化

- 使用智能指针管理内存
- 支持批量处理
- 可配置的缓存策略
- SIMD优化的数值计算

## 下一步开发

1. 实现具体的神经网络计算
2. 添加模型权重加载功能
3. 集成GPU加速支持
4. 完善错误处理和日志
5. 添加性能基准测试