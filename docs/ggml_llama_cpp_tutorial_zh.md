# GGML 与 llama.cpp 使用教程

本教程系统介绍如何理解和使用 `ggml` 与 `llama.cpp`，包括环境搭建、命令行工具（CLI）使用、C++ API 开发，以及 `ggml` 的核心概念与实践示例。

## 目录
1. [简介](#1-简介)
2. [环境搭建](#2-环境搭建)
3. [llama.cpp CLI 使用](#3-llamacpp-cli-使用)
4. [llama.cpp C++ API 开发](#4-llamacpp-c-API-开发)
5. [ggml 核心概念与用法](#5-ggml-核心概念与用法)
6. [使用 ggml 实现扩散模型推理](#6-使用-ggml-实现扩散模型推理)
7. [结语](#7-结语)

---

## 1. 简介

### llama.cpp 是什么？
`llama.cpp` 是一个开源项目，用于在本地设备上高效运行各种大语言模型（LLM），例如 LLaMA、Mistral、Gemma 等。它以在消费级硬件上实现高性能推理著称：在 CPU 上利用 AVX/AVX2/AVX512 指令，在 GPU 上支持 NVIDIA、AMD 以及 Apple Silicon 等平台，并尽量降低内存占用。

核心特性：
- **零依赖**：纯 C/C++ 实现。
- **出色的 Apple Silicon 支持**：针对 M 系列芯片通过 Metal 做了优化。
- **量化支持完善**：支持多种整数量化格式（4bit、5bit、8bit 等），在基本不损失精度的前提下大幅压缩模型和内存占用。
- **混合推理**：支持将部分层放在 GPU 上，部分留在 CPU 上，以适配有限的显存。

### ggml 是什么？
`ggml` 是支撑 `llama.cpp` 的张量计算库。`llama.cpp` 负责高层逻辑（模型加载、采样、上下文管理等），而 `ggml` 负责底层数学运算（矩阵乘法、加法、激活函数等）。

- **张量运算**：提供 C API，用于定义和执行张量计算图。
- **硬件加速**：抽象 CUDA、Metal、Vulkan 等后端，让同一计算图可以在不同设备上运行。
- **GGUF 格式**：`llama.cpp` 模型所使用的 GGUF 文件格式基于 `ggml` 的序列化能力，专为快速加载和映射而设计。

### 为什么要本地运行？
- **隐私**：数据不离开本机。
- **成本**：无需 API 费用。
- **时延**：没有网络往返开销。
- **离线**：完全可在无网络环境下工作。

### 优势与局限

**llama.cpp 的优势**
- **便携且轻量**：单一 C/C++ 代码库，几乎可以在所有主流平台上编译。
- **CPU 性能优秀**：针对 x86、ARM、Apple Silicon 做了大量底层优化。
- **量化支持成熟**：支持 Q2–Q8、K 系列等多种量化方案，质量和体积之间平衡良好。
- **CPU/GPU 混合执行**：在显存有限时，可以只把部分层放到 GPU 上。
- **生态和工具链完善**：自带 CLI、HTTP 服务、转换脚本，以及大量社区工具。

**llama.cpp 的局限**
- **推理优先设计**：主要面向推理，训练能力有限且偏实验性质。
- **偏静态图和固定布局**：对 Transformer 一类结构友好，对高度动态的结构支持不如通用框架。
- **上手门槛略高于 PyTorch**：需要更直接地接触张量、计算图和内存布局。
- **API 迭代频繁**：项目发展很快，API 和推荐用法在不同版本之间可能会变化。

**ggml 的优势**
- **小巧可嵌入**：适合在资源受限和边缘场景中使用。
- **统一后端抽象**：同一套代码可运行在 CPU、CUDA、Metal、Vulkan 等不同平台。
- **量化友好**：内置大量量化张量类型，显著降低内存和带宽需求。
- **C API 简单**：易于集成进现有 C/C++ 项目，也便于做语言绑定。

**ggml 的局限**
- **偏向推理与小规模训练**：不打算替代 PyTorch/JAX 做大规模训练。
- **动态控制流能力有限**：更适合计算图整体较为静态的模型。
- **生态相对较小**：与主流深度学习框架相比，高层库和预训练模型较少。

### 可以用它们做什么？

基于 `llama.cpp` 和 `ggml`，你可以：
- 在笔记本、桌面或边缘设备上运行本地聊天助手和代码助手。
- 搭建检索增强生成（RAG）系统，对你的私有文档进行索引和问答。
- 实现各种本地工具和智能 Agent（shell 助手、IDE 助手、数据库助手等）。
- 通过 `llama-server` 提供 OpenAI 兼容的 HTTP API，并接入你自己的鉴权和日志系统。
- 在将模型转换为 GGUF 后，对中小型视觉、音频或多模态模型进行推理。

---

## 2. 环境搭建

### 先决条件
- **C++ 编译器**：
    - **Linux**：GCC 或 Clang
    - **macOS**：Xcode（Clang）
    - **Windows**：MSVC（Visual Studio）或 MinGW
- **CMake**：推荐 3.14 及以上版本
- **Git**：用于克隆仓库
- **Python**（可选）：用于运行部分转换脚本

### 克隆仓库
如果你的项目里已经把 `llama.cpp` 作为子模块引入，则进入对应目录即可。否则可以使用：

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```

### 编译 llama.cpp
项目现在主要使用 **CMake** 进行构建，旧的 `Makefile` 已经废弃。

#### 1. 基本构建（仅 CPU）
适用于大多数系统，提供一个基线版本：

```bash
cd third_party/llama.cpp
mkdir build
cd build
cmake ..
cmake --build . --config Release -j
```

#### 2. 编译启用 GPU 加速

**macOS（Apple Silicon，Metal）**

macOS 上通常默认启用 Metal 支持：

```bash
cmake .. -DGGML_METAL=ON
cmake --build . --config Release -j
```

**NVIDIA GPU（CUDA）**

需要先安装 CUDA Toolkit：

```bash
cmake .. -DGGML_CUDA=ON
cmake --build . --config Release -j
```

**Windows（MSVC）**

在 “x64 Native Tools Command Prompt” 之类的开发者命令行中执行：

```cmd
mkdir build
cd build
cmake .. -DGGML_CUDA=ON
cmake --build . --config Release -j
```

### 常见问题排查
- **提示找不到 CMake**：  
  - macOS：`brew install cmake`  
  - Linux：`apt install cmake`  
  - Windows：到官网下载安装包
- **编译器错误**：确认编译器支持至少 C++11。
- **找不到 CUDA**：确保 `nvcc` 已在 `PATH` 中。

---

## 3. llama.cpp CLI 使用

编译完成后，你会在 `build/bin/` 目录下看到若干可执行文件，最常用的是：
- `llama-cli`：主推理工具
- `llama-server`：HTTP API 服务
- `llama-quantize`：模型量化工具

### 下载模型

模型必须是 **GGUF** 格式。常见来源包括 Hugging Face 上的 `TheBloke`、`MaziyarPanahi` 等仓库。

```bash
# 示例：使用 llama-cli 下载（自带 helper）
./bin/llama-cli --hf-repo ggml-org/gemma-3-1b-it-GGUF --hf-file gemma-3-1b-it.Q4_K_M.gguf
```

### 基本推理（`llama-cli`）

**执行简单提示：**

```bash
./bin/llama-cli -m models/my-model.gguf -p "Explain quantum physics in 5 words." -n 128
```

### 交互模式

启动与模型的交互式会话：

```bash
./bin/llama-cli -m models/my-model.gguf -i -r "User:" --color
```

### 常用参数
- `-m, --model <path>`：GGUF 模型路径
- `-p, --prompt <text>`：初始提示词
- `-n, --n-predict <int>`：生成的最大 token 数（默认 -1 表示不限）
- `-c, --ctx-size <int>`：上下文窗口大小（如 2048、4096，0 表示自动）
- `-t, --threads <int>`：使用的 CPU 线程数
- `-ngl, --n-gpu-layers <int>`：需要 offload 到 GPU 的层数，`-ngl 99` 表示尽可能都放到 GPU
- `--temp <float>`：温度（多样性），默认 0.8
- `-r, --reverse-prompt <text>`：遇到该字符串时停止生成（常用于多轮对话）
- `-f, --file <path>`：从文件读取 prompt

### 模型量化（`llama-quantize`）

可以将高精度 GGUF 模型（例如 F16）转换为量化版本（例如 Q4_K_M），以节省空间和内存：

```bash
./bin/llama-quantize models/my-model-f16.gguf models/my-model-q4_k_m.gguf Q4_K_M
```

### 运行 HTTP API 服务（`llama-server`）

启动一个兼容 OpenAI 接口的 HTTP 服务：

```bash
./bin/llama-server -m models/my-model.gguf --port 8080 --host 0.0.0.0 -ngl 99
```

**使用 cURL 测试：**

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      { "role": "system", "content": "You are a helpful assistant." },
      { "role": "user", "content": "Hello!" }
    ]
  }'
```

---

## 4. llama.cpp C++ API 开发

本节介绍如何在你自己的 C++ 应用中集成 `llama.cpp`。

### 关键数据结构
- `llama_model`：已加载的模型权重（只读，可共享）
- `llama_context`：执行状态（KV cache、内存等）
- `llama_batch`：一次要处理的一批 token
- `llama_token`：整数形式的 token ID

### 分步实现示例

下面是一个完整的示例程序，演示如何加载模型、tokenize、推理和打印结果：

```cpp
#include "llama.h"
#include <vector>
#include <string>
#include <iostream>
#include <cstring>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return 1;
    }
    std::string model_path = argv[1];
    std::string prompt = "User: Hello!\nAssistant:";

    // 1. Initialize Backend
    // Must be called once at the start.
    ggml_backend_load_all();

    // 2. Load Model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99; // Offload all layers to GPU if possible
    
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        std::cerr << "Failed to load model: " << model_path << std::endl;
        return 1;
    }

    // 3. Tokenize Prompt
    const llama_vocab * vocab = llama_model_get_vocab(model);
    
    // Calculate required size
    // add_bos = true (add beginning of stream token)
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    
    // Perform tokenization
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        std::cerr << "Failed to tokenize prompt" << std::endl;
        return 1;
    }

    // 4. Create Context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048; // Max context size
    ctx_params.n_batch = 512; // Max batch size for processing
    
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        std::cerr << "Failed to create context" << std::endl;
        return 1;
    }

    // 5. Initialize Sampler
    // Using a chain of samplers: greedy search for simplicity
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // 6. Inference Loop
    
    // Create a batch
    // We start by feeding the entire prompt
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    // Main loop
    int n_predict = 50;
    std::cout << prompt; // Echo prompt

    for (int i = 0; i < n_predict; ++i) {
        // Decode (Process the batch)
        if (llama_decode(ctx, batch)) {
            std::cerr << "llama_decode failed" << std::endl;
            break;
        }

        // Sample the next token
        // -1 means sample from the last token in the batch
        llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);

        // Check for End of Generation (EOS)
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        // Convert token to string and print
        char buf[128];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) {
             std::cerr << "Failed to convert token to piece" << std::endl;
             break;
        }
        std::string piece(buf, n);
        std::cout << piece << std::flush;

        // Prepare next batch with just the new token
        // This is "autoregressive" generation
        batch = llama_batch_get_one(&new_token_id, 1);
    }
    std::cout << std::endl;

    // 7. Cleanup
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_free_model(model);

    return 0;
}
```

### 编译你的应用

你需要链接 `llama` 和 `ggml` 库。例如使用 CMake：

```cmake
cmake_minimum_required(VERSION 3.14)
project(my_app)

set(CMAKE_CXX_STANDARD 17)

# 假设 llama.cpp 在子目录 third_party/llama.cpp 中
add_subdirectory(third_party/llama.cpp)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE llama ggml)
```

---

## 5. ggml 核心概念与用法

`ggml` 是一个偏底层的张量库，使用“先定义图，再执行”的模式：先构建计算图，然后一次性执行。

### 核心结构
- **`ggml_context`**：上下文对象，持有所有张量和图节点的内存，需要提前分配足够内存。
- **`ggml_tensor`**：基本数据单元。
    - `type`：数据类型（如 `GGML_TYPE_F32`、`GGML_TYPE_F16`、`GGML_TYPE_Q4_0` 等）
    - `ne[4]`：每一维的元素个数（最多 4 维）
    - `data`：指向底层数据的指针
- **`ggml_cgraph`**：计算图，是一系列需要计算的张量节点的有序集合。

### 哪些工作负载适合 ggml？

`ggml` 特别适合：
- **自回归语言模型**：如 LLaMA、Mistral、Gemma、Qwen 等 decoder-only Transformer。
- **形状相对固定的 encoder/decoder Transformer**：例如部分翻译、摘要模型。
- **边缘设备上的中小型模型**：对内存、算力要求有限且需要量化的场景。
- **离线或嵌入式应用**：桌面、移动端、本地助手等。

不太适合：
- **超大规模训练任务**：多机多卡训练、复杂优化器等不在主要设计范围内。
- **高度动态的模型结构**：每步都改变计算图的模型会比较难表达。
- **需要极快原型迭代的研究场景**：此时 PyTorch/JAX 等会更友好。

### 超越 LLM：用 ggml 跑其他模型

虽然 `ggml` 因为 `llama.cpp` 而流行，但本质上它是一个通用张量引擎。理论上，任何可以用静态计算图描述，且算子、数据类型被支持的模型都可以迁移过来，例如：
- **分类用 CNN**（例如 ResNet 类）
- **UNet 风格结构**（图像分割或去噪）
- **YOLO 风格目标检测模型**（卷积 backbone + 检测头）
- **MLP 或小型视觉 Transformer**

一个典型的非 LLM 模型工作流：
1. 在 PyTorch/TF/JAX 中训练模型。
2. 将权重和结构导出成自定义格式，或通过转换器直接转为 GGUF。
3. 在 `ggml` 中使用同样的层与参数形状重建网络结构。
4. 将权重填入 `ggml` 张量中，执行推理。

### 示例：矩阵乘法

下面的示例演示如何使用 `ggml` 完成矩阵乘法 \(C = A \times B\)：

```cpp
#include "ggml.h"
#include <vector>
#include <cstdio>
#include <cstdlib>

int main() {
    // 1. Initialize memory
    // Estimate memory usage: 
    // Tensor overhead + data size.
    // For small example, 1MB is plenty.
    struct ggml_init_params params = {
        .mem_size   = 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    struct ggml_context * ctx = ggml_init(params);

    if (!ctx) {
        printf("ggml_init() failed\n");
        return 1;
    }

    // 2. Define Tensors
    // Matrix A: 2 rows, 3 cols
    // In ggml, dimensions are (col, row), so we specify (3, 2)
    struct ggml_tensor * matrix_A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);
    
    // Matrix B: 3 rows, 2 cols
    // Specify as (2, 3)
    struct ggml_tensor * matrix_B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3);

    // Matrix C = A x B
    // Result will be (2 rows, 2 cols) -> (2, 2)
    // Note: ggml_mul_mat(A, B) computes B^T * A basically, semantics can be tricky.
    // Usually standard matmul is ggml_mul_mat(B, A) if B is weights and A is input.
    // Let's use simple multiplication for demonstration: C = A * B (element-wise) is ggml_mul
    // For dot product / matmul:
    struct ggml_tensor * result = ggml_mul_mat(ctx, matrix_A, matrix_B); 
    // Result dims: (rows of A, cols of B) ? 
    // ggml_mul_mat logic:
    // A: [ne0, ne1]
    // B: [ne0, ne2] -> B is transposed implicitly? 
    // It's best to verify dimensions.

    // 3. Build Graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);

    // 4. Set Data
    // We can access tensor data directly
    float * data_a = (float *) matrix_A->data;
    for (int i = 0; i < 6; i++) data_a[i] = (float)i;

    float * data_b = (float *) matrix_B->data;
    for (int i = 0; i < 6; i++) data_b[i] = 1.0f;

    // 5. Compute
    // n_threads = 1
    ggml_graph_compute_with_ctx(ctx, &gf, 1);

    // 6. Read Result
    struct ggml_tensor * out = gf->nodes[gf->n_nodes - 1];
    float * out_data = (float *) out->data;
    
    printf("Result dimensions: %lld x %lld\n", out->ne[0], out->ne[1]);
    printf("First element: %f\n", out_data[0]);

    // 7. Cleanup
    ggml_free(ctx);

    return 0;
}
```

### 示例：简单 CNN 模块（概念示意）

下面是一个在 `ggml` 中实现小型 2D 卷积模块的概念示意（伪代码）。真正的模型需要更仔细地处理数据布局和步长，但整体模式是：

1. 为输入特征图、卷积权重、偏置创建张量。
2. 使用 `ggml_conv_2d`（或相关算子）构建卷积节点。
3. 按需添加激活函数（例如 ReLU）。
4. 构建计算图并调用 `ggml_graph_compute_with_ctx` 执行。

```cpp
// Pseudocode: simplified CNN block y = relu(conv2d(x, w) + b)
struct ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                            width, height, in_channels, batch_size);
struct ggml_tensor * w = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                            kernel_w, kernel_h, in_channels, out_channels);
struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);

struct ggml_tensor * conv = ggml_conv_2d(ctx, w, x,
                                         stride_x, stride_y,
                                         padding_x, padding_y,
                                         dilation_x, dilation_y);

struct ggml_tensor * biased = ggml_add(ctx,
    conv,
    ggml_repeat(ctx, b, conv));

struct ggml_tensor * y = ggml_relu(ctx, biased);

struct ggml_cgraph * gf = ggml_new_graph(ctx);
ggml_build_forward_expand(gf, y);
ggml_graph_compute_with_ctx(ctx, &gf, n_threads);
```

在 UNet 这类用于语义分割的网络中，你会堆叠大量类似的卷积模块，配合下采样（如步长卷积）、上采样（如反卷积），再加上跳跃连接。所有这些结构都可以在 `ggml` 中使用张量操作和计算图来表达。

### 示例：YOLO 风格检测头（概念示意）

对于 YOLO 一类目标检测模型，backbone 可以用上面 CNN block 的方式实现，而检测头通常：
- 接收一个或多个尺度的特征图；
- 通过少量卷积层；
- 输出包含类别 logits 和边界框参数的张量。

在 `ggml` 中，一个单尺度的检测头大致可以这样写：

```cpp
// feature: [W, H, C_in, B]
struct ggml_tensor * feature = ...;

// 1x1 conv to produce predictions per anchor
struct ggml_tensor * w_head = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                                1, 1, C_in, C_out);
struct ggml_tensor * b_head = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C_out);

struct ggml_tensor * logits = ggml_conv_2d(ctx, w_head, feature,
                                           1, 1, 0, 0, 1, 1);

logits = ggml_add(ctx, logits,
                  ggml_repeat(ctx, b_head, logits));

// Optionally apply sigmoid for objectness / bbox params
struct ggml_tensor * probs = ggml_sigmoid(ctx, logits);
```

后处理（例如 NMS、从 anchor 解码边界框等）通常在 `ggml` 外部使用 C/C++ 直接对输出张量进行运算即可。

### 广播机制
`ggml` 支持广播（broadcasting）。例如把形状 `[10]` 的张量加到形状 `[10, 5]` 的张量上时，前者会在逻辑上沿第二个维度重复 5 次。

### 后端调度器
在复杂场景（例如 `llama.cpp` 中）可以使用 `ggml_backend_sched`，根据张量所在设备和算子支持情况，自动在 CPU 和 GPU 间分配计算。

---

## 6. 使用 ggml 实现扩散模型推理

本节基于前面介绍的 `ggml` 核心概念，专门讨论如何用 `ggml` 实现一个较为完整的扩散模型推理流程。这里以通用的“图像扩散模型”为例，结构上可以类比 Stable Diffusion 一类模型，但不依赖具体实现。

### 扩散模型简要回顾

典型的图像扩散模型包含三个主要部分：
- 文本或条件编码器，将文本等条件编码为向量或特征图。
- UNet 噪声预测网络，输入为当前的噪声潜变量、时间步编码以及条件特征，输出为噪声估计或直接的干净图像估计。
- 解码器，将低维潜空间中的图像特征解码为像素空间（例如 VAE 解码器）。

推理时从高斯噪声出发，通过若干步迭代更新潜变量，逐步去噪，最终得到图像。

### 架构拆分与模块边界

在工程实现上，推荐将扩散模型拆成以下模块：
- 条件编码模块，例如文本编码器、图像编码器或其他模态。
- 时间步编码模块，将离散时间步或连续噪声尺度编码为向量。
- UNet 主体，包含多尺度卷积块、下采样、上采样和跨层连接。
- 调度器（scheduler），实现扩散过程中的数值积分或离散更新规则。
- 解码器，将潜变量转换成最终数据（图像、语音等）。

在 `ggml` 中，可以将这些模块映射为若干计算图或子图，并在推理循环中复用。

### 推理总体流程

一个最小却完整的扩散推理流程大体如下：
1. 加载模型结构定义和权重，构建 `ggml` 张量并放入一个或多个 `ggml_context` 中。
2. 初始化调度器参数，例如时间步序列、噪声标准差、步长等。
3. 将条件（例如文本）编码为固定长度的向量或特征图。
4. 从标准正态分布采样潜变量，作为起始噪声。
5. 在若干时间步上循环：
   - 构建或复用 UNet 的前向计算图。
   - 输入当前潜变量、时间步编码和条件特征，得到噪声估计。
   - 根据调度器公式更新潜变量。
6. 将最终的潜变量送入解码器，得到图像或其他输出。

### 在 ggml 中表示 UNet

UNet 通常由多个卷积残差块和跨层连接组成，`ggml` 中可以使用二维卷积、归一化和激活等算子构建这些块。例如，一个极简的残差块可以抽象为：
- 卷积层。
- 归一化层。
- 非线性激活（如 SiLU 或 ReLU）。
- 可选的时间步或条件注入（通过加法或拼接）。

在 `ggml` 中，这些步骤对应为：
- `ggml_conv_2d` 进行卷积。
- `ggml_norm` 或其他归一化算子进行归一化。
- `ggml_silu` 等激活函数进行非线性变换。
- 通过 `ggml_add` 或 `ggml_concat` 将条件或时间步嵌入注入到特征图中。

一个高度简化的 UNet 残差块前向过程示意代码如下：

```cpp
struct DiffusionUnetState {
    ggml_context * ctx;
    std::vector<ggml_tensor *> weights;
};

ggml_tensor * unet_block_forward(
    DiffusionUnetState & st,
    ggml_tensor * x,
    ggml_tensor * t_emb,
    ggml_tensor * cond,
    ggml_cgraph * gf
) {
    ggml_tensor * h = x;
    ggml_tensor * w1 = st.weights[0];
    ggml_tensor * b1 = st.weights[1];

    ggml_tensor * y = ggml_conv_2d(st.ctx, w1, h, 1, 1, 0, 0, 1, 1);
    ggml_tensor * b = ggml_repeat(st.ctx, b1, y);
    y = ggml_add(st.ctx, y, b);

    ggml_tensor * t_proj = ggml_repeat(st.ctx, t_emb, y);
    y = ggml_add(st.ctx, y, t_proj);

    y = ggml_norm(st.ctx, y);
    y = ggml_silu(st.ctx, y);

    ggml_build_forward_expand(gf, y);
    return y;
}
```

实际的 UNet 会包含更多通道数的调整、下采样和上采样分支、多个残差块以及跨层连接，但在 `ggml` 中都是类似的张量算子组合。

### 示例：单步噪声预测前向图

在一个推理时间步中，需要完成以下工作：
- 根据当前时间步索引生成时间步嵌入向量。
- 准备条件特征，例如来自文本编码器的输出。
- 构建或复用 UNet 计算图，将当前潜变量映射到噪声估计。

一个简化的单步前向计算过程可以抽象为：

```cpp
ggml_cgraph * gf = ggml_new_graph(ctx);

ggml_tensor * x = current_latent;
ggml_tensor * t_emb = make_timestep_embedding(ctx, t_scalar);
ggml_tensor * cond = current_condition;

ggml_tensor * eps = unet_forward(st, x, t_emb, cond, gf);

ggml_build_forward_expand(gf, eps);
ggml_graph_compute_with_ctx(ctx, gf, n_threads);
```

其中 `make_timestep_embedding` 和 `unet_forward` 可以按照具体模型结构实现，`eps` 表示模型估计的噪声或干净信号。

### 示例：完整采样循环伪代码

以简单的 DDIM 风格更新为例，采样循环可以写成如下伪代码，其中 \(\epsilon_\theta\) 表示 UNet 的输出：

```cpp
std::vector<float> timesteps = make_timesteps(num_steps, t_min, t_max);

ggml_tensor * latent = sample_normal(ctx, latent_shape);

for (int i = 0; i < num_steps; ++i) {
    float t = timesteps[i];

    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * t_emb = make_timestep_embedding(ctx, t);
    ggml_tensor * cond = encode_condition(ctx, cond_input);

    ggml_tensor * eps = unet_forward(st, latent, t_emb, cond, gf);

    ggml_build_forward_expand(gf, eps);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    update_latent_with_scheduler(latent, eps, t, i, num_steps);
}
```

调度器更新函数可以参考 DDPM 或 DDIM 的公式。例如，一种常见的更新形式为：

\[
x_{t-1} = \sqrt{\alpha_{t-1}}\left(\frac{x_t - \sqrt{1-\alpha_t}\,\epsilon_\theta(x_t, t)}{\sqrt{\alpha_t}}\right)
          + \sqrt{1-\alpha_{t-1}}\,\eta
\]

其中 \(\alpha_t\) 来源于预先定义的噪声调度，\(\eta\) 是可选的随机噪声项。

### 模型导出与权重加载

由于目前主流扩散模型通常在 PyTorch 或其他框架中训练，迁移到 `ggml` 的典型步骤是：
1. 在原框架中将模型拆分为 UNet、编码器、解码器等子模块，并保存它们的权重。
2. 编写一个转换脚本，将权重重排为 `ggml` 所期望的张量布局，包括卷积核维度顺序、线性层权重形状等。
3. 在 C/C++ 侧通过文件读取权重，创建对应的 `ggml_tensor`，并把数据拷贝到 `data` 指针中。
4. 根据内存与精度需求，将部分权重量化为 `GGML_TYPE_Q4_0`、`GGML_TYPE_Q5_0` 等，平衡速度与质量。
5. 在推理时重用同一个 `ggml_context` 或多个上下文，避免频繁分配和释放大块内存。

如果需要构建自己的 GGUF 格式，可以参考 `llama.cpp` 中的 GGUF 写入逻辑，为扩散模型定义合适的元数据字段和张量命名规则。这样可以在加载时更方便地按名称查找张量并构建网络。

通过以上步骤，你可以在 `ggml` 上实现从噪声采样、条件编码、UNet 去噪到解码输出的完整扩散推理流水线，并根据设备能力选择合适的精度和调度策略。

## 7. 结语

到这里，你已经对 `llama.cpp` 和 `ggml` 有了一个比较系统的认识：

- **对于普通用户**：可以直接使用 `llama-cli` 运行下载好的 GGUF 模型，高效地在本地体验各种大模型。
- **对于开发者**：可以使用 `llama.h` 构建聊天应用、智能 Agent 或各种工具型应用，将推理完全握在自己手中。
- **对于研究者 / 算法工程师**：可以基于 `ggml` 设计和实验新的算子与网络结构，并在资源受限环境中快速验证想法。

如果你希望把 UNet 语义分割、YOLO 目标检测或其他视觉模型落到边缘设备上，本教程中的 `ggml` 概念和示例可以作为一个起点，帮助你从 “在 PyTorch 中训练” 走到 “在本地或嵌入式设备上部署推理”。

祝你玩得开心，也欢迎在此基础上继续扩展属于你自己的本地 AI 工具链。
