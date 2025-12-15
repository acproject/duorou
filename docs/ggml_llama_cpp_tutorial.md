# GGML and llama.cpp Usage Tutorial

This document provides a comprehensive guide to understanding and using `ggml` and `llama.cpp`. It covers environment setup, CLI usage, C++ API development, and core `ggml` concepts.

## Table of Contents
1. [Introduction](#1-introduction)
2. [Environment Setup](#2-environment-setup)
3. [llama.cpp CLI Usage](#3-llamacpp-cli-usage)
4. [llama.cpp C++ API Development](#4-llamacpp-c-api-development)
5. [ggml Core Concepts and Usage](#5-ggml-core-concepts-and-usage)
6. [Conclusion](#6-conclusion)

---

## 1. Introduction

### What is llama.cpp?
`llama.cpp` is a state-of-the-art open-source project dedicated to running Large Language Models (LLMs) like LLaMA, Mistral, and Gemma locally on consumer-grade hardware. It is famous for enabling high-performance inference on CPUs (using AVX/AVX2/AVX512 instructions) and GPUs (NVIDIA, AMD, Apple Silicon, etc.) with minimal memory usage.

Key features include:
- **No Dependencies**: Pure C/C++ implementation.
- **Apple Silicon Support**: Optimized specifically for M-series chips using the Metal framework.
- **Quantization**: Supports various integer quantization formats (e.g., 4-bit, 5-bit, 8-bit) to drastically reduce model size and memory footprint with negligible accuracy loss.
- **Hybrid Inference**: Can offload parts of the model to the GPU while keeping the rest on the CPU.

### What is ggml?
`ggml` is the tensor library that powers `llama.cpp`. While `llama.cpp` provides the high-level logic for LLMs (loading models, sampling tokens, managing context), `ggml` handles the low-level mathematical operations (matrix multiplication, addition, activation functions).

- **Tensor Operations**: Provides a C API for defining and executing tensor computation graphs.
- **Hardware Acceleration**: Abstracts hardware backends (CUDA, Metal, Vulkan, etc.) so the same graph can run on different devices.
- **GGUF Format**: The file format used by `llama.cpp` models is based on `ggml`'s serialization capabilities. It is a binary format designed for fast loading and mapping.

### Why Run Locally?
- **Privacy**: No data leaves your machine.
- **Cost**: No API fees.
- **Latency**: No network overhead.
- **Offline Access**: Works without internet.

### Strengths and Limitations

**Strengths of llama.cpp**
- **Portable and lightweight**: Single C/C++ codebase that builds almost everywhere.
- **Great CPU performance**: Highly optimized kernels for x86, ARM, and Apple Silicon.
- **Rich quantization support**: Mature quantization schemes (Q2–Q8, K-quant, etc.) with good quality.
- **Hybrid CPU/GPU execution**: Can offload only part of the model to GPU when VRAM is limited.
- **Ecosystem and tooling**: Comes with CLI, server, conversion scripts, and many community tools.

**Limitations of llama.cpp**
- **Inference-first design**: Focused on inference; training support is minimal and experimental.
- **Static graph and layouts**: Best suited for transformer-like architectures; very dynamic models can be harder.
- **Less “plug-and-play” than PyTorch**: You need to work closer to the metal (graphs, tensors, buffers).
- **Evolving APIs**: The project moves fast; APIs and recommended flows can change between versions.

**Strengths of ggml**
- **Small and embeddable**: Designed for use in resource-constrained environments and edge devices.
- **Backend abstraction**: Same code can target CPU, CUDA, Metal, Vulkan, etc.
- **Quantization-aware**: Many tensor types are quantized, which reduces memory and bandwidth usage.
- **Simple C API**: Easy to bind to other languages and integrate into existing C/C++ projects.

**Limitations of ggml**
- **Focused on inference and small training loops**: Not a full replacement for PyTorch/JAX for large-scale training.
- **Limited dynamic control flow**: Best when the computation graph is mostly static.
- **Smaller ecosystem**: Fewer high-level libraries and pretrained models compared to mainstream DL frameworks.

### What Can You Build With Them?

With `llama.cpp` and `ggml`, you can:
- Run local chat assistants and coding helpers on laptops, desktops, or edge devices.
- Build RAG (Retrieval-Augmented Generation) systems that index your own documents.
- Implement local tools and agents (shell assistant, IDE assistant, database assistant).
- Serve OpenAI-compatible APIs (`llama-server`) behind your own auth and logging.
- Prototype and deploy small to medium-sized vision, audio, or multimodal models when converted to GGUF.

---

## 2. Environment Setup

### Prerequisites
- **C++ Compiler**:
    - **Linux**: GCC or Clang.
    - **macOS**: Xcode (Clang).
    - **Windows**: MSVC (Visual Studio) or MinGW.
- **CMake**: Version 3.14 or later.
- **Git**: For cloning the repository.
- **Python**: (Optional) For running conversion scripts.

### Cloning the Repository
If `llama.cpp` is already a submodule in your project, navigate to it. Otherwise, clone it:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```

### Building llama.cpp
The project now primarily uses **CMake** for building. The old `Makefile` has been deprecated.

#### 1. Basic Build (CPU Only)
This works on most systems and provides a baseline.

```bash
cd third_party/llama.cpp
mkdir build
cd build
cmake ..
cmake --build . --config Release -j
```

#### 2. Building with GPU Acceleration

**macOS (Apple Silicon - Metal)**
Metal support is usually enabled by default on macOS.
```bash
cmake .. -DGGML_METAL=ON
cmake --build . --config Release -j
```

**NVIDIA GPU (CUDA)**
Requires CUDA Toolkit installed.
```bash
cmake .. -DGGML_CUDA=ON
cmake --build . --config Release -j
```

**Windows (MSVC)**
Open a developer command prompt (e.g., "x64 Native Tools Command Prompt").
```cmd
mkdir build
cd build
cmake .. -DGGML_CUDA=ON
cmake --build . --config Release -j
```

### Troubleshooting
- **Missing CMake**: Install via `brew install cmake` (macOS), `apt install cmake` (Linux), or download installer (Windows).
- **Compiler Errors**: Ensure you have a C++11 compliant compiler.
- **CUDA Not Found**: Make sure `nvcc` is in your PATH.

---

## 3. llama.cpp CLI Usage

After building, you will find several binaries in `build/bin/`. The most important ones are `llama-cli` (main inference tool), `llama-server` (API server), and `llama-quantize`.

### Downloading Models
Models must be in **GGUF** format. Popular sources include Hugging Face repositories like `TheBloke` or `MaziyarPanahi`.

```bash
# Example: Download using llama-cli (built-in helper)
./bin/llama-cli --hf-repo ggml-org/gemma-3-1b-it-GGUF --hf-file gemma-3-1b-it.Q4_K_M.gguf
```

### Basic Inference (`llama-cli`)

**Run a prompt:**
```bash
./bin/llama-cli -m models/my-model.gguf -p "Explain quantum physics in 5 words." -n 128
```

**Interactive Mode:**
Start a conversation with the AI.
```bash
./bin/llama-cli -m models/my-model.gguf -i -r "User:" --color
```

**Common Arguments:**
- `-m, --model <path>`: Path to the GGUF file.
- `-p, --prompt <text>`: Initial prompt.
- `-n, --n-predict <int>`: Max tokens to generate (default: -1 = infinity).
- `-c, --ctx-size <int>`: Context window size (e.g., 2048, 4096). 0 = auto-detect.
- `-t, --threads <int>`: Number of CPU threads to use.
- `-ngl, --n-gpu-layers <int>`: Layers to offload to GPU. `-ngl 99` loads everything to GPU.
- `--temp <float>`: Temperature (creativity). Default 0.8.
- `-r, --reverse-prompt <text>`: String that halts generation (useful for chat).
- `-f, --file <path>`: Read prompt from a file.

### Quantizing Models (`llama-quantize`)
You can convert a high-precision GGUF model (e.g., F16) to a quantized version (e.g., Q4_K_M) to save space.

```bash
./bin/llama-quantize models/my-model-f16.gguf models/my-model-q4_k_m.gguf Q4_K_M
```

### Running an API Server (`llama-server`)
Hosts an OpenAI-compatible HTTP server.

```bash
./bin/llama-server -m models/my-model.gguf --port 8080 --host 0.0.0.0 -ngl 99
```

**Test with cURL:**
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

## 4. llama.cpp C++ API Development

This section explains how to integrate `llama.cpp` into your own C++ application.

### Key Data Structures
- `llama_model`: Represents the loaded model weights (read-only, shared).
- `llama_context`: Represents the execution state (KV cache, memory).
- `llama_batch`: A batch of tokens to be processed.
- `llama_token`: Integer ID representing a token.

### Step-by-Step Implementation

Here is a complete, commented example.

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

### Compiling Your Application
You need to link against `llama` and `ggml`.

```cmake
cmake_minimum_required(VERSION 3.14)
project(my_app)

set(CMAKE_CXX_STANDARD 17)

# Assuming llama.cpp is in a subdirectory
add_subdirectory(third_party/llama.cpp)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE llama ggml)
```

---

## 5. ggml Core Concepts and Usage

`ggml` is a low-level tensor library. It uses a "define-and-run" approach where you first build a computation graph, and then execute it.

### Core Structures
- **`ggml_context`**: A container that holds the memory for all tensors and graph nodes. You must allocate enough memory upfront.
- **`ggml_tensor`**: The basic unit of data.
    - `type`: Data type (`GGML_TYPE_F32`, `GGML_TYPE_F16`, `GGML_TYPE_Q4_0`, etc.).
    - `ne[4]`: Number of elements in each dimension (up to 4 dimensions).
    - `data`: Pointer to the raw data.
- **`ggml_cgraph`**: A computation graph. It is a list of nodes (tensors) that need to be computed.

### What Workloads Fit ggml Well?

`ggml` is particularly well suited for:
- **Autoregressive language models**: Decoder-only transformers such as LLaMA, Mistral, Gemma, Qwen, etc.
- **Encoder/decoder transformers with mostly static shapes**: E.g. some translation or summarization models.
- **Small to medium-sized models on edge devices**: Where memory and compute are limited and quantization is needed.
- **Offline or embedded applications**: Desktop, mobile, or on-device assistants.

It is less ideal for:
- **Very large-scale training**: Multi-node training with advanced optimizers is outside its main scope.
- **Highly dynamic architectures**: Models that change their computation graph every step in complex ways.
- **Research that needs rapid prototyping**: Frameworks like PyTorch/JAX are still more convenient there.

### Using ggml Beyond LLMs

Although `ggml` became popular through `llama.cpp`, it is a general tensor engine. In principle, any model that can be expressed as a static computation graph with supported ops and tensor types can be run:
- **CNNs for classification** (e.g., ResNet-like).
- **UNet-style architectures** for image segmentation or denoising.
- **Object detection models** (YOLO-like) that combine convolutions with simple heads.
- **MLPs or small vision transformers**.

Typical workflow for non-LLM models:
1. Train the model in PyTorch/TF/JAX.
2. Export weights and structure to a custom format or directly to GGUF if you have a converter.
3. Rebuild the network in `ggml` using the same layers and parameter shapes.
4. Load weights into `ggml` tensors and run inference.

### Example: Matrix Multiplication

This example demonstrates how to perform $C = A \times B$.

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

### Example: Simple CNN Block (Conceptual)

Below is a conceptual sketch of how a small 2D convolution block could look in `ggml`. Real models will need careful layout and stride handling, but the overall pattern is:

1. Create tensors for input feature maps, convolution weights, and bias.
2. Use `ggml_conv_2d` (or related ops) to build the convolution node.
3. Optionally apply activation such as ReLU.
4. Build a graph and run it with `ggml_graph_compute_with_ctx`.

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

In a UNet-like model for semantic segmentation, you would stack many such blocks, combine them with downsampling (e.g., strided convolutions) and upsampling (e.g., transposed convolutions), and add skip connections. `ggml` expresses all of these as tensor operations connected in one graph.

### Example: YOLO-style Detection Head (Conceptual)

For YOLO-like models, the backbone can be implemented as a series of convolutional blocks as above. The detection head usually:
- Takes feature maps at multiple scales.
- Applies a small number of convolutions.
- Produces tensors containing class logits and bounding box parameters.

In `ggml`, a single-scale detection head might look like:

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

Post-processing (NMS, decoding bounding boxes from anchors, etc.) can be done in C/C++ outside `ggml`, using the raw tensor values.

### Broadcasting
`ggml` supports broadcasting. If you add a tensor of shape `[10]` to a tensor of shape `[10, 5]`, the first tensor is logically repeated 5 times.

### Backend Scheduler
For complex usage (like in `llama.cpp`), `ggml_backend_sched` is used to automatically distribute operations across CPU and GPU based on tensor location and operation support.

---

## 6. Conclusion

You now have a solid foundation for working with `llama.cpp` and `ggml`.

- **For Users**: Use `llama-cli` to run downloaded GGUF models efficiently.
- **For Developers**: Use `llama.h` to build chat applications, agents, or tools powered by LLMs.
- **For Researchers**: Use `ggml` to experiment with new tensor operations or model architectures.

Happy coding!
