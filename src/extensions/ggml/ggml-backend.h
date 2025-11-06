// Shim header to expose ggml-backend definitions through llama.cpp/include
// This forwards to the actual ggml backend header in the ggml/include directory.

#pragma once

// 指向第三方 llama.cpp 的 ggml 后端头文件（项目内统一转发）
#include "../../third_party/llama.cpp/ggml/include/ggml-backend.h"