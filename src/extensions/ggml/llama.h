// Shim header to expose llama.h through project include tree
// This forwards to the actual llama header in the third_party directory.

#pragma once

// 指向第三方 llama.cpp 的主头文件（项目内统一转发）
#include "../../third_party/llama.cpp/include/llama.h"