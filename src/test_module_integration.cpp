#include <cassert>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// 包含 ollama 扩展的头文件
#include "extensions/ollama/gguf_parser.h"
#include "extensions/ollama/ollama_model_manager.h"
#include "extensions/ollama/ollama_path_resolver.h"

// 为了测试在没有实际模型文件的情况下运行
#include "core/text_generator.h"
#include "kvcache/cache.h"
#include "ml/context.h"
#include "model/qwen_text_model.h"

using namespace duorou::extensions::ollama;
