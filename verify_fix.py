#!/usr/bin/env python3

def normalize_model_id(model_name):
    """
    Python版本的normalizeModelId函数，用于验证C++实现
    """
    model_id = model_name
    result = ""
    for c in model_id:
        if c.isalnum() or c in ['_', '-', '.']:
            result += c
        else:
            result += '_'
    return result

def test_normalization():
    """测试模型ID归一化"""
    test_cases = [
        "registry.ollama.ai/library/qwen2.5vl:7b",
        "registry.ollama.ai/rockn/Qwen2.5-Omni-7B-Q4_K_M:latest",
        "simple_model",
        "model-with-dashes",
        "model.with.dots",
        "model:with:colons",
        "model/with/slashes"
    ]
    
    print("Testing Model ID Normalization:")
    print("=" * 60)
    
    for original in test_cases:
        normalized = normalize_model_id(original)
        print(f"Original:   {original}")
        print(f"Normalized: {normalized}")
        print("-" * 40)
    
    # 特别测试我们关心的模型
    target_model = "registry.ollama.ai/library/qwen2.5vl:7b"
    expected_normalized = "registry.ollama.ai_library_qwen2.5vl_7b"  # 点号保留
    actual_normalized = normalize_model_id(target_model)
    
    print(f"\nTarget Model Test:")
    print(f"Original:   {target_model}")
    print(f"Expected:   {expected_normalized}")
    print(f"Actual:     {actual_normalized}")
    print(f"Match:      {expected_normalized == actual_normalized}")
    
    if expected_normalized == actual_normalized:
        print("✅ Model ID normalization is working correctly!")
    else:
        print("❌ Model ID normalization has issues!")

if __name__ == "__main__":
    test_normalization()