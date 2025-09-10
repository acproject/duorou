#!/usr/bin/env python3
from transformers import AutoTokenizer

# 测试token列表
ids = [198, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]

print("Loading Qwen tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("/Users/acproject/workspace/cpp_projects/duorou/models/Qwen2.5-VL-7B-Instruct")

print("\n=== Individual Token Analysis ===")
for i, token_id in enumerate(ids[:10]):
    try:
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        print(f"Token {i}: {token_id} -> '{decoded}' (bytes: {[hex(ord(c)) for c in decoded]})")
    except Exception as e:
        print(f"Token {i}: {token_id} -> ERROR: {e}")

print("\n=== Full Sequence Decode ===")
try:
    full_decoded = tokenizer.decode(ids, skip_special_tokens=False)
    print(f"Full decode: '{full_decoded}'")
    print(f"Length: {len(full_decoded)} characters")
    
    # 检查是否包含替换字符
    replacement_chars = full_decoded.count('\ufffd')
    if replacement_chars > 0:
        print(f"WARNING: Contains {replacement_chars} UTF-8 replacement characters")
    else:
        print("SUCCESS: No UTF-8 replacement characters found")
        
except Exception as e:
    print(f"Full decode ERROR: {e}")

print("\n=== Byte-level Analysis ===")
for i, token_id in enumerate(ids[:5]):
    try:
        # 获取token的原始字符串表示
        token_str = tokenizer.convert_ids_to_tokens([token_id])[0]
        print(f"Token {i}: {token_id} -> token_str: '{token_str}'")
    except Exception as e:
        print(f"Token {i}: {token_id} -> token_str ERROR: {e}")