#!/usr/bin/env python3
import struct
import sys

def read_string(f):
    length = struct.unpack('<Q', f.read(8))[0]
    if length > 10000:  # Sanity check
        print(f"Warning: String length {length} seems too large, file may be corrupted")
        return "<CORRUPTED>"
    data = f.read(length)
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        return data.decode('utf-8', errors='replace')

def read_gguf_metadata(filename):
    with open(filename, 'rb') as f:
        # Read header
        magic = struct.unpack('<I', f.read(4))[0]
        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        kv_count = struct.unpack('<Q', f.read(8))[0]
        
        print(f"Magic: {hex(magic)}, Version: {version}")
        print(f"Tensor count: {tensor_count}, KV count: {kv_count}")
        print("\nMetadata keys:")
        
        # Read metadata
        for i in range(min(kv_count, 50)):  # Limit to first 50 to avoid issues
            try:
                key = read_string(f)
                if key == "<CORRUPTED>":
                    print(f"  {i+1:2d}. Corrupted key, stopping")
                    break
                    
                type_value = struct.unpack('<I', f.read(4))[0]
                
                print(f"  {i+1:2d}. Key: '{key}', Type: {type_value}", end="")
                
                if key == "tokenizer.ggml.pre":
                    print(f" *** FOUND tokenizer.ggml.pre with type {type_value} ***")
                else:
                    print()
                
                # Skip the value data based on type
                if type_value == 8:  # STRING
                    value_len = struct.unpack('<Q', f.read(8))[0]
                    if value_len > 10000:
                        print(f"      Warning: String value length {value_len} too large, skipping")
                        break
                    value = f.read(value_len)
                    if key == "tokenizer.ggml.pre":
                        try:
                            print(f"      Value: '{value.decode('utf-8')}'")
                        except UnicodeDecodeError:
                            print(f"      Value (hex): {value.hex()}")
                elif type_value == 9:  # ARRAY
                    array_type = struct.unpack('<I', f.read(4))[0]
                    array_len = struct.unpack('<Q', f.read(8))[0]
                    if key == "tokenizer.ggml.pre":
                        print(f"      Array type: {array_type}, Array length: {array_len}")
                    # Skip array data safely
                    if array_len > 100000:
                        print(f"      Warning: Array length {array_len} too large, stopping")
                        break
                    if array_type == 8:  # STRING array
                        for j in range(array_len):
                            str_len = struct.unpack('<Q', f.read(8))[0]
                            if str_len > 10000:
                                print(f"      Warning: String in array too large, stopping")
                                return
                            f.read(str_len)
                    else:
                        # For other types, calculate size
                        type_sizes = {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 4, 6: 4, 7: 4, 8: 8, 9: 8, 10: 8}
                        if array_type in type_sizes:
                            f.read(array_len * type_sizes[array_type])
                else:
                    # Handle other basic types
                    type_sizes = {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 4, 6: 4, 7: 4, 8: 8, 9: 8, 10: 8}
                    if type_value in type_sizes:
                        f.read(type_sizes[type_value])
            except Exception as e:
                print(f"  Error reading key {i+1}: {e}")
                break

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 debug_gguf.py <gguf_file>")
        sys.exit(1)
    
    read_gguf_metadata(sys.argv[1])