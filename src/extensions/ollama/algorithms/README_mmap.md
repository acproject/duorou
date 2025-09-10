# Memory-Mapped Matrix Operations

基于 llama.cpp mmap 机制的内存映射矩阵操作实现，提供高效的大矩阵处理能力。

## 特性

- **零拷贝操作**: 直接在映射内存上进行矩阵运算，避免数据复制
- **跨平台支持**: 支持 Linux、macOS 和 Windows 平台
- **内存锁定**: 支持将关键矩阵锁定在物理内存中
- **NUMA 优化**: 在支持的平台上进行 NUMA 内存优化
- **类型转换**: 支持多种数据类型（F32、F16、BF16等）的自动转换

## 文件结构

```
matrix_mmap.h              # 头文件定义
matrix_mmap.cpp            # 实现文件
test_matrix_mmap.cpp       # 测试和示例代码
```

## 核心类

### MatrixFile
跨平台文件操作封装，支持大文件读写。

### MatrixMmap
内存映射管理，基于 llama.cpp 的实现：
- Linux/macOS: 使用 `mmap()` 系统调用
- Windows: 使用 `CreateFileMapping()` 和 `MapViewOfFile()`

### MatrixMlock
内存锁定管理，防止重要数据被交换到磁盘：
- Linux/macOS: 使用 `mlock()` 系统调用
- Windows: 使用 `VirtualLock()`

### MmapMatrixOperations
主要的矩阵操作类，实现 `IMatrixAlgorithm` 接口。

## 使用示例

### 基本使用

```cpp
#include "matrix_mmap.h"

using namespace duorou::extensions::ollama::algorithms;

// 创建 mmap 矩阵操作实例
auto mmap_ops = createMmapMatrixOperations(true); // 启用详细日志

// 初始化
ModelConfig config;
AlgorithmContext context;
context.device = "cpu";
context.num_threads = 4;

mmap_ops->initialize(config, context);
```

### 加载和使用矩阵文件

```cpp
// 从文件加载矩阵（零拷贝）
if (mmap_ops->loadMatrixFromFile("/path/to/matrix.bin", "my_matrix")) {
    // 获取映射的矩阵信息
    const auto* matrix_data = mmap_ops->getMappedMatrix("my_matrix");
    
    std::cout << "Matrix: " << matrix_data->rows << "x" << matrix_data->cols 
              << ", type: " << matrix_data->dtype << std::endl;
    
    // 锁定矩阵到物理内存
    mmap_ops->lockMatrix("my_matrix");
    
    // 使用映射的矩阵进行计算...
    
    // 解锁和清理
    mmap_ops->unlockMatrix("my_matrix");
    mmap_ops->unmapMatrix("my_matrix");
}
```

### 矩阵乘法

```cpp
// 标准矩阵乘法
std::vector<float> a(m * k), b(k * n), c(m * n);
// ... 填充数据 ...

mmap_ops->multiply(a.data(), b.data(), c.data(), m, n, k);

// 零拷贝矩阵乘法（使用映射的矩阵）
mmap_ops->multiplyMapped("matrix_a", "matrix_b", result.data(), m, n);
```

### 向量操作

```cpp
std::vector<float> a(size), b(size), result(size);
// ... 填充数据 ...

// 向量加法
mmap_ops->vectorAdd(a.data(), b.data(), result.data(), size);

// 向量乘法
mmap_ops->vectorMul(a.data(), b.data(), result.data(), size);
```

## 矩阵文件格式

支持自定义的二进制矩阵格式：

```
[Header]
- Magic Number (4 bytes): 0x4D4D4150 ("MMAP")
- Version (4 bytes): 1
- Rows (8 bytes): 矩阵行数
- Cols (8 bytes): 矩阵列数
- Data Type Length (4 bytes): 数据类型字符串长度
- Data Type (variable): 数据类型字符串 ("F32", "F16", "BF16", etc.)

[Data]
- Matrix Data: 按行优先顺序存储的矩阵数据
```

## 性能优化

1. **内存预取**: 自动预取即将访问的内存页
2. **NUMA 感知**: 在多 NUMA 节点系统上优化内存分配
3. **页面对齐**: 确保内存映射按页面边界对齐
4. **分片解锁**: 支持部分内存区域的精确解锁

## 平台支持

| 平台 | mmap 支持 | mlock 支持 | NUMA 优化 |
|------|-----------|------------|----------|
| Linux | ✅ | ✅ | ✅ |
| macOS | ✅ | ✅ | ❌ |
| Windows | ✅ | ✅ | ❌ |

## 编译要求

- C++11 或更高版本
- 支持的编译器：GCC 4.8+, Clang 3.4+, MSVC 2015+

### Linux/macOS 编译选项
```bash
g++ -std=c++11 -O3 -march=native -fopenmp matrix_mmap.cpp test_matrix_mmap.cpp -o test_mmap
```

### Windows 编译选项
```cmd
cl /std:c++11 /O2 /openmp matrix_mmap.cpp test_matrix_mmap.cpp /Fe:test_mmap.exe
```

## 测试

运行测试程序：

```bash
./test_mmap
```

测试包括：
- 基本 mmap 操作
- 矩阵文件加载和保存
- 矩阵乘法性能测试
- 向量操作测试

## 注意事项

1. **文件大小限制**: 受系统虚拟内存限制
2. **内存锁定限制**: 受系统 `ulimit -l` 设置限制
3. **并发安全**: 当前实现不是线程安全的，需要外部同步
4. **错误处理**: 建议检查所有返回值和异常

## 与 llama.cpp 的兼容性

本实现基于 llama.cpp 的 mmap 机制，保持了以下兼容性：
- 相同的内存映射策略
- 相同的平台抽象层
- 相同的错误处理模式
- 相同的性能优化技术

这确保了与 llama.cpp 生态系统的良好集成。