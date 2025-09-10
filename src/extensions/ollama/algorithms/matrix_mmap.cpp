#include "matrix_mmap.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <thread>

namespace duorou {
namespace extensions {
namespace ollama {
namespace algorithms {

// MatrixFile实现
struct MatrixFile::impl {
#ifdef _WIN32
    HANDLE fp_win32;
    
    impl(const char* fname, const char* mode) {
        DWORD access = 0;
        DWORD creation = 0;
        
        if (strchr(mode, 'r')) access |= GENERIC_READ;
        if (strchr(mode, 'w')) access |= GENERIC_WRITE;
        if (strchr(mode, 'w')) creation = CREATE_ALWAYS;
        else creation = OPEN_EXISTING;
        
        fp_win32 = CreateFileA(fname, access, FILE_SHARE_READ | FILE_SHARE_WRITE,
                              NULL, creation, FILE_ATTRIBUTE_NORMAL, NULL);
        if (fp_win32 == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("Failed to open file: " + std::string(fname));
        }
        
        LARGE_INTEGER file_size;
        if (!GetFileSizeEx(fp_win32, &file_size)) {
            CloseHandle(fp_win32);
            throw std::runtime_error("Failed to get file size");
        }
        size = static_cast<size_t>(file_size.QuadPart);
    }
    
    ~impl() {
        if (fp_win32 != INVALID_HANDLE_VALUE) {
            CloseHandle(fp_win32);
        }
    }
    
    size_t tell() const {
        LARGE_INTEGER pos;
        LARGE_INTEGER zero = {0};
        if (!SetFilePointerEx(fp_win32, zero, &pos, FILE_CURRENT)) {
            throw std::runtime_error("Failed to get file position");
        }
        return static_cast<size_t>(pos.QuadPart);
    }
    
    void seek(size_t offset, int whence) const {
        DWORD move_method;
        switch (whence) {
            case SEEK_SET: move_method = FILE_BEGIN; break;
            case SEEK_CUR: move_method = FILE_CURRENT; break;
            case SEEK_END: move_method = FILE_END; break;
            default: throw std::runtime_error("Invalid whence value");
        }
        
        LARGE_INTEGER li;
        li.QuadPart = offset;
        if (!SetFilePointerEx(fp_win32, li, NULL, move_method)) {
            throw std::runtime_error("Failed to seek file");
        }
    }
    
    void read_raw(void* ptr, size_t len) const {
        DWORD bytes_read;
        if (!ReadFile(fp_win32, ptr, static_cast<DWORD>(len), &bytes_read, NULL) ||
            bytes_read != len) {
            throw std::runtime_error("Failed to read from file");
        }
    }
    
    void write_raw(const void* ptr, size_t len) const {
        DWORD bytes_written;
        if (!WriteFile(fp_win32, ptr, static_cast<DWORD>(len), &bytes_written, NULL) ||
            bytes_written != len) {
            throw std::runtime_error("Failed to write to file");
        }
    }
#else
    FILE* fp;
    
    impl(const char* fname, const char* mode) {
        fp = fopen(fname, mode);
        if (!fp) {
            throw std::runtime_error("Failed to open file: " + std::string(fname) + 
                                   " (" + strerror(errno) + ")");
        }
        
        fseek(fp, 0, SEEK_END);
        size = ftell(fp);
        fseek(fp, 0, SEEK_SET);
    }
    
    ~impl() {
        if (fp) {
            fclose(fp);
        }
    }
    
    size_t tell() const {
        long pos = ftell(fp);
        if (pos < 0) {
            throw std::runtime_error("Failed to get file position");
        }
        return static_cast<size_t>(pos);
    }
    
    void seek(size_t offset, int whence) const {
        if (fseek(fp, static_cast<long>(offset), whence) != 0) {
            throw std::runtime_error("Failed to seek file");
        }
    }
    
    void read_raw(void* ptr, size_t len) const {
        if (len == 0) return;
        size_t ret = fread(ptr, len, 1, fp);
        if (ret != 1) {
            throw std::runtime_error("Failed to read from file: " + std::string(strerror(errno)));
        }
    }
    
    void write_raw(const void* ptr, size_t len) const {
        if (len == 0) return;
        size_t ret = fwrite(ptr, len, 1, fp);
        if (ret != 1) {
            throw std::runtime_error("Failed to write to file: " + std::string(strerror(errno)));
        }
    }
#endif
    
    size_t size;
};

MatrixFile::MatrixFile(const char* fname, const char* mode) 
    : pimpl(std::make_unique<impl>(fname, mode)) {}

MatrixFile::~MatrixFile() = default;

size_t MatrixFile::tell() const { return pimpl->tell(); }
size_t MatrixFile::size() const { return pimpl->size; }

int MatrixFile::file_id() const {
#ifdef _WIN32
    return _open_osfhandle(reinterpret_cast<intptr_t>(pimpl->fp_win32), 0);
#else
    return fileno(pimpl->fp);
#endif
}

void MatrixFile::seek(size_t offset, int whence) const { pimpl->seek(offset, whence); }
void MatrixFile::read_raw(void* ptr, size_t len) const { pimpl->read_raw(ptr, len); }

uint32_t MatrixFile::read_u32() const {
    uint32_t ret;
    read_raw(&ret, sizeof(ret));
    return ret;
}

void MatrixFile::write_raw(const void* ptr, size_t len) const { pimpl->write_raw(ptr, len); }
void MatrixFile::write_u32(uint32_t val) const { write_raw(&val, sizeof(val)); }

// MatrixMmap实现
struct MatrixMmap::impl {
#ifdef _POSIX_MAPPED_FILES
    std::vector<std::pair<size_t, size_t>> mapped_fragments;
    
    impl(MatrixFile* file, size_t prefetch, bool numa) {
        size = file->size();
        int fd = file->file_id();
        int flags = MAP_SHARED;
        if (numa) { prefetch = 0; }
        
#ifdef __linux__
        if (posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL)) {
            std::cerr << "Warning: posix_fadvise(.., POSIX_FADV_SEQUENTIAL) failed: " 
                     << strerror(errno) << std::endl;
        }
        if (prefetch) { flags |= MAP_POPULATE; }
#endif
        
        addr = mmap(NULL, file->size(), PROT_READ, flags, fd, 0);
        if (addr == MAP_FAILED) {
            throw std::runtime_error("mmap failed: " + std::string(strerror(errno)));
        }
        
        if (prefetch > 0) {
            if (posix_madvise(addr, std::min(file->size(), prefetch), POSIX_MADV_WILLNEED)) {
                std::cerr << "Warning: posix_madvise(.., POSIX_MADV_WILLNEED) failed: " 
                         << strerror(errno) << std::endl;
            }
        }
        if (numa) {
            if (posix_madvise(addr, file->size(), POSIX_MADV_RANDOM)) {
                std::cerr << "Warning: posix_madvise(.., POSIX_MADV_RANDOM) failed: " 
                         << strerror(errno) << std::endl;
            }
        }
        
        mapped_fragments.emplace_back(0, file->size());
    }
    
    void unmap_fragment(size_t first, size_t last) {
        int page_size = sysconf(_SC_PAGESIZE);
        
        // 对齐到页边界
        size_t offset_in_page = first & (page_size - 1);
        size_t offset_to_page = offset_in_page == 0 ? 0 : page_size - offset_in_page;
        first += offset_to_page;
        last = last & ~(page_size - 1);
        
        if (last <= first) {
            last = first;
            return;
        }
        
        size_t len = last - first;
        if (len == 0) return;
        
        void* next_page_start = static_cast<uint8_t*>(addr) + first;
        if (munmap(next_page_start, len)) {
            std::cerr << "Warning: munmap failed: " << strerror(errno) << std::endl;
        }
        
        // 更新映射片段
        std::vector<std::pair<size_t, size_t>> new_mapped_fragments;
        for (const auto& frag : mapped_fragments) {
            if (frag.first < first && frag.second > last) {
                new_mapped_fragments.emplace_back(frag.first, first);
                new_mapped_fragments.emplace_back(last, frag.second);
            } else if (frag.first < first && frag.second > first) {
                new_mapped_fragments.emplace_back(frag.first, first);
            } else if (frag.first < last && frag.second > last) {
                new_mapped_fragments.emplace_back(last, frag.second);
            } else if (frag.first >= first && frag.second <= last) {
                // 完全在卸载范围内，跳过
            } else {
                new_mapped_fragments.push_back(frag);
            }
        }
        mapped_fragments = std::move(new_mapped_fragments);
    }
    
    ~impl() {
        for (const auto& frag : mapped_fragments) {
            if (munmap(static_cast<char*>(addr) + frag.first, frag.second - frag.first)) {
                std::cerr << "Warning: munmap failed: " << strerror(errno) << std::endl;
            }
        }
    }
#elif defined(_WIN32)
    impl(MatrixFile* file, size_t prefetch, bool numa) {
        size = file->size();
        
        HANDLE hFile = reinterpret_cast<HANDLE>(_get_osfhandle(file->file_id()));
        HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        
        if (hMapping == NULL) {
            DWORD error = GetLastError();
            throw std::runtime_error("CreateFileMappingA failed: " + std::to_string(error));
        }
        
        addr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
        DWORD error = GetLastError();
        CloseHandle(hMapping);
        
        if (addr == NULL) {
            throw std::runtime_error("MapViewOfFile failed: " + std::to_string(error));
        }
        
        // Windows预取支持
        if (prefetch > 0) {
#if _WIN32_WINNT >= 0x602
            BOOL (WINAPI *pPrefetchVirtualMemory)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
            HMODULE hKernel32 = GetModuleHandleW(L"kernel32.dll");
            
            pPrefetchVirtualMemory = reinterpret_cast<decltype(pPrefetchVirtualMemory)>(
                GetProcAddress(hKernel32, "PrefetchVirtualMemory"));
            
            if (pPrefetchVirtualMemory) {
                WIN32_MEMORY_RANGE_ENTRY range;
                range.VirtualAddress = addr;
                range.NumberOfBytes = static_cast<SIZE_T>(std::min(size, prefetch));
                if (!pPrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0)) {
                    std::cerr << "Warning: PrefetchVirtualMemory failed: " 
                             << GetLastError() << std::endl;
                }
            }
#endif
        }
    }
    
    void unmap_fragment(size_t first, size_t last) {
        // Windows不支持部分卸载
    }
    
    ~impl() {
        if (!UnmapViewOfFile(addr)) {
            std::cerr << "Warning: UnmapViewOfFile failed: " << GetLastError() << std::endl;
        }
    }
#else
    impl(MatrixFile* file, size_t prefetch, bool numa) {
        throw std::runtime_error("mmap not supported on this platform");
    }
    
    void unmap_fragment(size_t first, size_t last) {
        throw std::runtime_error("mmap not supported on this platform");
    }
#endif
    
    void* addr;
    size_t size;
};

MatrixMmap::MatrixMmap(MatrixFile* file, size_t prefetch, bool numa)
    : pimpl(std::make_unique<impl>(file, prefetch, numa)) {}

MatrixMmap::~MatrixMmap() = default;

size_t MatrixMmap::size() const { return pimpl->size; }
void* MatrixMmap::addr() const { return pimpl->addr; }
void MatrixMmap::unmap_fragment(size_t first, size_t last) { pimpl->unmap_fragment(first, last); }

#if defined(_POSIX_MAPPED_FILES) || defined(_WIN32)
const bool MatrixMmap::SUPPORTED = true;
#else
const bool MatrixMmap::SUPPORTED = false;
#endif

// MatrixMlock实现
struct MatrixMlock::impl {
#ifdef _POSIX_MEMLOCK_RANGE
    static size_t lock_granularity() {
        return static_cast<size_t>(sysconf(_SC_PAGESIZE));
    }
    
    bool raw_lock(const void* addr, size_t size) const {
        return mlock(addr, size) == 0;
    }
    
    static void raw_unlock(void* addr, size_t size) {
        munlock(addr, size);
    }
#elif defined(_WIN32)
    static size_t lock_granularity() {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return static_cast<size_t>(si.dwPageSize);
    }
    
    bool raw_lock(void* ptr, size_t len) const {
        return VirtualLock(ptr, len) != 0;
    }
    
    static void raw_unlock(void* ptr, size_t len) {
        VirtualUnlock(ptr, len);
    }
#else
    static size_t lock_granularity() {
        return 65536; // 64KB默认页大小
    }
    
    bool raw_lock(const void* addr, size_t len) const {
        return false; // 不支持
    }
    
    static void raw_unlock(void* addr, size_t len) {
        // 不支持
    }
#endif
    
    void init(void* ptr) {
        addr = ptr;
        size = 0;
        failed_already = false;
    }
    
    void grow_to(size_t target_size) {
        if (failed_already) return;
        
        size_t granularity = lock_granularity();
        target_size = (target_size + granularity - 1) & ~(granularity - 1);
        
        if (target_size > size) {
            if (raw_lock(static_cast<uint8_t*>(addr) + size, target_size - size)) {
                size = target_size;
            } else {
                failed_already = true;
            }
        }
    }
    
    ~impl() {
        if (size > 0) {
            raw_unlock(addr, size);
        }
    }
    
    void* addr;
    size_t size;
    bool failed_already;
};

MatrixMlock::MatrixMlock() : pimpl(std::make_unique<impl>()) {}
MatrixMlock::~MatrixMlock() = default;

void MatrixMlock::init(void* ptr) { pimpl->init(ptr); }
void MatrixMlock::grow_to(size_t target_size) { pimpl->grow_to(target_size); }

#if defined(_POSIX_MEMLOCK_RANGE) || defined(_WIN32)
const bool MatrixMlock::SUPPORTED = true;
#else
const bool MatrixMlock::SUPPORTED = false;
#endif

// MmapMatrixOperations实现
MmapMatrixOperations::MmapMatrixOperations(bool verbose)
    : verbose_(verbose), total_mapped_size_(0), total_locked_size_(0) {}

MmapMatrixOperations::~MmapMatrixOperations() {
    unmapAll();
}

bool MmapMatrixOperations::initialize(const ModelConfig& config, const AlgorithmContext& context) {
    context_ = context;
    log("INFO", "MmapMatrixOperations initialized");
    return true;
}

bool MmapMatrixOperations::validateInput(const Tensor& input) const {
    return !input.data.empty() && input.size > 0;
}

void MmapMatrixOperations::multiply(const float* a, const float* b, float* c,
                                   size_t m, size_t n, size_t k) {
    // 标准矩阵乘法实现
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t l = 0; l < k; ++l) {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

void MmapMatrixOperations::vectorAdd(const float* a, const float* b, float* result, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void MmapMatrixOperations::vectorMul(const float* a, const float* b, float* result, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

bool MmapMatrixOperations::loadMatrixFromFile(const std::string& filepath, const std::string& matrix_name) {
    try {
        auto mapping = std::make_unique<MatrixMapping>();
        mapping->file = std::make_unique<MatrixFile>(filepath.c_str(), "rb");
        
        // 解析矩阵头部
        if (!parseMatrixHeader(mapping->file.get(), mapping->data)) {
            log("ERROR", "Failed to parse matrix header for: " + matrix_name);
            return false;
        }
        
        // 创建内存映射
        mapping->mmap = std::make_unique<MatrixMmap>(mapping->file.get());
        mapping->data.data_ptr = mapping->mmap->addr();
        
        total_mapped_size_ += mapping->data.total_size;
        mappings_[matrix_name] = std::move(mapping);
        
        log("INFO", "Successfully loaded matrix: " + matrix_name + 
            " (" + std::to_string(mapping->data.rows) + "x" + 
            std::to_string(mapping->data.cols) + ")");
        
        return true;
    } catch (const std::exception& e) {
        log("ERROR", "Failed to load matrix " + matrix_name + ": " + e.what());
        return false;
    }
}

bool MmapMatrixOperations::saveMatrixToFile(const std::string& filepath, const std::string& matrix_name,
                                           const float* data, size_t rows, size_t cols) {
    try {
        MatrixFile file(filepath.c_str(), "wb");
        
        MmapMatrixData matrix_data;
        matrix_data.rows = rows;
        matrix_data.cols = cols;
        matrix_data.element_size = sizeof(float);
        matrix_data.total_size = rows * cols * sizeof(float);
        matrix_data.dtype = "F32";
        
        writeMatrixHeader(&file, matrix_data);
        file.write_raw(data, matrix_data.total_size);
        
        log("INFO", "Successfully saved matrix: " + matrix_name);
        return true;
    } catch (const std::exception& e) {
        log("ERROR", "Failed to save matrix " + matrix_name + ": " + e.what());
        return false;
    }
}

const MmapMatrixData* MmapMatrixOperations::getMappedMatrix(const std::string& name) const {
    auto it = mappings_.find(name);
    if (it != mappings_.end()) {
        return &it->second->data;
    }
    return nullptr;
}

void MmapMatrixOperations::multiplyMapped(const std::string& matrix_a_name,
                                         const std::string& matrix_b_name,
                                         float* result, size_t result_rows, size_t result_cols) {
    const auto* matrix_a = getMappedMatrix(matrix_a_name);
    const auto* matrix_b = getMappedMatrix(matrix_b_name);
    
    if (!matrix_a || !matrix_b) {
        throw std::runtime_error("Matrix not found in mappings");
    }
    
    if (matrix_a->cols != matrix_b->rows) {
        throw std::runtime_error("Matrix dimensions mismatch for multiplication");
    }
    
    const float* a_data = static_cast<const float*>(matrix_a->data_ptr);
    const float* b_data = static_cast<const float*>(matrix_b->data_ptr);
    
    multiply(a_data, b_data, result, matrix_a->rows, matrix_b->cols, matrix_a->cols);
}

void MmapMatrixOperations::prefetchMatrix(const std::string& matrix_name, size_t size) {
    auto it = mappings_.find(matrix_name);
    if (it != mappings_.end()) {
        void* addr = it->second->data.data_ptr;
        size_t prefetch_size = size > 0 ? size : it->second->data.total_size;
        
#ifdef __linux__
        if (posix_madvise(addr, prefetch_size, POSIX_MADV_WILLNEED)) {
            log("WARNING", "Failed to prefetch matrix: " + matrix_name);
        }
#endif
    }
}

bool MmapMatrixOperations::lockMatrix(const std::string& matrix_name) {
    auto it = mappings_.find(matrix_name);
    if (it != mappings_.end() && !it->second->is_locked) {
        if (MatrixMlock::SUPPORTED) {
            it->second->mlock = std::make_unique<MatrixMlock>();
            it->second->mlock->init(it->second->data.data_ptr);
            it->second->mlock->grow_to(it->second->data.total_size);
            it->second->is_locked = true;
            total_locked_size_ += it->second->data.total_size;
            return true;
        }
    }
    return false;
}

void MmapMatrixOperations::unlockMatrix(const std::string& matrix_name) {
    auto it = mappings_.find(matrix_name);
    if (it != mappings_.end() && it->second->is_locked) {
        it->second->mlock.reset();
        it->second->is_locked = false;
        total_locked_size_ -= it->second->data.total_size;
    }
}

size_t MmapMatrixOperations::getTotalMappedSize() const {
    return total_mapped_size_;
}

size_t MmapMatrixOperations::getLockedSize() const {
    return total_locked_size_;
}

void MmapMatrixOperations::unmapMatrix(const std::string& matrix_name) {
    auto it = mappings_.find(matrix_name);
    if (it != mappings_.end()) {
        if (it->second->is_locked) {
            total_locked_size_ -= it->second->data.total_size;
        }
        total_mapped_size_ -= it->second->data.total_size;
        mappings_.erase(it);
        log("INFO", "Unmapped matrix: " + matrix_name);
    }
}

void MmapMatrixOperations::unmapAll() {
    mappings_.clear();
    total_mapped_size_ = 0;
    total_locked_size_ = 0;
    log("INFO", "All matrices unmapped");
}

bool MmapMatrixOperations::parseMatrixHeader(MatrixFile* file, MmapMatrixData& data) {
    try {
        // 简单的二进制格式：rows(8字节) + cols(8字节) + dtype_len(4字节) + dtype + data
        file->seek(0, SEEK_SET);
        
        uint64_t rows, cols;
        file->read_raw(&rows, sizeof(rows));
        file->read_raw(&cols, sizeof(cols));
        
        uint32_t dtype_len = file->read_u32();
        std::vector<char> dtype_buf(dtype_len + 1, 0);
        file->read_raw(dtype_buf.data(), dtype_len);
        
        data.rows = rows;
        data.cols = cols;
        data.dtype = std::string(dtype_buf.data());
        data.element_size = (data.dtype == "F32") ? 4 : (data.dtype == "F16") ? 2 : 4;
        data.total_size = rows * cols * data.element_size;
        
        return true;
    } catch (const std::exception& e) {
        log("ERROR", "Failed to parse matrix header: " + std::string(e.what()));
        return false;
    }
}

void MmapMatrixOperations::writeMatrixHeader(MatrixFile* file, const MmapMatrixData& data) {
    uint64_t rows = data.rows;
    uint64_t cols = data.cols;
    uint32_t dtype_len = static_cast<uint32_t>(data.dtype.length());
    
    file->write_raw(&rows, sizeof(rows));
    file->write_raw(&cols, sizeof(cols));
    file->write_u32(dtype_len);
    file->write_raw(data.dtype.c_str(), dtype_len);
}

void MmapMatrixOperations::convertToFloat(const void* src, float* dst, size_t count, const std::string& dtype) {
    if (dtype == "F32") {
        std::memcpy(dst, src, count * sizeof(float));
    } else if (dtype == "F16") {
        // 简化的F16到F32转换
        const uint16_t* src16 = static_cast<const uint16_t*>(src);
        for (size_t i = 0; i < count; ++i) {
            // 这里应该实现正确的F16到F32转换
            dst[i] = static_cast<float>(src16[i]) / 65536.0f;
        }
    }
}

void MmapMatrixOperations::convertFromFloat(const float* src, void* dst, size_t count, const std::string& dtype) {
    if (dtype == "F32") {
        std::memcpy(dst, src, count * sizeof(float));
    } else if (dtype == "F16") {
        // 简化的F32到F16转换
        uint16_t* dst16 = static_cast<uint16_t*>(dst);
        for (size_t i = 0; i < count; ++i) {
            dst16[i] = static_cast<uint16_t>(src[i] * 65536.0f);
        }
    }
}

void MmapMatrixOperations::log(const std::string& level, const std::string& message) const {
    if (verbose_) {
        std::cout << "[" << level << "] MmapMatrixOperations: " << message << std::endl;
    }
}

size_t MmapMatrixOperations::alignSize(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

void MmapMatrixOperations::optimizeForNuma(void* ptr, size_t size) {
#ifdef __linux__
    // NUMA优化提示
    if (posix_madvise(ptr, size, POSIX_MADV_RANDOM)) {
        log("WARNING", "Failed to set NUMA optimization hints");
    }
#endif
}

// 工厂函数
std::unique_ptr<MmapMatrixOperations> createMmapMatrixOperations(bool verbose) {
    return std::make_unique<MmapMatrixOperations>(verbose);
}

} // namespace algorithms
} // namespace ollama
} // namespace extensions
} // namespace duorou