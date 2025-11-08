#include "object_store.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <string>
#include <chrono>
#include <cstdint>

// Minimal SHA256 implementation (compact, self-contained)
namespace mini_sha256 {
  struct SHA256Ctx {
    uint32_t state[8];
    uint64_t bitlen;
    unsigned char data[64];
    size_t datalen;
  };

  static inline uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }
  static inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
  static inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
  static inline uint32_t ep0(uint32_t x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
  static inline uint32_t ep1(uint32_t x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
  static inline uint32_t sig0(uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
  static inline uint32_t sig1(uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

  static const uint32_t k[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
  };

  static void init(SHA256Ctx &ctx) {
    ctx.datalen = 0; ctx.bitlen = 0;
    ctx.state[0] = 0x6a09e667; ctx.state[1] = 0xbb67ae85; ctx.state[2] = 0x3c6ef372; ctx.state[3] = 0xa54ff53a;
    ctx.state[4] = 0x510e527f; ctx.state[5] = 0x9b05688c; ctx.state[6] = 0x1f83d9ab; ctx.state[7] = 0x5be0cd19;
  }

  static void transform(SHA256Ctx &ctx, const unsigned char data[]) {
    uint32_t m[64];
    for (int i = 0; i < 16; ++i) {
      m[i] = (data[i * 4] << 24) | (data[i * 4 + 1] << 16) | (data[i * 4 + 2] << 8) | (data[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      m[i] = sig1(m[i - 2]) + m[i - 7] + sig0(m[i - 15]) + m[i - 16];
    }

    uint32_t a = ctx.state[0], b = ctx.state[1], c = ctx.state[2], d = ctx.state[3];
    uint32_t e = ctx.state[4], f = ctx.state[5], g = ctx.state[6], h = ctx.state[7];

    for (int i = 0; i < 64; ++i) {
      uint32_t t1 = h + ep1(e) + ch(e, f, g) + k[i] + m[i];
      uint32_t t2 = ep0(a) + maj(a, b, c);
      h = g; g = f; f = e; e = d + t1;
      d = c; c = b; b = a; a = t1 + t2;
    }

    ctx.state[0] += a; ctx.state[1] += b; ctx.state[2] += c; ctx.state[3] += d;
    ctx.state[4] += e; ctx.state[5] += f; ctx.state[6] += g; ctx.state[7] += h;
  }

  static void update(SHA256Ctx &ctx, const unsigned char *data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
      ctx.data[ctx.datalen] = data[i];
      ctx.datalen++;
      if (ctx.datalen == 64) {
        transform(ctx, ctx.data);
        ctx.bitlen += 512;
        ctx.datalen = 0;
      }
    }
  }

  static void final(SHA256Ctx &ctx, unsigned char hash[32]) {
    size_t i = ctx.datalen;
    // Pad
    if (ctx.datalen < 56) {
      ctx.data[i++] = 0x80;
      while (i < 56) ctx.data[i++] = 0x00;
    } else {
      ctx.data[i++] = 0x80;
      while (i < 64) ctx.data[i++] = 0x00;
      transform(ctx, ctx.data);
      for (size_t j = 0; j < 64; ++j) ctx.data[j] = 0;
    }
    ctx.bitlen += ctx.datalen * 8;
    ctx.data[63] = ctx.bitlen & 0xFF;
    ctx.data[62] = (ctx.bitlen >> 8) & 0xFF;
    ctx.data[61] = (ctx.bitlen >> 16) & 0xFF;
    ctx.data[60] = (ctx.bitlen >> 24) & 0xFF;
    ctx.data[59] = (ctx.bitlen >> 32) & 0xFF;
    ctx.data[58] = (ctx.bitlen >> 40) & 0xFF;
    ctx.data[57] = (ctx.bitlen >> 48) & 0xFF;
    ctx.data[56] = (ctx.bitlen >> 56) & 0xFF;
    transform(ctx, ctx.data);

    for (i = 0; i < 8; ++i) {
      hash[i * 4]     = (ctx.state[i] >> 24) & 0xFF;
      hash[i * 4 + 1] = (ctx.state[i] >> 16) & 0xFF;
      hash[i * 4 + 2] = (ctx.state[i] >> 8) & 0xFF;
      hash[i * 4 + 3] = (ctx.state[i]) & 0xFF;
    }
  }
}

namespace duorou {
namespace utils {

namespace fs = std::filesystem;

static std::string join_path(const std::string &a, const std::string &b) {
  fs::path pa(a);
  fs::path pb(b);
  return (pa / pb).string();
}

static std::string get_home_dir() {
#ifdef _WIN32
  const char *home = std::getenv("USERPROFILE");
  if (!home) {
    const char *drive = std::getenv("HOMEDRIVE");
    const char *path  = std::getenv("HOMEPATH");
    if (drive && path) {
      return std::string(drive) + std::string(path);
    }
  }
#else
  const char *home = std::getenv("HOME");
#endif
  return home ? std::string(home) : std::string(".");
}

std::string ObjectStore::objects_dir() {
  fs::path base = get_home_dir();
  fs::path dir = base / ".duorou" / "objects";
  std::error_code ec;
  fs::create_directories(dir, ec);
  return dir.string();
}

static std::string file_extension(const std::string &path) {
  fs::path p(path);
  std::string ext = p.extension().string();
  return ext;
}

static std::string sha256_file(const std::string &path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs.is_open()) return "";

  mini_sha256::SHA256Ctx ctx; mini_sha256::init(ctx);
  unsigned char buf[8192];
  while (ifs.good()) {
    ifs.read(reinterpret_cast<char *>(buf), sizeof(buf));
    std::streamsize got = ifs.gcount();
    if (got > 0) {
      mini_sha256::update(ctx, buf, static_cast<size_t>(got));
    }
  }
  unsigned char hash[32];
  mini_sha256::final(ctx, hash);

  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (int i = 0; i < 32; ++i) {
    oss << std::setw(2) << static_cast<int>(hash[i]);
  }
  return oss.str();
}

std::string ObjectStore::store_file(const std::string &src_path) {
  if (src_path.empty()) return "";
  std::string id = sha256_file(src_path);
  if (id.empty()) {
    // Fallback: simple pseudo-random id if hashing failed
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    auto seed = static_cast<unsigned long long>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    for (int i = 0; i < 8; ++i) {
      unsigned int v = static_cast<unsigned int>((seed >> (i * 8)) & 0xFFull);
      oss << std::setw(2) << v;
    }
    id = oss.str();
  }
  std::string ext = file_extension(src_path);
  std::string dest_dir = objects_dir();
  std::string dest_path = join_path(dest_dir, id + ext);

  // If file already exists (same hash), skip copy
  std::error_code ec;
  if (!fs::exists(dest_path)) {
    fs::copy_file(src_path, dest_path, fs::copy_options::overwrite_existing, ec);
  }

  return dest_path;
}

std::string ObjectStore::to_file_uri(const std::string &path) {
  if (path.empty()) return "";
  fs::path p = fs::absolute(fs::path(path));
#ifdef _WIN32
  // Use generic form (forward slashes), and add triple slash
  std::string s = p.generic_u8string();
  return std::string("file:///") + s;
#else
  std::string s = p.u8string();
  return std::string("file://") + s;
#endif
}

} // namespace utils
} // namespace duorou