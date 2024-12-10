#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <sys/mman.h>

#include <algorithm>
#include <tuple>

#ifdef NEIGHBOR_HASH_SIMD_FIND
#include <immintrin.h>
#endif  // NEIGHBOR_HASH_SIMD_FIND

namespace neighbor {
namespace policy {

struct DefaultIntegerHash {
  static constexpr uint64_t kMul = 0x9ddfea08eb382d6;

  uint64_t operator()(uint64_t key) const {
    key *= kMul;
    return static_cast<uint64_t>(key ^ (key >> (sizeof(key) * 8 / 2)));
  }

  uint32_t operator()(uint32_t key) const {
    key *= kMul;
    return static_cast<uint32_t>(key ^ (key >> (sizeof(key) * 8 / 2)));
  }

#ifdef NEIGHBOR_HASH_SIMD_FIND
  __m512i v_hash_64(__m512i key) const {
    __m512i mul = _mm512_mullo_epi64(_mm512_set1_epi64(kMul), key);
    return _mm512_xor_epi64(
        mul, _mm512_srli_epi64(mul, (sizeof(uint64_t) * 8 / 2)));
  }
#endif  // NEIGHBOR_HASH_SIMD_FIND
};

template <class K, class V>
struct DefaultPolicy {
  static constexpr size_t kPayloadBitCount = sizeof(V) * 8 - 12;
  static constexpr size_t kHugePageSize = 2UL * 1024 * 1024;

  static size_t NormalizeCapacity(size_t n) {
    size_t power = 1;
    while (power < n) {
      power <<= 1;
    }
    return power;
  }

  static bool ShouldGrow(size_t size, size_t capactiy) {
    return size >= capactiy * 0.81;
  }

  static size_t GrowthSize(size_t n) { return std::max(n * 2, 1024UL); }

#ifdef NEIGHBOR_HASH_HUGEPAGE
  static size_t round_to_huge_page_size(size_t n) {
    return (((n - 1) / kHugePageSize) + 1) * kHugePageSize;
  }

  template <size_t alignment = 64>
  static void* Allocate(size_t size) {
    void* addr = mmap(0, round_to_huge_page_size(size), PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, 0, 0);
    if (addr == MAP_FAILED) {
      abort();
    }
    // std::cout << "Returned address is:" << addr << " size:" << size << std::endl;
    return addr;
  }

  static void Deallocate(void* ptr, size_t size) {
    munmap(ptr, round_to_huge_page_size(size));
  }
#else   // NEIGHBOR_HASH_HUGEPAGE
  template <size_t alignment = 64>
  static void* Allocate(size_t size) {
    auto* p = std::aligned_alloc(
        alignment, (size + alignment - 1) / alignment * alignment);
    return p;
  }

  static void Deallocate(void* ptr, size_t) { free(ptr); }
#endif  // NEIGHBOR_HASH_HUGEPAGE

  using Hash = DefaultIntegerHash;

  struct Hash2Slot {
    uint64_t operator()(size_t hash, size_t capacity) const {
      return hash & (capacity - 1);
    }

#ifdef NEIGHBOR_HASH_SIMD_FIND
    __m512i v_hash2slot(__m512i data, size_t capacity) const {
      return _mm512_and_si512(data, _mm512_set1_epi64(capacity - 1));
    }
#endif  // NEIGHBOR_HASH_SIMD_FIND
  };
};

template <class K, class V, size_t NumGroupBits = 0>
struct DefaultMultiGroupPolicy : public DefaultPolicy<K, V> {
  static std::tuple<int64_t, int64_t> SubRange(
      int64_t head_slot_index, size_t capacity) {
    size_t per_group_size = capacity >> NumGroupBits;
    size_t bits = __builtin_ctz(per_group_size) + 1;
    int64_t start = (head_slot_index >> bits) << bits;
    return {start, start + per_group_size};
  }
};

}  // namespace policy
}  // namespace neighbor
