#include "absl/container/flat_hash_map.h"

namespace neighbor {
namespace test {

template <class K, class V>
struct TestPolicyTraits {
  static constexpr size_t kPayloadBitCount = sizeof(V) * 8 - 12;

  static size_t NormalizeCapacity(size_t n) {
    size_t power = 1;
    while (power < n) {
      power <<= 1;
    }
    return power;
  }

  static bool ShouldGrow(size_t size, size_t capactiy) {
    return size >= capactiy * 0.9;
  }

  static size_t GrowthSize(size_t n) {
    return std::max(n * 2, 1024UL);
  }

  template <size_t alignment = 64>
  static void* Allocate(size_t size) {
    return std::aligned_alloc(alignment, size);
  }

  static void Deallocate(void* ptr, size_t) {
    free(ptr);
  }

  using Hash = absl::Hash<K>;

  struct Hash2Slot {
    uint64_t operator()(size_t hash, size_t capacity) const {
      return hash & (capacity - 1);
    }
  };
};


template <class K, class V, size_t NumGroupBits = 0>
struct TestMultiGroupPolicyTraits : public TestPolicyTraits<K, V> {
  static std::tuple<int64_t, int64_t> SubRange(int64_t head_slot_index, size_t capacity) {
    size_t per_group_size = capacity >> NumGroupBits;
    size_t bits = __builtin_ctz(per_group_size) + 1;
    int64_t start = (head_slot_index >> bits) << bits;
    return {start, start + per_group_size};
  }
};

}
}
