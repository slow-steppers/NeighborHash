#include <stdint.h>  // for uint32_t, uint64_t, int64_t
#include <stdlib.h>  // for size_t, aligned_alloc, free

#include <algorithm>      // for max
#include <functional>     // for hash
#include <random>         // for uniform_int_distribution
#include <unordered_map>  // for unordered_map, operator!=

#include "absl/hash/hash.h"  // for Hash
#include "gtest/gtest.h"     // for Message, TestPartResult

#include "neighbor_hash/common_policy.h"  // for DefaultIntegerHash
#include "neighbor_hash/neighbor_hash.h"  // for NeighborHashMap, operator!=
#include "neighbor_hash/slot_type.h"      // for neighbor

namespace neighbor {
namespace {

template <typename K, typename V, typename H>
struct TestPolicy {
  static constexpr size_t kPayloadBitCount = sizeof(V) * 8 - 12;

  static constexpr size_t NormalizeCapacity(size_t n) {
    size_t power = 1;
    while (power < n) {
      power <<= 1;
    }
    return power;
  }

  static constexpr bool ShouldGrow(size_t size, size_t capactiy) {
    return size >= capactiy * 0.81;
  }

  static constexpr size_t GrowthSize(size_t n) {
    return std::max(n * 2, 1024UL);
  }

  template <size_t alignment = 64>
  static void* Allocate(size_t size) {
    auto* p = std::aligned_alloc(
        alignment, (size + alignment - 1) / alignment * alignment);
    return p;
  }

  static void Deallocate(void* ptr, size_t) { free(ptr); }

  using Hash = H;

  struct Hash2Slot {
    uint64_t operator()(size_t hash, size_t capacity) const {
      return hash & (capacity - 1);
    }
  };
};

template <typename K, typename V, typename H>
class RandomEraseTest : public ::testing::Test {
 protected:
  using P = TestPolicy<K, V, H>;
  using M = NeighborHashMap<K, V, P>;

  void RandomEraseSuiteOnce(const uint32_t p) {
    static constexpr K kBound =
        (static_cast<K>(1) << (P::kPayloadBitCount - 1));  // NOTE: ~50%.

    static constexpr uint32_t kCountBase = 16384;
    static constexpr uint32_t kCountTimes = 24;

    map_.clear();

    std::unordered_map<K, V> ref;

    std::default_random_engine gen(rd_());
    std::uniform_int_distribution<K> dist(0, kBound - 1);

    for (uint32_t t = 1; t <= kCountTimes; ++t) {
      const uint32_t bound = kCountBase * t;

      for (uint32_t i = 1; i <= bound; ++i) {
        const K key = dist(gen);
        map_[key] = i;
        ref[key] = i;
      }
      for (uint32_t i = 1; i <= bound; ++i) {
        const K key = dist(gen);
        map_.erase(key);
        ref.erase(key);
      }

      EXPECT_EQ(map_.size(), ref.size());

      for (const auto& x : ref) {
        const auto it = map_.find(x.first);
        EXPECT_NE(it, map_.end());
        EXPECT_EQ(it->second, x.second);
      }
    }
  }

  std::random_device rd_;

  M map_;
};

using RandomErase32STest =
    RandomEraseTest<uint32_t, uint32_t, std::hash<uint32_t>>;

TEST_F(RandomErase32STest, Repeat21) {
  for (uint32_t p = 0; p < 21; ++p) {
    RandomEraseSuiteOnce(p);
  }
}

using RandomErase32ATest =
    RandomEraseTest<uint32_t, uint32_t, absl::Hash<uint32_t>>;

TEST_F(RandomErase32ATest, Repeat21) {
  for (uint32_t p = 0; p < 21; ++p) {
    RandomEraseSuiteOnce(p);
  }
}

using RandomErase32DTest =
    RandomEraseTest<uint32_t, uint32_t, policy::DefaultIntegerHash>;

TEST_F(RandomErase32DTest, Repeat21) {
  for (uint32_t p = 0; p < 21; ++p) {
    RandomEraseSuiteOnce(p);
  }
}

using RandomErase64STest =
    RandomEraseTest<uint64_t, uint64_t, std::hash<uint64_t>>;

TEST_F(RandomErase64STest, Repeat21) {
  for (uint32_t p = 0; p < 21; ++p) {
    RandomEraseSuiteOnce(p);
  }
}

using RandomErase64ATest =
    RandomEraseTest<uint64_t, uint64_t, absl::Hash<uint64_t>>;

TEST_F(RandomErase64ATest, Repeat21) {
  for (uint32_t p = 0; p < 21; ++p) {
    RandomEraseSuiteOnce(p);
  }
}

using RandomErase64DTest =
    RandomEraseTest<uint64_t, uint64_t, policy::DefaultIntegerHash>;

TEST_F(RandomErase64DTest, Repeat21) {
  for (uint32_t p = 0; p < 21; ++p) {
    RandomEraseSuiteOnce(p);
  }
}

}  // namespace
}  // namespace neighbor
