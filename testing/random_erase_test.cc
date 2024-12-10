#include <stdint.h>  // for uint32_t, uint64_t, int64_t
#include <stdlib.h>  // for size_t, aligned_alloc, free

#include <algorithm>      // for max
#include <random>         // for uniform_int_distribution
#include <unordered_map>  // for unordered_map, operator!=

#include "gtest/gtest.h"  // for Message, TestPartResult

#include "neighbor_hash/neighbor_hash.h"  // for NeighborHashMap, operator!=
#include "neighbor_hash/slot_type.h"      // for neighbor

namespace neighbor {
namespace {

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

  static size_t GrowthSize(size_t n) { return std::max(n * 2, 1024UL); }

  template <size_t alignment = 64>
  static void* Allocate(size_t size) {
    return std::aligned_alloc(alignment, size);
  }

  static void Deallocate(void* ptr, size_t) { free(ptr); }

  using Hash = std::hash<K>;

  struct Hash2Slot {
    uint64_t operator()(size_t hash, size_t capacity) const {
      return hash & (capacity - 1);
    }
  };
};

using TestK = uint32_t;
using TestV = uint32_t;
using TestPT = TestPolicyTraits<TestK, TestV>;
using TestNHM = NeighborHashMap<TestK, TestV, TestPT>;

class RandomEraseTest : public ::testing::Test {
 protected:
  TestNHM map;
};

static void RandomEraseOne(const uint32_t k, std::random_device& rd, TestNHM& map) {
  static constexpr uint32_t kRange = 524288;  // NOTE: ~50%.

  map.clear();

  std::unordered_map<uint32_t, uint32_t> ref_map;

  std::default_random_engine gen(rd());
  std::uniform_int_distribution<uint32_t> dist(0, kRange - 1);

  for (uint32_t j = 16; j > 0; --j) {
    const auto bound = kRange / j;

    for (uint32_t i = 0; i < bound; ++i) {
      uint32_t key = dist(gen);
      uint32_t value = i + 1;
      map[key] = value;
      ref_map[key] = value;
    }
    for (uint32_t i = 0; i < bound; ++i) {
      uint32_t key = dist(gen);
      map.erase(key);
      ref_map.erase(key);
    }

    EXPECT_EQ(map.size(), ref_map.size());

    for (const auto& x : ref_map) {
      const auto it = map.find(x.first);
      EXPECT_NE(it, map.end());
      EXPECT_EQ(it->second, x.second);
    }
  }
}

TEST_F(RandomEraseTest, RepeatedRandomInsertAndErase) {
  std::random_device rd;

  for (uint32_t k = 0; k < 32; ++k) {
    RandomEraseOne(k, rd, map);
  }
}

}  // namespace
}  // namespace neighbor
