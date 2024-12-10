#include <stdint.h>  // for uint32_t, uint64_t, int64_t
#include <stdlib.h>  // for size_t, aligned_alloc, free

#include <algorithm>      // for max
#include <map>            // for map, operator!=, _Rb_tree_i...
#include <random>         // for uniform_int_distribution
#include <set>            // for set
#include <tuple>          // for tuple
#include <unordered_map>  // for unordered_map, operator!=
#include <utility>        // for pair
#include <vector>         // for allocator, vector

#include "gtest/gtest.h"  // for Message, TestPartResult

#include "neighbor_hash/neighbor_hash.h"  // for NeighborHashMap, operator!=
#include "neighbor_hash/slot_type.h"      // for neighbor

using namespace neighbor;

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

template <class K, class V, size_t NumGroupBits = 0>
struct TestMultiGroupPolicyTraits : public TestPolicyTraits<K, V> {
  static std::tuple<int64_t, int64_t> SubRange(
      int64_t head_slot_index, size_t capacity) {
    size_t per_group_size = capacity >> NumGroupBits;
    size_t bits = __builtin_ctz(per_group_size) + 1;
    int64_t start = (head_slot_index >> bits) << bits;
    return {start, start + per_group_size};
  }
};

class PayloadProxyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Set up any common test conditions here
  }

  void TearDown() override {
    // Clean up after each test
  }
};

using MyPolicyTraits = TestPolicyTraits<uint32_t, uint32_t>;

TEST_F(PayloadProxyTest, DefaultValue) {
  NeighborHashMap<uint32_t, uint32_t, MyPolicyTraits> hashmap;
  NeighborHashMap<uint32_t, uint32_t, MyPolicyTraits>::payload_proxy_type
      payload;

  // Test that the default value of a payload is 0
  ASSERT_EQ(static_cast<uint32_t>(payload), 0);
}

TEST_F(PayloadProxyTest, Assignment) {
  NeighborHashMap<uint32_t, uint32_t, MyPolicyTraits>::payload_proxy_type
      payload;

  payload = 42;

  // Test assignment and retrieval of the payload value
  ASSERT_EQ(static_cast<uint32_t>(payload), 42);
}

TEST_F(PayloadProxyTest, Masking) {
  NeighborHashMap<uint32_t, uint32_t, MyPolicyTraits>::payload_proxy_type
      payload;
  payload = 0x12345678;

  // Test that the payload only stores the lower bits
  ASSERT_EQ(static_cast<uint32_t>(payload), (0x12345678U << 12) >> 12);
}

TEST_F(PayloadProxyTest, ImplicitConversion) {
  NeighborHashMap<uint32_t, uint32_t, MyPolicyTraits>::payload_proxy_type
      payload;
  payload = 0x12345678;

  // Test that the payload only stores the lower bits
  ASSERT_TRUE(payload == (0x12345678U << 12) >> 12);
}

// Test fixture for NeighborHashMap tests
class NeighborHashMapTest : public ::testing::Test {
 protected:
  NeighborHashMap<uint32_t, uint32_t, TestPolicyTraits<uint32_t, uint32_t>> map;
};

TEST_F(NeighborHashMapTest, EmptyHashMap) {
  EXPECT_TRUE(map.empty());
  EXPECT_EQ(map.size(), 0U);
}

TEST_F(NeighborHashMapTest, InsertAndFind) {
  auto result = map.emplace(42, 100);
  EXPECT_TRUE(result.second);  // Insertion successful
  EXPECT_EQ(result.first->second, 100);

  auto found = map.find(42);
  EXPECT_NE(found, map.end());
  EXPECT_EQ(found->second, 100);
}

TEST_F(NeighborHashMapTest, InsertDuplicate) {
  map.insert(42, 100);
  auto result = map.insert(42, 200);  // Attempt to insert a duplicate key
  EXPECT_FALSE(result.second);        // Insertion should fail

  auto found = map.find(42);
  EXPECT_NE(found, map.end());
  EXPECT_EQ(found->second, 100);  // Value should remain unchanged
}

TEST_F(NeighborHashMapTest, Erase) {
  map.insert(42, 100);
  map.insert(43, 200);

  map.erase(42);
  EXPECT_EQ(map.size(), 1U);

  auto found = map.find(42);
  EXPECT_EQ(found, map.end());

  auto special_key = decltype(map)::kUnOccupiedKey;
  map.insert(special_key, 128);
  auto found2 = map.find(special_key);
  ASSERT_NE(found2, map.end());
  EXPECT_EQ(found2->second, 128);

  map.erase(found2);
  ASSERT_EQ(map.find(special_key), map.end());
  EXPECT_EQ(map.size(), 1U);

  auto found3 = map.find(43);
  ASSERT_NE(found3, map.end());
  EXPECT_EQ(found3->second, 200);
}

TEST_F(NeighborHashMapTest, OperatorSquareBrackets) {
  map[42] = 100;
  EXPECT_EQ(map.size(), 1U);
  EXPECT_EQ(map[42], 100);

  map[42] = 200;
  EXPECT_EQ(map.size(), 1U);
  EXPECT_EQ(map[42], 200);
}

TEST_F(NeighborHashMapTest, Rehash) {
  for (int i = 0; i < 1000; ++i) {
    map.insert(i, i * 10 + 1);
  }

  EXPECT_EQ(map.size(), 1000U);

  map.rehash(2000);  // Increase capacity

  for (int i = 0; i < 1000; ++i) {
    auto found = map.find(i);
    EXPECT_NE(found, map.end());
    EXPECT_EQ(found->second, i * 10 + 1);
  }
}

TEST_F(NeighborHashMapTest, Iteration) {
  ASSERT_EQ(map.begin(), map.end());

  for (size_t i = 1; i < 1000; ++i) {
    map.insert(i, i);
  }

  // Test iterating through the elements and counting them
  std::set<uint32_t> keys;
  for (auto it = map.begin(); it != map.end(); ++it) {
    ASSERT_EQ(it->first, it->second);
    keys.insert(it->first);
  }
  ASSERT_EQ(keys.size(), 999);

  map.clear();
  ASSERT_EQ(map.begin(), map.end());
}

TEST_F(NeighborHashMapTest, Clear) {
  map.insert(42, 100);
  map.insert(123, 200);
  ASSERT_EQ(map.size(), 2);

  map.clear();  // Clear the hashmap
  ASSERT_EQ(map.size(), 0);

  auto found1 = map.find(42);   // Try to find the first key
  auto found2 = map.find(123);  // Try to find the second key

  ASSERT_EQ(found1, map.end());  // First key should not be found
  ASSERT_EQ(found2, map.end());  // Second key should not be found
}

TEST_F(NeighborHashMapTest, RandomInsert) {
  std::unordered_map<uint32_t, uint32_t> reference_map;

  std::default_random_engine generator(0xdeadbeef);
  std::uniform_int_distribution<uint32_t> distribution;

  // Insert a large number of random key-value pairs
  const int num_insertions = 1024 * 1024 - 1;
  for (int i = 0; i < num_insertions; ++i) {
    uint32_t key = distribution(generator);
    uint32_t value = i + 1;
    map[key] = value;
    reference_map[key] = value;
  }

  // Verify that the inserted values can be found correctly
  for (const auto& pair : reference_map) {
    auto found = map.find(pair.first);
    // std::cout << "find:" << pair.first << std::endl;
    ASSERT_NE(found, map.end());
    ASSERT_EQ(found->second, pair.second);
  }
}

TEST_F(NeighborHashMapTest, MoveChainHead) {
  std::default_random_engine generator(0xdeadbeef);
  std::uniform_int_distribution<uint32_t> distribution;

  map[1] = 2;
  map[2] = 3;
  uint32_t collisions = map.bucket_count() + 1;
  map[collisions] = 4;
  ASSERT_EQ(map[1], 2);
  ASSERT_EQ(map[collisions], 4);

  auto it = map.find(collisions);
  ASSERT_NE(it, map.end());
  map.move_to_probing_chain_head(it);

  ASSERT_EQ(map[1], 2);
  ASSERT_EQ(map[collisions], 4);
}

#ifdef SIMD_AMAC
TEST_F(NeighborHashMapTest, SIMDHash) {
  std::default_random_engine generator(0xdeadbeef);
  std::uniform_int_distribution<uint64_t> distribution;

  NeighborHashMap<uint64_t, uint64_t,
      neighbor::policy::DefaultPolicy<uint64_t, uint64_t>>
      hashmap;

  for (int i = 0; i < 1024; ++i) {
    uint64_t keys[8];
    for (int x = 0; x < 8; ++x) {
      keys[x] = distribution(generator);
    }

    __m512i vkeys = _mm512_loadu_epi64(keys);

    uint64_t tmps[8];
    neighbor::policy::DefaultIntegerHash hash;
    auto vhash = hash.v_hash_64(vkeys);
    _mm512_storeu_epi64(tmps, vhash);
    for (int x = 0; x < 8; ++x) {
      ASSERT_EQ(tmps[x], hash(keys[x]));
    }

    auto result = hashmap.V_SlotIndex(vkeys);
    uint64_t slot_index[8];
    _mm512_storeu_epi64(slot_index, result);
    for (int x = 0; x < 8; ++x) {
      EXPECT_EQ(slot_index[x], hashmap.SlotIndex(keys[x]));
    }
  }
}
#endif

#ifdef NEIGHBOR_HASH_COROUTINE_FIND
TEST_F(NeighborHashMapTest, CoroFind) {
  auto result = map.emplace(42, 100);
  EXPECT_TRUE(result.second);  // Insertion successful
  EXPECT_EQ(result.first->second, 100);

  auto task = map.coro_find(42);
  EXPECT_FALSE(task.handle.done());
  task.handle.resume();
  EXPECT_TRUE(task.handle.done());
  EXPECT_EQ(task.result(), 100);
}
#endif  // NEIGHBOR_HASH_COROUTINE_FIND

TEST_F(NeighborHashMapTest, AMACFind) {
  std::vector<uint32_t> keys;
  for (int i = 0; i < 1024 * 1024; ++i) {
    map[i] = i;
    keys.push_back(i);
  }

  size_t count = 0;
  map.amac_find<4>(&keys[0], keys.size(), [&count](uint32_t k, uint32_t v) {
    EXPECT_EQ(v, k);
    count += 1;
  });
  EXPECT_EQ(count, keys.size());
}

using atomic_hashmap = AtomicNeighborHashMap<uint64_t, uint64_t,
    TestMultiGroupPolicyTraits<uint64_t, uint64_t>>;

class AtomicNeighborHashMapTest : public ::testing::Test {
 protected:
  atomic_hashmap map;
};

TEST_F(AtomicNeighborHashMapTest, InsertAndFind) {
  // atomic insert
  auto status = map.atomic_insert_or_update<true>(42, 100);
  EXPECT_TRUE(
      status == atomic_hashmap::Status::kInserted);  // Insertion successful

  // non-atomic insert
  auto result = map.emplace(43, 200);
  EXPECT_TRUE(result.second);  // Insertion successful
  EXPECT_EQ(result.first->second, 200);

  auto found = map.find(42);
  EXPECT_NE(found, map.end());
  EXPECT_EQ(found->second, 100);

  auto found2 = map.find(43);
  EXPECT_NE(found2, map.end());
  EXPECT_EQ(found2->second, 200);

  // emplace existed
  auto result2 = map.emplace(42, 300);
  EXPECT_FALSE(result2.second);  // Insertion successful
  EXPECT_EQ(result2.first->second, 100);
}

TEST_F(AtomicNeighborHashMapTest, AtomicInsert) {
  map.rehash(1024);
  EXPECT_EQ(map.bucket_count(), 1024);

  auto status = map.atomic_insert_or_update<false>(42, 100);
  EXPECT_TRUE(
      status == atomic_hashmap::Status::kInserted);  // Insertion successful

  status = map.atomic_insert_or_update<false>(42 + 1024, 200);
  EXPECT_TRUE(status == atomic_hashmap::Status::kTailInsertPrepared);

  EXPECT_EQ(map.size(), 1);
  EXPECT_EQ(map.find(42 + 1024), map.end());

  status = map.atomic_insert_or_update<true>(42 + 1024, 200);
  EXPECT_TRUE(status == atomic_hashmap::Status::kInserted);
  EXPECT_EQ(map.size(), 2);
  EXPECT_NE(map.find(42 + 1024), map.end());

  status = map.atomic_insert_or_update<false>(43, 300);
  EXPECT_TRUE(status == atomic_hashmap::Status::kHeadInsertPrepared);
  EXPECT_EQ(map.size(), 2);

  status = map.atomic_insert_or_update<true>(43, 300);
  EXPECT_TRUE(status == atomic_hashmap::Status::kInserted);
  EXPECT_EQ(map.size(), 3);

  auto found = map.find(42);
  ASSERT_NE(found, map.end());
  EXPECT_EQ(found->second, 100);

  found = map.find(42 + 1024);
  ASSERT_NE(found, map.end());
  EXPECT_EQ(found->second, 200);

  found = map.find(43);
  ASSERT_NE(found, map.end());
  EXPECT_EQ(found->second, 300);
}

TEST_F(AtomicNeighborHashMapTest, RandomInsert) {
  std::map<uint64_t, uint64_t> reference_map;

  std::default_random_engine generator(0xdeadbeef);
  std::uniform_int_distribution<uint64_t> distribution;

  // Insert a large number of random key-value pairs
  map[0] = 0;
  reference_map[0] = 0;
  const int num_insertions = 1024 * 1024 - 1;
  for (int i = 0; i < num_insertions; ++i) {
    uint64_t key = distribution(generator);
    uint64_t value = 2 * i + 1;
    map[key] = value;
    reference_map[key] = value;
  }

  // Verify that the inserted values can be found correctly
  for (const auto& pair : reference_map) {
    auto found = map.find(pair.first);
    // std::cout << "find:" << pair.first << std::endl;
    ASSERT_NE(found, map.end());
    ASSERT_EQ(found->second, pair.second);
  }

  // atomic insert a large number of random key-value pairs
  std::vector<std::pair<uint64_t, uint64_t>> items;
  items.emplace_back(0, 1);
  reference_map[0] = 1;
  for (int i = 0; i < num_insertions; ++i) {
    uint64_t key = distribution(generator);
    uint64_t value = 2 * i;
    items.emplace_back(key, value);
    reference_map[key] = value;
  }

  std::vector<bool> inserted;
  inserted.resize(items.size(), false);

  // avoid failure
  map.rehash(map.size() * 4);

  for (size_t i = 0; i < inserted.size(); ++i) {
    if (inserted[i]) {
      continue;
    }
    auto& item = items[i];
    auto st = map.atomic_insert_or_update<false>(item.first, item.second);
    if (st == atomic_hashmap::Status::kInserted ||
        st == atomic_hashmap::Status::kUpdated) {
      inserted[i] = true;
    } else {
      ASSERT_NE(st, atomic_hashmap::Status::kPrepareFailed);
    }
  }

  // wait an epoch
  for (size_t i = 0; i < inserted.size(); ++i) {
    if (inserted[i]) {
      continue;
    }
    auto& item = items[i];
    auto st = map.atomic_insert_or_update<true>(item.first, item.second);
    ASSERT_EQ(st, atomic_hashmap::Status::kInserted);
  }

  EXPECT_EQ(map.size(), reference_map.size());
  for (const auto& pair : reference_map) {
    auto found = map.find(pair.first);
    EXPECT_NE(found, map.end());
    EXPECT_EQ(found->second, pair.second);
  }

  std::map<uint64_t, uint64_t> dump;
  map.atomic_foreach([&dump](uint64_t k, uint64_t v) { dump[k] = v; });
  EXPECT_EQ(dump, reference_map);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
