#include <stddef.h>    // for size_t, NULL, ptrd...
#include <stdint.h>    // for uint64_t, int64_t
#include <stdlib.h>    // for abort
#include <sys/time.h>  // for gettimeofday, timeval

#include <algorithm>         // for max, sort, shuffle
#include <fstream>           // for ifstream
#include <functional>        // for equal_to
#include <initializer_list>  // for initializer_list
#include <iostream>          // for operator<<, basic_...
#include <iterator>          // for forward_iterator_tag
#include <limits>            // for numeric_limits
#include <random>            // for seed_seq, mt19937
#include <string>            // for string
#include <unordered_map>     // for unordered_map, ope...
#include <utility>           // for pair
#include <vector>            // for vector, allocator

#include "absl/base/prefetch.h"                    // for ABSL_HAVE_PREFETCH
#include "absl/container/flat_hash_map.h"          // for flat_hash_map, ope...
#include "absl/hash/hash.h"                        // for Hash
#include "absl/random/uniform_int_distribution.h"  // for uniform_int_distri...
#include "absl/random/zipf_distribution.h"         // for zipf_distribution
#include "absl/strings/str_format.h"               // for StrFormat
#include "ankerl/unordered_dense.h"                // for map, standard, hash
#include "benchmark/benchmark.h"                   // for Benchmark, BENCHMA...
#include "bytell_hash_map.hpp"                     // for bytell_hash_map
#include "hash_table7.hpp"                         // for HashMap
#include "tsl/hopscotch_hash.h"                    // for operator!=
#include "tsl/hopscotch_map.h"                     // for hopscotch_map

#include "neighbor_hash/common_policy.h"   // for DefaultPolicy
#include "neighbor_hash/linear_probing.h"  // for LinearProbingHashMap
#include "neighbor_hash/neighbor_hash.h"   // for NeighborHashMap

#ifdef NEIGHBOR_HASH_SIMD_FIND
#include "neighbor_hash/bucketing_simd.h"
#endif

#ifdef BENCHMARK_CUCKOO_HASHMAP
#include "libcuckoo/cuckoohash_map.hh"
#endif  // BENCHMARK_CUCKOO_HASHMAP

#ifdef BENCHMARK_BOOST_FLAT_MAP
#include "boost/unordered/unordered_flat_map.hpp"
#endif

using namespace neighbor;

using MyPolicyTraits = neighbor::policy::DefaultPolicy<uint64_t, uint64_t>;

inline double gettime() {
  struct timeval now_tv;
  gettimeofday(&now_tv, NULL);
  return ((double)now_tv.tv_sec) + ((double)now_tv.tv_usec) / 1000000.0;
}

using RandomEngine = std::mt19937;

template <class T>
std::vector<T> GenerateRandomNumbers(int count, T range) {
  std::vector<T> numbers;
  numbers.reserve(count);
  std::seed_seq sseq{1, 2, 3};
  RandomEngine gen(sseq);
  absl::uniform_int_distribution<T> distribution(0, range);
  for (int i = 0; i < count; ++i) {
    numbers.push_back(distribution(gen));
  }
  return numbers;
}

template <typename MapType>
static void BM_RandomInsert(benchmark::State& state) {
  auto random_numbers = GenerateRandomNumbers<uint64_t>(
      state.range(0), std::numeric_limits<uint64_t>::max());
  size_t value = 1;
  for (auto _ : state) {
    state.PauseTiming();
    MapType map;
    map.reserve(random_numbers.size() * 2);
    state.ResumeTiming();
    for (const auto& num : random_numbers) {
      map[num] = value++;
    }
    benchmark::DoNotOptimize(map);
  }
}

const std::vector<std::pair<int64_t, int64_t>> benchmark_ranges = {
    {1, 1 << 16}};

BENCHMARK_TEMPLATE(BM_RandomInsert, std::unordered_map<uint64_t, uint64_t>)
    ->Ranges(benchmark_ranges);
BENCHMARK_TEMPLATE(BM_RandomInsert, absl::flat_hash_map<uint64_t, uint64_t>)
    ->Ranges(benchmark_ranges);
BENCHMARK_TEMPLATE(
    BM_RandomInsert, NeighborHashMap<uint64_t, uint64_t, MyPolicyTraits>)
    ->Ranges(benchmark_ranges);

template <class K, class V, class PolicyTraits = MyPolicyTraits>
class Array {
 public:
  using SlotType = std::pair<K, V>;

  ~Array() {
    if (data_) {
      PolicyTraits::Deallocate(data_, sizeof(SlotType) * capacity_);
    }
  }

  void reserve(size_t size) {
    size_t capacity = PolicyTraits::NormalizeCapacity(size);
    if (data_) {
      PolicyTraits::Deallocate(data_, sizeof(SlotType) * capacity_);
    }
    data_ = (SlotType*)PolicyTraits::template Allocate<64>(
        sizeof(SlotType) * capacity);
    capacity_ = capacity;
  }

  V& operator[](K key) {
    auto& pair = data_[hash_(key) & (capacity_ - 1)];
    pair.first = key;
    return pair.second;
  }

  struct iterator {
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::pair<K, V>;
    using reference = value_type&;
    using pointer = value_type*;
    using difference_type = ptrdiff_t;

    iterator() {}

    iterator(value_type* value) : value_(value) {}

    reference operator*() const { return *value_; }

    pointer operator->() const { return &operator*(); }

    friend bool operator==(const iterator& a, const iterator& b) {
      return a.value_ == b.value_;
    }

    friend bool operator!=(const iterator& a, const iterator& b) {
      return !(a == b);
    }

    std::pair<K, V>* value_{nullptr};
  };

  iterator end() const { return iterator{}; }

  iterator find(K key) const {
    return iterator{&data_[hash_(key) & (capacity_ - 1)]};
  }

#ifdef ABSL_HAVE_PREFETCH
#ifdef NEIGHBOR_HASH_SIMD_FIND
  struct AMAC_State {
    __m512i vslot_index;
  };

  template <int kPipelineSize, class Fn>
  void amac_find(const K* keys, int64_t keys_size, Fn&& fn) {
    AMAC_State states[kPipelineSize];
    int64_t keys_index = 0;
    constexpr int kSIMDWidth = 8;

    // align
    while (intptr_t(&keys[keys_index]) % 64 != 0 && keys_index < keys_size) {
      auto key = keys[keys_index];
      auto it = find(key);
      if (it != end()) {
        fn(key, it->second);
      }
      keys_index += 1;
    }

    if (keys_index + kSIMDWidth > keys_size) {
      while (keys_index < keys_size) {
        auto key = keys[keys_index++];
        auto it = find(key);
        if (it != end()) {
          fn(key, it->second);
        }
      }
    }

    alignas(64) uint64_t tmp[kSIMDWidth];
    // initial-fill
    for (int i = 0; i < kPipelineSize; ++i) {
      auto& state = states[i];
      __m512i vkey = _mm512_load_epi64(&keys[keys_index]);
      state.vslot_index =
          hash2slot_.v_hash2slot(hash_.v_hash_64(vkey), capacity_);
      keys_index += kSIMDWidth;
      _mm512_store_epi64(tmp, state.vslot_index);
      for (int x = 0; x < kSIMDWidth; ++x) {
        absl::PrefetchToLocalCache(data_ + tmp[x]);
      }
    }
    absl::PrefetchToLocalCache(&keys[keys_index]);

    int state_circular_buffer_index = 0;
    int state_circular_buffer_stop_index = -1;
    while (true) {
      auto& state = states[state_circular_buffer_index & (kPipelineSize - 1)];
      __m512i vslot_index_by_byte = _mm512_slli_epi64(state.vslot_index, 4);
      __m512i vslot_value = _mm512_i64gather_epi64(
          vslot_index_by_byte, reinterpret_cast<uint64_t*>(data_) + 1, 1);
      _mm512_store_epi64(tmp, vslot_value);
      for (int x = 0; x < kSIMDWidth; ++x) {
        fn(0, tmp[x]);
      }
      // fetch next keys
      if (keys_index + kSIMDWidth > keys_size) {
        state_circular_buffer_stop_index =
            state_circular_buffer_index & (kPipelineSize - 1);
        break;
      }
      __m512i vkey = _mm512_load_epi64(&keys[keys_index]);
      state.vslot_index =
          hash2slot_.v_hash2slot(hash_.v_hash_64(vkey), capacity_);
      keys_index += kSIMDWidth;

      // prefetch values
      _mm512_store_epi64(tmp, state.vslot_index);
      for (int x = 0; x < kSIMDWidth; ++x) {
        absl::PrefetchToLocalCache(data_ + tmp[x]);
      }
      absl::PrefetchToLocalCache(&keys[keys_index]);

      state_circular_buffer_index += 1;
    }

    // finish all tasks
    for (int i = 0; i < kPipelineSize; ++i) {
      if (i == state_circular_buffer_stop_index) {
        continue;
      }
      auto& state = states[i];
      _mm512_store_epi64(tmp, state.vslot_index);
      for (int x = 0; x < kSIMDWidth; ++x) {
        fn(0, data_[tmp[x]].second);
      }
    }

    while (keys_index < keys_size) {
      auto key = keys[keys_index++];
      auto it = find(key);
      if (it != end()) {
        fn(key, it->second);
      }
    }
  }
#endif
#endif

  float load_factor() const { return 1.0; }

 private:
  typename PolicyTraits::Hash hash_;
  typename PolicyTraits::Hash2Slot hash2slot_;

  SlotType* data_ = nullptr;
  size_t capacity_ = 0;
};

std::vector<uint64_t> g_numbers;
std::vector<uint64_t> g_access_numbers;

template <class MapType, class Iter, class ValueAccumulator>
void batch_find_or_default(const MapType& map, Iter key_begin, Iter key_end,
    const ValueAccumulator& value_acc) {
  for (auto key_it = key_begin; key_it != key_end; ++key_it) {
    auto it = map.find(*key_it);
    if (it != map.end()) {
      value_acc(it->second);
    } else {
      value_acc(0);
    }
  }
}

struct once {
  size_t operator()(size_t) { return size_++; }

  size_t size_{0};
};

struct uniform {
  size_t operator()(size_t size) {
    return absl::uniform_int_distribution<size_t>(0, size)(gen_);
  }

  std::seed_seq sseq_{1, 2, 3};
  RandomEngine gen_{sseq_};
};

struct zipf {
  size_t operator()(size_t size) {
    return absl::zipf_distribution<size_t>(size, 2.0, size / 10)(gen_);
  }

  std::seed_seq sseq_{1, 2, 3};
  RandomEngine gen_{sseq_};
};

template <typename MapType>
void insert(MapType& map, uint64_t key, uint64_t value) {
  map[key] = value;
}

#ifdef BENCHMARK_CUCKOO_HASHMAP
void insert(
    libcuckoo::cuckoohash_map<uint64_t, uint64_t, absl::Hash<uint64_t>>& map,
    uint64_t key, uint64_t value) {
  map.insert(key, value);
}

template <class Iter, class ValueAccumulator>
void batch_find_or_default(
    const libcuckoo::cuckoohash_map<uint64_t, uint64_t, absl::Hash<uint64_t>>&
        map,
    Iter key_begin, Iter key_end, const ValueAccumulator& value_acc) {
  for (auto key_it = key_begin; key_it != key_end; ++key_it) {
    if (!map.find_fn(*key_it, value_acc)) {
      value_acc(0);
    }
  }
}

#endif

template <class Map, int kShard>
class MultiShard {
 public:
  typename Map::mapped_type& operator[](const typename Map::key_type& key) {
    int shard = key & (kShard - 1);
    return maps_[shard][key];
  }

  float load_factor() const { return maps_[0].load_factor(); }

  void reserve(size_t size) {
    for (int i = 0; i < kShard; ++i) {
      maps_[i].reserve(size / kShard);
    }
  }

  Map maps_[kShard];
};

template <class Map, int kShard, class Iter, class ValueAccumulator>
void batch_find_or_default(const MultiShard<Map, kShard>& map, Iter key_begin,
    Iter key_end, const ValueAccumulator& value_acc) {
  for (int i = 0; i < kShard; ++i) {
    for (auto key_it = key_begin; key_it != key_end; ++key_it) {
      auto key = *key_it;
      int shard = key & (kShard - 1);
      if (shard == i) {
        auto it = map.maps_[shard].find(key);
        if (it != map.maps_[shard].end()) {
          value_acc(it->second);
        } else {
          value_acc(0);
        }
      }
    }
  }
}

template <int kPipelineSize, class MapType, class Iter, class ValueAccumulator>
void batch_find_coro(const MapType& map, Iter key_begin, Iter key_end,
    const ValueAccumulator& value_acc) {
  std::vector<typename MapType::coro_find_task> tasks;
  auto key_it = key_begin;
  for (int i = 0; i < kPipelineSize; ++i) {
    tasks.emplace_back(map.coro_find(*key_it++));
  }

  int state_circular_buffer_index = 0;
  while (key_it != key_end) {
    auto& task = tasks[state_circular_buffer_index & (kPipelineSize - 1)];
    state_circular_buffer_index += 1;

    task.handle.resume();
    if (task.handle.done()) {
      value_acc(task.result());
      task = map.coro_find(*key_it++);
    }
  }

  for (int i = 0; i < kPipelineSize; ++i) {
    while (!tasks[i].handle.done()) {
      tasks[i].handle.resume();
    }
    value_acc(tasks[i].result());
  }
}

template <class T, class... Tags>
class TaggedMap : public T {};

template <class T>
void print_debug_info(const T&) {}

template <class K, class V, class P, class... Tags>
void print_debug_info(const TaggedMap<NeighborHashMap<K, V, P>, Tags...>& map) {
  // map.print_offset();
  map.count_hit_keys_cacheline_access();
}

template <class K, class V, class P, class... Tags>
void print_debug_info(
    const TaggedMap<neighbor::LinearProbingHashMap<K, V, P>, Tags...>& map) {
  // map.print_offset();
  map.count_hit_keys_cacheline_access();
}

#ifdef NEIGHBOR_HASH_SIMD_FIND
template <class K, class V, class P, class F, class... Tags>
void print_debug_info(
    const TaggedMap<neighbor::BucketingSIMDHashTable<K, V, P, F>, Tags...>&
        map) {
  // map.print_offset();
  map.count_hit_keys_cacheline_access();
}
#endif

template <class K, class T>
bool filter_key(K key, const T& map) {
  return true;
}

template <class K, class V, class P>
bool filter_key(K key, const neighbor::LinearProbingHashMap<K, V, P>& map) {
  return map.is_one_cacheline_access(key);
}

template <class K, class V, class P>
bool filter_key(K key, const NeighborHashMap<K, V, P>& map) {
  return map.is_one_cacheline_access(key);
}

template <int kWindowSize>
struct AMAC {
  static constexpr int value = kWindowSize;
};

struct MultiThreading {};

struct Vec {};

struct IntraVec {};

struct Scalar {};

struct QFGO {};

using NeighborHash = NeighborHashMap<uint64_t, uint64_t, MyPolicyTraits>;

template <typename MapType, typename Distrib,
    bool kQueryGuideOptimization = false, typename AMACWindow = AMAC<0>,
    bool kSIMD = false>
static void BM_RandomAccess(benchmark::State& state) {
  double hit_ratio = static_cast<double>(state.range(1)) / 100;
  double load_factor = static_cast<double>(state.range(2)) / 100;
  size_t dataset_size = state.range(0) * load_factor;

  size_t keys_count = dataset_size / hit_ratio;
  auto numbers =
      GenerateRandomNumbers(keys_count, std::numeric_limits<uint64_t>::max());

  std::seed_seq sseq{1, 2, 3};
  RandomEngine gen(sseq);

  MapType map;
  map.reserve(dataset_size);
  for (size_t i = 0; i < dataset_size; ++i) {
    auto num = numbers[i];
    int value = absl::uniform_int_distribution<int>(0, dataset_size)(gen);
    insert(map, num, value);
  }

  std::shuffle(numbers.begin(), numbers.end(), gen);
  std::vector<uint64_t> access_numbers;

  Distrib distrib;
  for (size_t i = 0; i < numbers.size(); ++i) {
    size_t sampled_index = distrib(numbers.size());
    auto key = numbers[sampled_index];
    // if (!filter_key(key, map)) continue;
    access_numbers.push_back(key);
  }
  // std::cout << "checksum:" << checksum << std::endl;
  // std::cout << "access_numbers:" << access_numbers.size() << std::endl;

  if constexpr (kQueryGuideOptimization) {
    std::unordered_map<uint64_t, int> query_freq;
    for (auto k : access_numbers) {
      query_freq[k] += 1;
    }
    std::vector<std::pair<uint64_t, int>> freqs;
    for (auto&& it : query_freq) {
      freqs.push_back(it);
    }
    std::sort(freqs.begin(), freqs.end(),
        [](const std::pair<uint64_t, int>& a,
            const std::pair<uint64_t, int>& b) { return a.second < b.second; });
    for (auto& pair : freqs) {
      auto it = map.find(pair.first);
      if (it != map.end()) {
        map.move_to_probing_chain_head(it);
      }
    }
  }

  // print_debug_info(map);
  size_t total_size = 0;
  uint64_t sum = 0;

#ifdef INFINIT_FIND_LOOP
  const int loop_count =
      std::max(8, static_cast<int>(100000000 / access_numbers.size()));
#else
  const int loop_count =
      std::max(1, static_cast<int>(100000000 / access_numbers.size()));
#endif

  if constexpr (kSIMD) {
    for (auto _ : state) {
#ifdef INFINIT_FIND_LOOP
      std::cout << "start to benchmark:" << loop_count << std::endl;
      for (int x = 0; x < 1024; ++x) {
        auto begin = gettime();
#endif
        for (int xx = 0; xx < loop_count; ++xx) {
          sum = 0;
          if constexpr (AMACWindow::value != 0) {
            map.template simd_amac_find<AMACWindow::value>(
                access_numbers, [&sum](auto, uint64_t v) { sum ^= v; });
          } else {
            map.template simd_find(
                access_numbers, [&sum](auto, uint64_t v) { sum ^= v; });
          }
          benchmark::DoNotOptimize(sum);
          total_size += access_numbers.size();
        }
#ifdef INFINIT_FIND_LOOP
        auto end = gettime();
        std::cout << "speed:"
                  << loop_count * access_numbers.size() / 1e6 / (end - begin)
                  << " M-ops/s" << std::endl;
      }
#endif
    }
  } else if constexpr (AMACWindow::value != 0) {
#ifdef INFINIT_FIND_LOOP
    std::cout << "start to benchmark:" << loop_count << std::endl;
    for (int x = 0; x < 1024; ++x) {
      auto begin = gettime();
      for (int xx = 0; xx < loop_count; ++xx) {
        map.template amac_find<AMACWindow::value>(&access_numbers[0],
            access_numbers.size(), [&sum](auto, uint64_t v) { sum ^= v; });
        benchmark::DoNotOptimize(sum);
      }
      auto end = gettime();
      std::cout << "speed:"
                << loop_count * access_numbers.size() / 1e6 / (end - begin)
                << " M-ops/s" << std::endl;
    }
#endif
    for (auto _ : state) {
      for (int xx = 0; xx < loop_count; ++xx) {
        sum = 0;
        map.template amac_find<AMACWindow::value>(&access_numbers[0],
            access_numbers.size(), [&sum](auto, uint64_t v) { sum ^= v; });
        benchmark::DoNotOptimize(sum);
        total_size += access_numbers.size();
      }
    }
  } else {
#ifdef INFINIT_FIND_LOOP
    std::cout << "start to benchmark" << std::endl;
    for (int x = 0; x < 1024; ++x) {
      auto begin = gettime();
      for (int xx = 0; xx < loop_count; ++xx) {
        batch_find_or_default(map, access_numbers.begin(), access_numbers.end(),
            [&sum](uint64_t v) { sum ^= v; });
      }
      auto end = gettime();
      std::cout << "speed:"
                << loop_count * access_numbers.size() / 1e6 / (end - begin)
                << " M-ops/s" << std::endl;
    }
#endif

    for (auto _ : state) {
      for (int xx = 0; xx < loop_count; ++xx) {
        sum = 0;
        batch_find_or_default(map, access_numbers.begin(), access_numbers.end(),
            [&sum](uint64_t v) { sum ^= v; });
        benchmark::DoNotOptimize(sum);
        total_size += access_numbers.size();
      }
    }
  }

  state.SetLabel(absl::StrFormat(
      "load_factor:%.1f%%, checksum:%lx", 100.0 * map.load_factor(), sum));
  state.SetItemsProcessed(total_size);
}

struct HashJoinTuple {
  uint64_t key;
  uint64_t value;
};

template <typename MapType, typename AMACWindow = AMAC<0>, bool kSIMD = false>
static void BM_HashJoinProbing(benchmark::State& state) {
  auto read_from_binary = [](const std::string& filename,
                              std::vector<HashJoinTuple>& output) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    output.resize(size / sizeof(HashJoinTuple));

    if (!file.read((char*)&output[0], size)) {
      abort();
    }
  };

  const std::string kRelation_R = "r_file";
  const std::string kRelation_S = "s_file";

  std::vector<HashJoinTuple> relation_s;
  std::vector<HashJoinTuple> relation_r;

  read_from_binary(kRelation_R, relation_r);
  read_from_binary(kRelation_S, relation_s);

  uint64_t max_r = 0;
  uint64_t max_s = 0;
  for (auto& x : relation_r) {
    max_r = std::max(x.key, max_r);
  }
  for (auto& x : relation_s) {
    max_s = std::max(x.key, max_s);
  }
  std::cout << "r:" << max_r << " s:" << max_s << std::endl;

  MapType map;
  map.reserve(relation_r.size());

  for (auto& tuple : relation_r) {
    insert(map, tuple.key, tuple.value);
  }
  std::cout << "relation_r:" << relation_r.size()
            << " relation_s:" << relation_s.size()
            << " hash size:" << map.size() << std::endl;

  std::vector<uint64_t> probe_keys;
  probe_keys.reserve(relation_s.size());
  std::vector<uint64_t> probe_values;
  probe_values.reserve(relation_r.size());
  for (auto& tuple : relation_s) {
    probe_keys.push_back(tuple.key);
    probe_values.push_back(tuple.value);
  }

  // print_debug_info(map);
  size_t total_size = 0;

  constexpr int kOutputBufferSize = 4096;
  alignas(64) uint64_t output[kOutputBufferSize];
  int write_pos = 0;
  uint64_t sum = 0;

  if constexpr (kSIMD) {
    for (auto _ : state) {
      sum = 0;
      // SIMD-version not write results, anyway, the fastest version is scalar AMAC
      if constexpr (AMACWindow::value != 0) {
        map.template simd_amac_find<AMACWindow::value>(
            probe_keys, [&sum](auto, uint64_t v) { sum ^= v; });
      } else {
        map.template simd_find(
            probe_keys, [&sum](auto, uint64_t v) { sum ^= v; });
      }
      benchmark::DoNotOptimize(sum);
      total_size += probe_keys.size();
    }
  } else if constexpr (AMACWindow::value != 0) {
    int iterations = 0;
    auto begin = gettime();
    for (auto _ : state) {
      sum = 0;
      map.template amac_find_with_index<AMACWindow::value>(&probe_keys[0],
          probe_keys.size(),
          [&sum, output, &write_pos, &probe_values](
              int key_index, auto, uint64_t v) mutable {
            output[write_pos & (kOutputBufferSize - 1)] = v;
            write_pos += 1;

            output[write_pos & (kOutputBufferSize - 1)] =
                probe_values[key_index];
            write_pos += 1;

            sum += 1;
          });
      benchmark::DoNotOptimize(sum);
      total_size += probe_keys.size();
      iterations += 1;
    }
    auto end = gettime();
    std::cout << "iterations:" << iterations
              << " per-iteration(ms):" << (end - begin) * 1000 / iterations
              << std::endl;
  } else {
    for (auto _ : state) {
      sum = 0;
      for (size_t key_index = 0; key_index < probe_keys.size(); ++key_index) {
        auto it = map.find(probe_keys[key_index]);
        if (it != map.end()) {
          output[write_pos & (kOutputBufferSize - 1)] = it->second;
          write_pos += 1;

          output[write_pos & (kOutputBufferSize - 1)] = probe_values[key_index];
          write_pos += 1;

          sum += 1;
        }
      }
      benchmark::DoNotOptimize(sum);
      total_size += probe_keys.size();
    }
  }

  state.SetLabel(absl::StrFormat(
      "load_factor:%.1f%%, checksum:%lx", 100.0 * map.load_factor(), sum));
  state.SetItemsProcessed(total_size);
}

template <bool kChasing>
static void BM_RandomChasing(benchmark::State& state) {
  size_t keys_count = state.range(0);
  auto keys =
      GenerateRandomNumbers(keys_count, std::numeric_limits<uint64_t>::max());
  auto values =
      GenerateRandomNumbers(keys_count, std::numeric_limits<uint64_t>::max());

  size_t total_size = 0;
  size_t capacity = state.range(0);
  for (auto _ : state) {
    uint64_t result = 0;
    uint64_t prev_value = keys[0];
    for (int64_t i = 0; i < state.range(0); ++i) {
      auto value = values[(keys[i] + prev_value) & (capacity - 1)];
      // prev value as part of next key
      if constexpr (kChasing) {
        prev_value = value;
      }
      // avoid optimize
      result ^= value;
    }
    benchmark::DoNotOptimize(result);
    total_size += state.range(0);
  }
  state.SetItemsProcessed(total_size);
}

static void RandomAccessBenchmark(benchmark::internal::Benchmark* b) {
  for (int64_t size : {
           1UL << 10,  // 1K
           1UL << 14,  // 16K
           1UL << 17,  // 128K
           1UL << 20,  // 1M
           1UL << 21,  // 2M
           1UL << 24,  // 16M
           1UL << 26,  // 64M
           1UL << 27,  // 128M
       }) {
    for (long sqr : {30, 50, 90, 100}) {
      b->Args({size, sqr, 79});
    }
  }

  // 128M, lf=0.7, sqr=100%
  b->Args({1UL << 27, 100, 70});
  b->Args({1UL << 27, 30, 70});
  b->Args({1UL << 20, 30, 70});
  b->Args({1UL << 17, 30, 70});

  // 128M, lf=0.5, sqr=100%
  b->Args({1UL << 27, 100, 50});
  b->Args({1UL << 27, 30, 50});
}

using LinearProbing =
    neighbor::LinearProbingHashMap<uint64_t, uint64_t, MyPolicyTraits>;

#ifdef NEIGHBOR_HASH_SIMD_FIND
using BucketingSIMD_16x16 = neighbor::BucketingSIMDHashTable<uint64_t, uint64_t,
    MyPolicyTraits, neighbor::Fingerprint_16x16>;
using BucketingSIMD_8x16 = neighbor::BucketingSIMDHashTable<uint64_t, uint64_t,
    MyPolicyTraits, neighbor::Fingerprint_8x16>;
using BucketingSIMD_16x8 = neighbor::BucketingSIMDHashTable<uint64_t, uint64_t,
    MyPolicyTraits, neighbor::Fingerprint_16x8>;
#endif

// zipf
BENCHMARK_TEMPLATE(BM_RandomAccess, Array<uint64_t, uint64_t>, zipf)
    ->Apply(RandomAccessBenchmark);

#ifdef NEIGHBOR_HASH_SIMD_FIND
#ifdef ABSL_HAVE_PREFETCH
BENCHMARK_TEMPLATE(
    BM_RandomAccess, Array<uint64_t, uint64_t>, zipf, false, AMAC<16>)
    ->Apply(RandomAccessBenchmark);
BENCHMARK_TEMPLATE(
    BM_RandomAccess, Array<uint64_t, uint64_t>, uniform, false, AMAC<16>)
    ->Apply(RandomAccessBenchmark);

BENCHMARK_TEMPLATE(BM_RandomAccess,
    TaggedMap<Array<uint64_t, uint64_t>, MultiThreading>, uniform, false,
    AMAC<16>)
    ->Apply(RandomAccessBenchmark)
    ->ThreadRange(1, 32);
#endif
#endif

BENCHMARK_TEMPLATE(
    BM_RandomAccess, ankerl::unordered_dense::map<uint64_t, uint64_t>, zipf)
    ->Apply(RandomAccessBenchmark);
BENCHMARK_TEMPLATE(
    BM_RandomAccess, std::unordered_map<uint64_t, uint64_t>, zipf)
    ->Apply(RandomAccessBenchmark);
BENCHMARK_TEMPLATE(BM_RandomAccess, NeighborHash, zipf)
    ->Apply(RandomAccessBenchmark)
    ->ThreadRange(1, 32);
BENCHMARK_TEMPLATE(
    BM_RandomAccess, absl::flat_hash_map<uint64_t, uint64_t>, zipf)
    ->Apply(RandomAccessBenchmark);

// uniform
BENCHMARK_TEMPLATE(BM_RandomAccess,
    TaggedMap<Array<uint64_t, uint64_t>, Scalar, IntraVec, Vec>, uniform)
    ->Apply(RandomAccessBenchmark);
BENCHMARK_TEMPLATE(BM_RandomAccess,
    TaggedMap<ankerl::unordered_dense::map<uint64_t, uint64_t>, Scalar>,
    uniform)
    ->Apply(RandomAccessBenchmark);
BENCHMARK_TEMPLATE(BM_RandomAccess,
    TaggedMap<std::unordered_map<uint64_t, uint64_t>, Scalar>, uniform)
    ->Apply(RandomAccessBenchmark);
BENCHMARK_TEMPLATE(
    BM_RandomAccess, TaggedMap<NeighborHash, Scalar, IntraVec, Vec>, uniform)
    ->Apply(RandomAccessBenchmark);
BENCHMARK_TEMPLATE(BM_RandomAccess,
    TaggedMap<absl::flat_hash_map<uint64_t, uint64_t>, IntraVec, Vec>, uniform)
    ->Apply(RandomAccessBenchmark);
BENCHMARK_TEMPLATE(BM_RandomAccess, TaggedMap<LinearProbing, Scalar>, uniform)
    ->Apply(RandomAccessBenchmark);
BENCHMARK_TEMPLATE(BM_RandomAccess,
    TaggedMap<ska::bytell_hash_map<uint64_t, uint64_t, absl::Hash<uint64_t>>,
        Scalar>,
    uniform)
    ->Apply(RandomAccessBenchmark);
BENCHMARK_TEMPLATE(BM_RandomAccess,
    TaggedMap<emhash7::HashMap<uint64_t, uint32_t, absl::Hash<uint64_t>>,
        Scalar>,
    uniform)
    ->Apply(RandomAccessBenchmark);

#ifdef BENCHMARK_BOOST_FLAT_MAP
BENCHMARK_TEMPLATE(BM_RandomAccess,
    TaggedMap<boost::unordered::unordered_flat_map<uint64_t, uint64_t,
                  absl::Hash<uint64_t>>,
        IntraVec, Vec>,
    uniform)
    ->Apply(RandomAccessBenchmark);
#endif

BENCHMARK_TEMPLATE(BM_RandomAccess,
    TaggedMap<tsl::hopscotch_map<uint64_t, uint64_t, absl::Hash<uint64_t>>,
        Scalar>,
    uniform)
    ->Apply(RandomAccessBenchmark);

BENCHMARK_TEMPLATE(BM_RandomAccess,
    TaggedMap<absl::flat_hash_map<uint64_t, uint64_t>, MultiThreading>, uniform)
    ->Apply(RandomAccessBenchmark)
    ->ThreadRange(1, 32);

#ifdef NEIGHBOR_HASH_SIMD_FIND
BENCHMARK_TEMPLATE(
    BM_RandomAccess, TaggedMap<BucketingSIMD_16x16, IntraVec, Vec>, uniform)
    ->Apply(RandomAccessBenchmark);
BENCHMARK_TEMPLATE(
    BM_RandomAccess, TaggedMap<BucketingSIMD_8x16, IntraVec, Vec>, uniform)
    ->Apply(RandomAccessBenchmark);
BENCHMARK_TEMPLATE(
    BM_RandomAccess, TaggedMap<BucketingSIMD_16x8, IntraVec, Vec>, uniform)
    ->Apply(RandomAccessBenchmark);
#endif

// BENCHMARK_TEMPLATE(BM_RandomAccess, MultiShard<absl::flat_hash_map<uint64_t, uint64_t>, 4>, uniform, false)->Apply(RandomAccessBenchmark);

// once V.S. query-guide-optimization
BENCHMARK_TEMPLATE(BM_RandomAccess, NeighborHash, once, false)
    ->Apply(RandomAccessBenchmark);
BENCHMARK_TEMPLATE(BM_RandomAccess, NeighborHash, once, true)
    ->Apply(RandomAccessBenchmark);

// zipf with query-guide-optimization
BENCHMARK_TEMPLATE(BM_RandomAccess, TaggedMap<NeighborHash, QFGO>, zipf, true)
    ->Apply(RandomAccessBenchmark);
BENCHMARK_TEMPLATE(BM_RandomAccess, TaggedMap<NeighborHash, QFGO>, zipf)
    ->Apply(RandomAccessBenchmark);

#ifdef ABSL_HAVE_PREFETCH
BENCHMARK_TEMPLATE(
    BM_RandomAccess, TaggedMap<NeighborHash, QFGO>, zipf, true, AMAC<64>)
    ->Apply(RandomAccessBenchmark);
BENCHMARK_TEMPLATE(
    BM_RandomAccess, TaggedMap<NeighborHash, QFGO>, zipf, false, AMAC<64>)
    ->Apply(RandomAccessBenchmark);

#ifdef NEIGHBOR_HASH_SIMD_FIND
BENCHMARK_TEMPLATE(
    BM_RandomAccess, TaggedMap<NeighborHash, QFGO>, zipf, true, AMAC<16>, true)
    ->Apply(RandomAccessBenchmark);
BENCHMARK_TEMPLATE(
    BM_RandomAccess, TaggedMap<NeighborHash, QFGO>, zipf, false, AMAC<16>, true)
    ->Apply(RandomAccessBenchmark);
#endif
#endif

BENCHMARK_TEMPLATE(BM_RandomAccess,
    neighbor::LinearProbingHashMap<uint64_t, uint64_t, MyPolicyTraits>, once)
    ->Apply(RandomAccessBenchmark);

// BENCHMARK_TEMPLATE(BM_RandomAccess, NeighborHashMap<uint64_t, uint64_t, MyPolicyTraits>, uniform, true)->Apply(RandomAccessBenchmark);

#ifdef BENCHMARK_CUCKOO_HASHMAP
BENCHMARK_TEMPLATE(BM_RandomAccess,
    libcuckoo::cuckoohash_map<uint64_t, uint64_t, absl::Hash<uint64_t>>, zipf)
    ->Apply(RandomAccessBenchmark);
#endif  // BENCHMARK_CUCKOO_HASHMAP

#ifdef NEIGHBOR_HASH_COROUTINE_FIND
// BENCHMARK_TEMPLATE(BM_RandomAccess, NeighborHashMap<uint64_t, uint64_t, MyPolicyTraits>, uniform, false, AMAC<0>, false, 32)->Apply(RandomAccessBenchmark);
#endif  // NEIGHBOR_HASH_COROUTINE_FIND

// AMAC
#ifdef ABSL_HAVE_PREFETCH
BENCHMARK_TEMPLATE(BM_RandomAccess, NeighborHash, zipf, false, AMAC<32>)
    ->Apply(RandomAccessBenchmark)
    ->ThreadRange(1, 32);
BENCHMARK_TEMPLATE(BM_RandomAccess, NeighborHash, zipf, true, AMAC<32>)
    ->Apply(RandomAccessBenchmark)
    ->ThreadRange(1, 32);

BENCHMARK_TEMPLATE(BM_RandomAccess, NeighborHash, uniform, false, AMAC<8>)
    ->Apply(RandomAccessBenchmark);
BENCHMARK_TEMPLATE(BM_RandomAccess, NeighborHash, uniform, false, AMAC<16>)
    ->Apply(RandomAccessBenchmark);
BENCHMARK_TEMPLATE(BM_RandomAccess, NeighborHash, uniform, false, AMAC<32>)
    ->Apply(RandomAccessBenchmark);
BENCHMARK_TEMPLATE(BM_RandomAccess, NeighborHash, uniform, false, AMAC<64>)
    ->Apply(RandomAccessBenchmark);

BENCHMARK_TEMPLATE(BM_RandomAccess, NeighborHash, uniform, false, AMAC<32>)
    ->Apply(RandomAccessBenchmark)
    ->ThreadRange(1, 32);

BENCHMARK_TEMPLATE(BM_RandomAccess, LinearProbing, uniform, false, AMAC<8>)
    ->Apply(RandomAccessBenchmark);
BENCHMARK_TEMPLATE(BM_RandomAccess, LinearProbing, uniform, false, AMAC<16>)
    ->Apply(RandomAccessBenchmark);
BENCHMARK_TEMPLATE(BM_RandomAccess, LinearProbing, uniform, false, AMAC<32>)
    ->Apply(RandomAccessBenchmark);
BENCHMARK_TEMPLATE(BM_RandomAccess, LinearProbing, uniform, false, AMAC<64>)
    ->Apply(RandomAccessBenchmark);
#endif

// multi-threading
BENCHMARK_TEMPLATE(BM_RandomAccess,
    NeighborHashMap<uint64_t, uint64_t, MyPolicyTraits>, uniform, false)
    ->Apply(RandomAccessBenchmark)
    ->ThreadRange(1, 32);

#ifdef NEIGHBOR_HASH_SIMD_FIND

BENCHMARK_TEMPLATE(BM_RandomAccess, TaggedMap<NeighborHash, Vec>, uniform,
    false, AMAC<0>, true)
    ->Apply(RandomAccessBenchmark);
BENCHMARK_TEMPLATE(BM_RandomAccess,
    neighbor::BucketingSIMDHashTable<uint64_t, uint64_t, MyPolicyTraits,
        neighbor::Fingerprint_16x16>,
    zipf)
    ->Apply(RandomAccessBenchmark);

#ifdef ABSL_HAVE_PREFETCH
BENCHMARK_TEMPLATE(BM_RandomAccess, TaggedMap<BucketingSIMD_16x16, Vec>,
    uniform, false, AMAC<8>)
    ->Apply(RandomAccessBenchmark);
BENCHMARK_TEMPLATE(BM_RandomAccess, TaggedMap<BucketingSIMD_16x16, Vec>,
    uniform, false, AMAC<16>)
    ->Apply(RandomAccessBenchmark);
BENCHMARK_TEMPLATE(BM_RandomAccess, TaggedMap<BucketingSIMD_16x16, Vec>,
    uniform, false, AMAC<32>)
    ->Apply(RandomAccessBenchmark);

BENCHMARK_TEMPLATE(BM_RandomAccess,
    TaggedMap<BucketingSIMD_16x16, MultiThreading>, uniform, false, AMAC<16>)
    ->Apply(RandomAccessBenchmark)
    ->ThreadRange(1, 32);

// SIMD+AMAC
BENCHMARK_TEMPLATE(BM_RandomAccess, TaggedMap<NeighborHash, MultiThreading>,
    uniform, false, AMAC<16>, true)
    ->Apply(RandomAccessBenchmark)
    ->ThreadRange(1, 32);
BENCHMARK_TEMPLATE(BM_RandomAccess, TaggedMap<NeighborHash, Vec>, uniform,
    false, AMAC<8>, true)
    ->Apply(RandomAccessBenchmark);
BENCHMARK_TEMPLATE(BM_RandomAccess, TaggedMap<NeighborHash, Vec>, uniform,
    false, AMAC<16>, true)
    ->Apply(RandomAccessBenchmark);
BENCHMARK_TEMPLATE(BM_RandomAccess, TaggedMap<NeighborHash, Vec>, uniform,
    false, AMAC<32>, true)
    ->Apply(RandomAccessBenchmark);
BENCHMARK_TEMPLATE(BM_RandomAccess, TaggedMap<NeighborHash, Vec>, uniform,
    false, AMAC<64>, true)
    ->Apply(RandomAccessBenchmark);

BENCHMARK_TEMPLATE(
    BM_RandomAccess, TaggedMap<NeighborHash, Vec>, zipf, false, AMAC<32>, true)
    ->Apply(RandomAccessBenchmark);

BENCHMARK_TEMPLATE(BM_HashJoinProbing,
    NeighborHashMap<uint64_t, uint64_t, MyPolicyTraits>, AMAC<32>, true);
#endif

// SIMD probing
BENCHMARK_TEMPLATE(BM_HashJoinProbing,
    NeighborHashMap<uint64_t, uint64_t, MyPolicyTraits>, AMAC<0>, true);

#endif

#ifdef ABSL_HAVE_PREFETCH
BENCHMARK_TEMPLATE(BM_HashJoinProbing,
    NeighborHashMap<uint64_t, uint64_t, MyPolicyTraits>, AMAC<32>, false);
BENCHMARK_TEMPLATE(BM_HashJoinProbing,
    NeighborHashMap<uint64_t, uint64_t, MyPolicyTraits>, AMAC<64>, false);
#endif

BENCHMARK_TEMPLATE(BM_HashJoinProbing, absl::flat_hash_map<uint64_t, uint64_t>,
    AMAC<0>, false);

// BENCHMARK_TEMPLATE(BM_RandomAccess, absl::flat_hash_map<uint64_t, uint64_t>)->ArgsProduct({{1 << 24}, {30, 90}})->ThreadRange(1, 64)->UseRealTime();
// BENCHMARK_TEMPLATE(BM_RandomAccess, NeighborHashMap<uint64_t, uint64_t, MyPolicyTraits>)->ArgsProduct({{1 << 24}, {30, 90}})->ThreadRange(1, 64)->UseRealTime();

// BENCHMARK_TEMPLATE(BM_RandomAccess, absl::flat_hash_map<uint64_t, uint64_t>)->ArgsProduct({{1 << 30}, {30, 90}});
// BENCHMARK_TEMPLATE(BM_RandomAccess, NeighborHashMap<uint64_t, uint64_t, MyPolicyTraits>)->ArgsProduct({{1 << 30}, {30, 90}});

BENCHMARK_TEMPLATE(BM_RandomChasing, true)->Ranges({{1UL << 10, 1UL << 24}});
BENCHMARK_TEMPLATE(BM_RandomChasing, false)->Ranges({{1UL << 10, 1UL << 24}});

BENCHMARK_MAIN();
