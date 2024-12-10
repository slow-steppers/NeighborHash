#include <stddef.h>    // for size_t, NULL
#include <stdint.h>    // for uint64_t, int64_t
#include <sys/time.h>  // for gettimeofday, timeval

#include <iostream>  // for operator<<, basic_...
#include <limits>    // for numeric_limits
#include <random>    // for seed_seq
#include <thread>    // for thread
#include <vector>    // for vector

#include "absl/random/random.h"                    // for pcg128_params, pcg...
#include "absl/random/uniform_int_distribution.h"  // for uniform_int_distri...
#include "folly/synchronization/Rcu.h"             // for synchronize_rcu

#include "neighbor_hash/common_policy.h"  // for DefaultPolicy
#include "neighbor_hash/neighbor_hash.h"  // for AtomicNeighborHashMap

inline double gettime() {
  struct timeval now_tv;
  gettimeofday(&now_tv, NULL);
  return ((double)now_tv.tv_sec) + ((double)now_tv.tv_usec) / 1000000.0;
}

template <class T>
std::vector<T> GenerateRandomNumbers(int count, T range, int seed = 3) {
  std::vector<T> numbers;
  numbers.reserve(count);
  std::seed_seq sseq{1, 2, seed};
  absl::InsecureBitGen gen(sseq);
  absl::uniform_int_distribution<T> distribution(0, range);
  for (int i = 0; i < count; ++i) {
    numbers.push_back(distribution(gen));
  }
  return numbers;
}

template <int AMACWindow = 0>
class BM_MultiThreading {
 public:
  void SetUp(size_t total_dataset_size, double hit_ratio, double load_factor) {
    size_t dataset_size = total_dataset_size * load_factor;
    size_t keys_count = dataset_size / hit_ratio;
    numbers_ =
        GenerateRandomNumbers(keys_count, std::numeric_limits<uint64_t>::max());

    std::seed_seq sseq{1, 2, 3};
    absl::InsecureBitGen gen(sseq);

    map_.reserve(dataset_size);
    for (size_t i = 0; i < dataset_size; ++i) {
      auto num = numbers_[i];
      int value = absl::uniform_int_distribution<int>(0, dataset_size)(gen);
      map_[num] = value;
    }
  }

  void TearDown() {}

  void BenchmarkInsertAndLookup(int thread_index) {
    std::seed_seq sseq{1, 2, thread_index};
    absl::InsecureBitGen gen(sseq);

    if (thread_index == 0) {
      // writer, insert:update = 1:99
      auto insertion_numbers = GenerateRandomNumbers(
          numbers_.size(), std::numeric_limits<uint64_t>::max(), 99);
      size_t insertion_index = 0;

      std::vector<uint64_t> update_numbers;
      for (size_t i = 0; i < numbers_.size(); ++i) {
        size_t sampled_index =
            absl::uniform_int_distribution<size_t>(0, numbers_.size())(gen);
        auto key = numbers_[sampled_index];
        update_numbers.emplace_back(key);
      }

      constexpr int kMaxDelayedSize = 32;
      std::cout << "writer start:" << std::endl;
      for (int x = 0; x < 1024; ++x) {
        auto begin = gettime();
        int64_t inserted = 0;
        int64_t updated = 0;
        std::vector<uint64_t> delayed_insertions;
        for (int xx = 0; xx < 8; ++xx) {
          for (size_t xxx = 0; xxx < update_numbers.size(); xxx += 1) {
            auto key = update_numbers[xxx];
            if (xxx % 100 == 0) {
              key = insertion_numbers[insertion_index++];
            }
            if (insertion_index >= insertion_numbers.size()) {
              std::cout << "run out of insertions" << std::endl;
              return;
            }
            auto st = map_.atomic_insert_or_update<false>(key, key + 1);
            if (st == atomic_hashmap::Status::kInserted) {
              inserted += 1;
            } else if (st == atomic_hashmap::Status::kUpdated) {
              updated += 1;
            } else {
              delayed_insertions.push_back(key);
              if (st == atomic_hashmap::Status::kPrepareFailed) {
                std::cout << "Failed to prepare" << std::endl;
                return;
              }
            }
            if (delayed_insertions.size() > kMaxDelayedSize) {
              folly::synchronize_rcu();
            }

            for (auto key : delayed_insertions) {
              auto st = map_.atomic_insert_or_update<true>(key, key + 1);
              if (st == atomic_hashmap::Status::kInserted) {
                inserted += 1;
              } else if (st == atomic_hashmap::Status::kUpdated) {
                updated += 1;
              } else {
                std::cout << "Failed to insert" << std::endl;
                return;
              }
            }
            delayed_insertions.clear();
          }

          folly::synchronize_rcu();
          for (auto key : delayed_insertions) {
            auto st = map_.atomic_insert_or_update<true>(key, key + 1);
            if (st == atomic_hashmap::Status::kInserted) {
              inserted += 1;
            } else if (st == atomic_hashmap::Status::kUpdated) {
              updated += 1;
            } else {
              std::cout << "Failed to insert" << std::endl;
              return;
            }
          }
          delayed_insertions.clear();
        }
        auto end = gettime();
        std::cout << "writer speed:"
                  << (inserted + updated) / 1e6 / (end - begin)
                  << " M-ops/s inserted/updated:"
                  << static_cast<float>(inserted) / updated
                  << " load factor:" << map_.load_factor() << std::endl;
      }
    } else {
      // reader
      std::vector<uint64_t> access_numbers;

      for (size_t i = 0; i < numbers_.size(); ++i) {
        size_t sampled_index =
            absl::uniform_int_distribution<size_t>(0, numbers_.size())(gen);
        auto key = numbers_[sampled_index];
        access_numbers.push_back(key);
      }

      size_t total_size = 0;
      uint64_t sum = 0;

      constexpr int kBatchSize = 1024;
      std::cout << "reader start:" << access_numbers.size() << std::endl;
      for (int x = 0; x < 1024; ++x) {
        constexpr int loops = 7;
        auto begin = gettime();
        for (int xx = 0; xx < loops; ++xx) {
          for (size_t xxx = 0; xxx < access_numbers.size() - kBatchSize;
               xxx += kBatchSize) {
            folly::rcu_reader rcu_guard;
            const auto* data = &access_numbers[xxx];
            if constexpr (AMACWindow != 0) {
#ifdef NEIGHBOR_HASH_SIMD_FIND
              map_.template simd_amac_find<AMACWindow>(
                  data, kBatchSize, [&sum](auto, uint64_t v) { sum ^= v; });
#else
              map_.template amac_find<AMACWindow>(
                  data, kBatchSize, [&sum](auto, uint64_t v) { sum ^= v; });
#endif
            } else {
#ifdef NEIGHBOR_HASH_SIMD_FIND
              map_.template simd_find(
                  data, kBatchSize, [&sum](auto, uint64_t v) { sum ^= v; });
#else
              map_.find(
                  data, kBatchSize, [&sum](auto, uint64_t v) { sum ^= v; });
#endif
            }
          }
        }
        auto end = gettime();
        if (thread_index == 1) {
          std::cout << "reader speed:"
                    << loops * access_numbers.size() / 1e6 / (end - begin)
                    << " M-ops/s"
                    << " sum:" << sum << std::endl;
        }
      }
    }
  }

  using MyPolicyTraits = neighbor::policy::DefaultPolicy<uint64_t, uint64_t>;
  using atomic_hashmap =
      neighbor::AtomicNeighborHashMap<uint64_t, uint64_t, MyPolicyTraits>;

 protected:
  atomic_hashmap map_;
  std::vector<uint64_t> numbers_;
};

int main() {
  BM_MultiThreading<16> bm;
  bm.SetUp(1 << 27UL, 0.9, 0.51);
  std::cout << "setup" << std::endl;
  std::vector<std::thread> threads;
  for (int i = 0; i < 16; ++i) {
    threads.emplace_back([&bm, i] { bm.BenchmarkInsertAndLookup(i); });
  }

  for (auto& th : threads) {
    th.join();
  }
}
