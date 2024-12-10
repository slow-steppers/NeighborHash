#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/base/prefetch.h"

#include <emmintrin.h>
#include <immintrin.h>
#include <xmmintrin.h>

namespace neighbor {

#define PREDICT_TURE(exp) (__builtin_expect((exp), 1))

class Fingerprint_16x16 {
 public:
  using MaskType = __mmask16;

  static MaskType compare(
      uint16_t key_fingerprint, const uint16_t* fingerprints) {
    __m256i vfingerprints = _mm256_loadu_epi16(fingerprints);
    __m256i vcompare = _mm256_set1_epi16(key_fingerprint);

    return _mm256_cmpeq_epi16_mask(vfingerprints, vcompare);
  }

  static int last_index(const uint16_t* fingerprints) {
    __m256i vfingerprints = _mm256_loadu_epi16(fingerprints);
    __mmask16 mask = _mm256_movepi16_mask(vfingerprints);
    return __builtin_popcount(mask);
  }

  using FingerprintT = uint16_t;
  static constexpr int kNumFingerprints = 16;
};

class Fingerprint_8x16 {
 public:
  using MaskType = __mmask16;

  static MaskType compare(
      uint8_t key_fingerprint, const uint8_t* fingerprints) {
    __m128i vfingerprints = _mm_loadu_epi8(fingerprints);
    __m128i vcompare = _mm_set1_epi8(key_fingerprint);

    return _mm_cmpeq_epi8_mask(vfingerprints, vcompare);
  }

  static int last_index(const uint8_t* fingerprints) {
    __m128i vfingerprints = _mm_loadu_epi8(fingerprints);
    __mmask16 mask = _mm_movepi8_mask(vfingerprints);
    return __builtin_popcount(mask);
  }

  using FingerprintT = uint8_t;
  static constexpr int kNumFingerprints = 16;
};

class Fingerprint_16x8 {
 public:
  using MaskType = __mmask8;

  static MaskType compare(
      uint16_t key_fingerprint, const uint16_t* fingerprints) {
    __m128i vfingerprints = _mm_loadu_epi16(fingerprints);
    __m128i vcompare = _mm_set1_epi16(key_fingerprint);

    return _mm_cmpeq_epi16_mask(vfingerprints, vcompare);
  }

  static int last_index(const uint16_t* fingerprints) {
    __m128i vfingerprints = _mm_loadu_epi16(fingerprints);
    __mmask8 mask = _mm_movepi16_mask(vfingerprints);
    return __builtin_popcount(mask);
  }

  using FingerprintT = uint16_t;
  static constexpr int kNumFingerprints = 8;
};

template <typename K, typename V, typename FingerprintGroup>
class KeyValueAoSStoringBucket {
 public:
  KeyValueAoSStoringBucket() {
    std::fill(std::begin(fingerprints_), std::end(fingerprints_), 0);
    std::fill(std::begin(keys_values_), std::end(keys_values_),
        std::make_pair(K(), V()));
  }

  using MaskType = typename FingerprintGroup::MaskType;
  using FingerprintT = typename FingerprintGroup::FingerprintT;
  static constexpr int kNumFingerprints = FingerprintGroup::kNumFingerprints;

  static constexpr int kOverFlowed = -2;
  static constexpr int kSuspend = -3;
  static constexpr int kNotFound = -1;

  static constexpr FingerprintT kEmpty = 0;
  static constexpr FingerprintT kFullMask = FingerprintT(1)
      << (sizeof(FingerprintT) * 8 - 1);

  void reset() {
    for (uint64_t i = 0; i < kNumFingerprints; ++i) {
      fingerprints_[i] = 0;
      keys_values_[i] = {K(), V()};
    }
  }

  std::pair<int, int> count_cacheline(
      K key, FingerprintT fingerprint, bool* found) {
    static constexpr size_t kCachelineMask = ~(64UL - 1);
    std::set<intptr_t> cachelines;
    int probing_count = 1;
    cachelines.insert(intptr_t(&fingerprints_[0]) & kCachelineMask);
    auto vequal =
        FingerprintGroup::compare(fingerprint | kFullMask, &fingerprints_[0]);
    while (vequal) {
      int index_in_bucket = __builtin_ctz(vequal);
      auto& kv_pair = keys_values_[index_in_bucket];
      cachelines.insert(intptr_t(&kv_pair) & kCachelineMask);
      probing_count += 1;
      if (PREDICT_TURE((std::get<0>(kv_pair) == key))) [[likely]] {
        *found = true;
        return {cachelines.size(), probing_count};
      }
      vequal &= vequal - 1;
    }
    *found = false;
    return {cachelines.size(), probing_count};
  }

#ifdef ABSL_HAVE_PREFETCH
  int amac_compare(MaskType& mask, FingerprintT fingerprint, K key) {
    static constexpr int kInFirstCachlineSize =
        (64 - sizeof(fingerprints_)) / sizeof(std::pair<K, V>);
    static constexpr int kItemsPerCacheline = 64 / sizeof(std::pair<K, V>);
    int current_cacheline_boundary = kInFirstCachlineSize;
    int index = 0;
    if (mask == 0) {
      mask =
          FingerprintGroup::compare(fingerprint | kFullMask, &fingerprints_[0]);
      if (mask == 0) {
        goto end;
      }
    } else {
      index = __builtin_ctz(mask);
      current_cacheline_boundary = kInFirstCachlineSize +
          (index - kInFirstCachlineSize) / kItemsPerCacheline *
              kItemsPerCacheline;
      goto compare;
    }

    do {
      index = __builtin_ctz(mask);
      if (index < current_cacheline_boundary) {
      compare:
        mask &= mask - 1;
        auto& kv_pair = keys_values_[index];
        if (PREDICT_TURE((std::get<0>(kv_pair) == key))) [[likely]] {
          return index;
        }
      } else {
        absl::PrefetchToLocalCache(&keys_values_[index]);
        return kSuspend;
      }
    } while (mask);
  end:
    if (!is_overflowed()) {
      return kNotFound;
    }
    return kOverFlowed;
  }
#endif  // ABSL_HAVE_PREFETCH

  int find(K key, FingerprintT fingerprint) {
    auto vequal =
        FingerprintGroup::compare(fingerprint | kFullMask, &fingerprints_[0]);
    while (vequal) {
      int index_in_bucket = __builtin_ctz(vequal);
      auto& kv_pair = keys_values_[index_in_bucket];
      if (PREDICT_TURE((std::get<0>(kv_pair) == key))) [[likely]] {
        return index_in_bucket;
      }
      vequal &= vequal - 1;
    }

    // No match within the bucket, we can check next bucket in case we overflowed
    // However, first we check if we can terminate due to reaching the maximum limit or not overflowing

    if (PREDICT_TURE(!is_overflowed())) [[likely]] {
      return kNotFound;
    }

    // We overflowed, tell to not terminate search
    return kOverFlowed;
  }

  bool insert(K key, V value, FingerprintT fingerprint) {
    if (!is_overflowed()) {
      auto index = last_index_in_bucket() + 1;
      fingerprints_[index] = fingerprint | kFullMask;
      keys_values_[index] = {key, value};
      return true;
    }
    return false;
  }

  void update(int index_in_bucket, V value) {
    keys_values_[index_in_bucket].second = value;
  }

  bool is_overflowed() const {
    return fingerprints_[kNumFingerprints - 1] != kEmpty;
  }

  size_t size() const { return last_index_in_bucket() + 1; }

  int last_index_in_bucket() const {
    return FingerprintGroup::last_index(&fingerprints_[0]) - 1;
  }

  const std::pair<K, V>& value(int index) const { return keys_values_[index]; }

  std::pair<K, V>& value(int index) { return keys_values_[index]; }

 private:
  alignas(64) FingerprintT fingerprints_[kNumFingerprints];
  std::pair<K, V> keys_values_[kNumFingerprints];
};

template <typename K, typename V, typename PolicyTraits,
    typename FingerprintGroup>
class BucketingSIMDHashTable {
 public:
  static constexpr int kCachelineSize = 64;

  using BucketT = KeyValueAoSStoringBucket<K, V, FingerprintGroup>;
  using FingerprintT = typename FingerprintGroup::FingerprintT;

  BucketingSIMDHashTable(size_t capacity = 1024) {
    capacity_ = PolicyTraits::NormalizeCapacity(capacity);
    num_buckets_ = capacity_ / FingerprintGroup::kNumFingerprints;
    buckets_ = (BucketT*)PolicyTraits::template Allocate<kCachelineSize>(
        sizeof(BucketT) * num_buckets_);
    for (size_t i = 0; i < num_buckets_; ++i)
      buckets_[i].reset();
    size_ = 0;
  }

  ~BucketingSIMDHashTable() {
    PolicyTraits::Deallocate(buckets_, sizeof(BucketT) * num_buckets_);
  }

  class iterator {
    friend class BucketingSIMDHashTable;

   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::pair<K, V>;
    using reference = value_type&;
    using pointer = value_type*;
    using difference_type = ptrdiff_t;

    iterator() {}

    // PRECONDITION: not an end() iterator.
    reference operator*() const { return bucket_->value(index_in_bucket_); }

    // PRECONDITION: not an end() iterator.
    pointer operator->() const { return &operator*(); }

    // PRECONDITION: not an end() iterator.
    iterator& operator++() {
      abort();
      return *this;
    }

    // PRECONDITION: not an end() iterator.
    iterator operator++(int) { abort(); }

    friend bool operator==(const iterator& a, const iterator& b) {
      return a.bucket_ == b.bucket_ && a.index_in_bucket_ == b.index_in_bucket_;
    }

    friend bool operator!=(const iterator& a, const iterator& b) {
      return !(a == b);
    }

   private:
    iterator(BucketT* bucket, int index_in_bucket)
        : bucket_(bucket), index_in_bucket_(index_in_bucket) {}

    void skip_empty() {}

    BucketT* bucket_ = nullptr;
    int index_in_bucket_ = -1;
  };

  struct AMAC_State {
    K key;
    BucketT* bucket;
    FingerprintT fingerprint;
    typename BucketT::MaskType mask = 0;
  };

#ifdef ABSL_HAVE_PREFETCH
  template <int kPipelineSize, class Fn>
  void amac_find(const K* keys, int64_t keys_size, Fn&& fn) {
    AMAC_State states[kPipelineSize];
    int64_t keys_index = 0;

    if (kPipelineSize > keys_size) {
      while (keys_index < keys_size) {
        auto key = keys[keys_index];
        auto it = find(key);
        if (it == end()) {
        } else {
          fn(key, it->second);
        }
        keys_index += 1;
      }
      return;
    }

    // initial-fill
    for (int i = 0; i < kPipelineSize; ++i) {
      auto& state = states[i];
      auto key = keys[keys_index];
      keys_index += 1;
      state.key = key;
      state.bucket = &buckets_[BucketOf(key, &state.fingerprint)];
      state.mask = 0;
      absl::PrefetchToLocalCache(state.bucket);
    }

    int state_circular_buffer_index = 0;
    int state_circular_buffer_stop_index = -1;
    constexpr int kExpandSize = 4;
    while (keys_index < keys_size) {
      auto* state_slice =
          &states[state_circular_buffer_index & (kPipelineSize - 1)];
      for (int i = 0; i < kExpandSize; ++i) {
        auto& state = state_slice[i];
        int index = state.bucket->amac_compare(
            state.mask, state.fingerprint, state.key);
        if (index == BucketT::kSuspend) {
          continue;
        } else if (index == BucketT::kOverFlowed) {
          // probing next
          state.bucket += 1;
          state.bucket =
              &buckets_[(state.bucket - buckets_) & (num_buckets_ - 1)];
          absl::PrefetchToLocalCache(state.bucket);
          continue;
        } else if (index >= 0) {
          fn(state.key, state.bucket->value(index).second);
        }  // else : not found

        // fetch next key
        if (keys_index >= keys_size) {
          state_circular_buffer_stop_index =
              (state_circular_buffer_index + i) & (kPipelineSize - 1);
          break;
        }
        auto key = keys[keys_index];
        keys_index += 1;
        state.key = key;
        state.bucket = &buckets_[BucketOf(key, &state.fingerprint)];
        state.mask = 0;
        absl::PrefetchToLocalCache(state.bucket);
      }
      absl::PrefetchToLocalCache(&keys[keys_index]);

      state_circular_buffer_index += kExpandSize;
    }

    // finish all tasks
    for (int i = 0; i < kPipelineSize; ++i) {
      if (i == state_circular_buffer_stop_index) {
        continue;
      }
      auto& state = states[i];
      auto it = find(state.key);
      if (it != end()) {
        fn(state.key, it->second);
      }
    }
  }
#endif  // ABSL_HAVE_PREFETCH

  iterator find(K key) const {
    FingerprintT fingerprint;
    int64_t bucket_index = BucketOf(key, &fingerprint);
    auto found = const_cast<BucketingSIMDHashTable*>(this)->find_impl(
        key, fingerprint, bucket_index);
    if (found.second >= 0) {
      return iterator{found.first, found.second};
    }
    return end();
  }

  void clear() {
    for (size_t i = 0; i < num_buckets_; ++i) {
      buckets_[i].reset();
    }

    size_ = 0;
  }

  iterator end() const { return iterator{nullptr, -1}; }

  size_t size() { return size_; }

  bool empty() const { return size() == 0; }

  float load_factor() const { return static_cast<float>(size_) / capacity_; }

  void reserve(size_t n) { rehash(n); }

  void rehash(size_t n) {
    if (capacity_ >= n) {
      return;
    }
    BucketingSIMDHashTable new_hash(n);
    for (size_t i = 0; i < num_buckets_; ++i) {
      auto& bucket = buckets_[i];
      for (size_t x = 0; x < bucket.size(); ++x) {
        auto kv = bucket.value(x);
        new_hash[kv.first] = kv.second;
      }
    }

    std::swap(buckets_, new_hash.buckets_);
    std::swap(size_, new_hash.size_);
    std::swap(capacity_, new_hash.capacity_);
    std::swap(num_buckets_, new_hash.num_buckets_);
  }

  V& operator[](K key) { return emplace(key, V()).first->second; }

  std::pair<iterator, bool> emplace(K key, V value) {
    return emplace_impl<false>(key, value);
  }

  std::pair<iterator, bool> insert(K key, V value) {
    return emplace_impl<false>(key, value);
  }

  std::pair<iterator, bool> insert_or_assign(K key, V value) {
    return emplace_impl<true>(key, value);
  }

  int64_t BucketOf(K key, FingerprintT* fingerprint) const {
    uint64_t hash = hash_(key);
    *fingerprint = key >> (sizeof(K) * 8 - sizeof(FingerprintT) * 8);
    return hash & (num_buckets_ - 1);
  }

  std::pair<BucketT*, int> find_impl(
      K key, FingerprintT fingerprint, int64_t bucket_index) {
    while (true) {
      auto& bucket = buckets_[bucket_index];
      auto index_in_bucket = bucket.find(key, fingerprint);
      if (ABSL_PREDICT_TRUE(index_in_bucket != BucketT::kOverFlowed))
          [[likely]] {
        return {&bucket, index_in_bucket};
      }

      bucket_index += 1;
      bucket_index &= num_buckets_ - 1;
    }
    return {nullptr, -1};
  }

  template <bool update_if_exist>
  std::pair<iterator, bool> emplace_impl(K key, V value) {
    FingerprintT fingerprint;
    int64_t bucket_index = BucketOf(key, &fingerprint);

    auto result = find_impl(key, fingerprint, bucket_index);
    // Check if element already exists in hashmap; if so, update
    if (result.second >= 0) {
      if constexpr (update_if_exist) {
        result.first->update(key, value, result.second);
      }
      return {iterator{result.first, result.second}, false};
    }

    if (PolicyTraits::ShouldGrow(size_, capacity_)) {
      rehash(capacity_ * 2);
      return emplace_impl<update_if_exist>(key, value);
    }

    bucket_index = result.first - buckets_;
    while (true) {
      auto& bucket = buckets_[bucket_index];
      if (bucket.insert(key, value, fingerprint)) {
        size_ += 1;
        return {iterator{&bucket, bucket.last_index_in_bucket()}, true};
      }

      bucket_index += 1;
      bucket_index &= num_buckets_ - 1;
    }
    return {end(), false};
  }

  void count_hit_keys_cacheline_access() const {
    size_t total_cacheline_access = 0;
    size_t probing_count = 0;

    std::map<size_t, size_t> cacheline_sizes;
    for (size_t i = 0; i < num_buckets_; ++i) {
      auto& bucket = buckets_[i];
      for (int x = 0; x < bucket.size(); ++x) {
        auto key = bucket.value(x).first;
        FingerprintT fingerprint;
        int64_t bucket_index = BucketOf(key, &fingerprint);
        bool found = false;
        int c = 0;
        while (!found) {
          auto [cachlines, probes] =
              buckets_[bucket_index].count_cacheline(key, fingerprint, &found);
          total_cacheline_access += cachlines;
          c += cachlines;
          probing_count += probes;
          bucket_index += 1;
          bucket_index &= num_buckets_ - 1;
        }
        cacheline_sizes[c] += 1;
      }
    }

    std::cout << "cachelines" << std::endl;
    for (auto& it : cacheline_sizes) {
      std::cout << it.first << ":" << it.second << std::endl;
    }
    std::cout << "size:" << size_
              << " total_cacheline_access:" << total_cacheline_access
              << " average:" << double(total_cacheline_access) / size_
              << " probings:" << probing_count << std::endl;
  }

 protected:
  typename PolicyTraits::Hash hash_;

  alignas(kCachelineSize) BucketT* buckets_;

  size_t num_buckets_;
  size_t capacity_;
  size_t size_;
};

}  // namespace neighbor
