#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <iostream>
#include <iterator>
#include <map>
#include <new>
#include <set>
#include <utility>

#include "absl/base/prefetch.h"

#include "neighbor_hash/slot_type.h"

namespace neighbor {

template <class K, class V, class PolicyTraits>
class LinearProbingHashMap {
 public:
  static constexpr size_t kValueBitCount = sizeof(V) * 8;
  static constexpr size_t kOffsetBitCount =
      kValueBitCount - PolicyTraits::kPayloadBitCount;
  static constexpr size_t kPayloadBitCount = PolicyTraits::kPayloadBitCount;

  static constexpr V kPayloadMask = V(-1) >> kOffsetBitCount;
  static constexpr V kOffsetMask = (V(-1) >> PolicyTraits::kPayloadBitCount)
      << PolicyTraits::kPayloadBitCount;
  static constexpr int kUnOccupied = 0;
  static constexpr int kOccupied = 1;
  static constexpr int kInvalidOffset = 2;

  static constexpr K kUnOccupiedKey = 0;
  static constexpr V kUnOccupiedValue = 0;

  using key_type = K;
  using mapped_type = V;

  using slot_type = detail::Slot<LinearProbingHashMap>;
  using payload_proxy_type = typename slot_type::payload_proxy_type;

  class iterator {
    friend class LinearProbingHashMap;

   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = typename slot_type::value_type;
    using reference = value_type&;
    using pointer = value_type*;
    using difference_type = ptrdiff_t;

    iterator() {}

    // PRECONDITION: not an end() iterator.
    reference operator*() const { return slot_->value(); }

    // PRECONDITION: not an end() iterator.
    pointer operator->() const { return &operator*(); }

    // PRECONDITION: not an end() iterator.
    iterator& operator++() {
      ++slot_;
      skip_unoccupied();
      return *this;
    }

    // PRECONDITION: not an end() iterator.
    iterator operator++(int) {
      auto tmp = *this;
      ++*this;
      return tmp;
    }

    friend bool operator==(const iterator& a, const iterator& b) {
      return a.slot_ == b.slot_;
    }

    friend bool operator!=(const iterator& a, const iterator& b) {
      return !(a == b);
    }

   private:
    iterator(slot_type* slot, const LinearProbingHashMap* container)
        : slot_(slot), container_(container) {}

    void skip_unoccupied() {
      if (slot_->offset() == kInvalidOffset) {
        slot_ = nullptr;
        return;
      }

      while (!container_->is_occupied_slot(slot_ - container_->slots_)) {
        slot_ += 1;
        if (slot_->offset() == kInvalidOffset) {
          slot_ = nullptr;
          return;
        }
      }
    }

    slot_type* slot_ = nullptr;
    const LinearProbingHashMap* container_ = nullptr;
  };

  iterator begin() const {
    for (size_t i = 0; i < capacity_; ++i) {
      if (is_occupied_slot(i)) {
        return iterator{&slots_[i], this};
      }
    }
    return end();
  }

  iterator end() const { return iterator{nullptr, this}; }

  explicit LinearProbingHashMap(size_t capacity) {
    capacity_ = PolicyTraits::NormalizeCapacity(capacity);
    slots_ = (slot_type*)PolicyTraits::template Allocate<64>(
        sizeof(slot_type) * (capacity_ + 1));
    for (size_t i = 0; i < capacity_ + 1; ++i) {
      new (&slots_[i]) slot_type;
    }
    slots_[capacity_].set_sentinel();
    size_ = 0;
  }

  LinearProbingHashMap() : LinearProbingHashMap(8) {}

  ~LinearProbingHashMap() {
    if (slots_) {
      PolicyTraits::Deallocate(slots_, sizeof(slot_type) * (capacity_ + 1));
    }
    slots_ = nullptr;
  }

  size_t size() const { return size_; }

  size_t bucket_count() const { return capacity_; }

  bool empty() const { return size() == 0; }

  float load_factor() const { return static_cast<float>(size_) / capacity_; }

  void reserve(size_t n) { rehash(n); }

  void rehash(size_t n) {
    if (capacity_ >= n) {
      return;
    }
    LinearProbingHashMap new_hash(n);
    for (size_t i = 0; i < capacity_; ++i) {
      if (is_occupied_slot(i)) {
        new_hash.emplace(slots_[i].key(), slots_[i].payload());
      }
    }

    std::swap(size_, new_hash.size_);
    std::swap(capacity_, new_hash.capacity_);
    std::swap(slots_, new_hash.slots_);
  }

  std::pair<iterator, bool> emplace(K key, V value) {
    return emplace_impl<false>(key, value);
  }

  std::pair<iterator, bool> insert(K key, V value) {
    return emplace_impl<false>(key, value);
  }

  std::pair<iterator, bool> insert_or_assign(K key, V value) {
    return emplace_impl<true>(key, value);
  }

  int64_t SlotIndex(K key) const { return hash2slot_(hash_(key), capacity_); }

  inline iterator find(K key) const {
    int64_t slot_index = SlotIndex(key);
    auto* slots = slots_;
    auto slot = slots[slot_index];
    while (slot.offset() != kUnOccupied) {
      if (slot.key() == key) {
        return iterator{&slots[slot_index], this};
      }

      slot_index += slot.offset();
      slot_index &= capacity_ - 1;
      slot = slots[slot_index];
    }
    return end();
  }

  payload_proxy_type& operator[](K key) {
    return emplace(key, V()).first->second;
  }

  void clear() {
    for (size_t i = 0; i < capacity_; ++i) {
      slots_[i].reset();
    }
    size_ = 0;
  }

  void erase(iterator it) { abort(); }

  size_t erase(K k) { abort(); }

  static constexpr size_t kCachelineMask = ~(64UL - 1);

  bool is_one_cacheline_access(K key) const {
    int slot_index = SlotIndex(key);
    if (slots_[slot_index].key() == key) {
      return false;
    }

    auto cacheline_id = ((intptr_t)(&slots_[slot_index]) & kCachelineMask);
    int count = 1;

    while (slots_[slot_index].offset() != kUnOccupied &&
        slots_[slot_index].key() != key) {
      slot_index += 1;
      slot_index &= capacity_ - 1;
      if (cacheline_id != ((intptr_t)(&slots_[slot_index]) & kCachelineMask)) {
        count += 1;
        cacheline_id = ((intptr_t)(&slots_[slot_index]) & kCachelineMask);
      }
    }
    return count == 2;
  }

  void count_hit_keys_cacheline_access() const {
    size_t total_cacheline_access = 0;
    size_t reprobed_keys = 0;
    size_t probing_count = 0;

    std::map<size_t, size_t> cacheline_sizes;
    for (size_t i = 0; i < capacity_; ++i) {
      if (slots_[i].offset() != kOccupied) {
        continue;
      }

      probing_count += 1;
      int slot_index = SlotIndex(slots_[i].key());
      if (slot_index != i) {
        reprobed_keys += 1;
        std::set<size_t> cachelines;
        while (slot_index != i) {
          cachelines.insert((intptr_t)(&slots_[slot_index]) & kCachelineMask);
          slot_index += 1;
          slot_index &= capacity_ - 1;
          probing_count += 1;
        }
        cachelines.insert((intptr_t)(&slots_[slot_index]) & kCachelineMask);

        total_cacheline_access += cachelines.size();
        cacheline_sizes[cachelines.size()] += 1;
      } else {
        total_cacheline_access += 1;
        cacheline_sizes[1] += 1;
      }
    }

    std::cout << "cachelines" << std::endl;
    for (auto& it : cacheline_sizes) {
      std::cout << it.first << ":" << it.second << std::endl;
    }
    std::cout << "size:" << size_
              << " total_cacheline_access:" << total_cacheline_access
              << " average:" << double(total_cacheline_access) / size_
              << " reprobed_keys:" << reprobed_keys
              << " probing_count:" << probing_count << std::endl;
  }

#ifdef ABSL_HAVE_PREFETCH
  struct AMAC_State {
    K key;
    slot_type* slot;
  };

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
      state.slot = &slots_[SlotIndex(key)];
    }

    int state_circular_buffer_index = 0;
    int state_circular_buffer_stop_index = -1;
    constexpr int kExpandSize = 4;
    while (keys_index < keys_size) {
      auto* state_slice =
          &states[state_circular_buffer_index & (kPipelineSize - 1)];
      for (int i = 0; i < kExpandSize; ++i) {
        auto& state = state_slice[i];
      try_find:
        if (state.slot->key() == state.key) {
          fn(state.key, state.slot->payload());
        } else {
          if (state.slot->offset() == kUnOccupied) {
            // do nothing
          } else {
            auto* prev_slot = state.slot;
            state.slot += 1;
            if (intptr_t(state.slot) / 64 == intptr_t(prev_slot) / 64) {
              // same cacheline
              goto try_find;
            }
            absl::PrefetchToLocalCache(state.slot);
            continue;
          }
        }
        // fetch next key
        if (keys_index >= keys_size) {
          state_circular_buffer_stop_index =
              (state_circular_buffer_index + i) & (kPipelineSize - 1);
          break;
        }
        auto key = keys[keys_index];
        int64_t slot_index = SlotIndex(key);
        keys_index += 1;
        state.key = key;
        state.slot = &slots_[slot_index];
        absl::PrefetchToLocalCache(state.slot);
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
      while (state.slot->key() != state.key) {
        if (state.slot->offset() == kUnOccupied) {
          goto next;
        }
        state.slot += 1;
      }
      fn(state.key, state.slot->payload());
    next:;
    }
  }
#endif  // ABSL_HAVE_PREFETCH

 private:
  bool is_occupied_slot(size_t index) {
    return slots_[index].offset() == kOccupied;
  }

  template <bool update_if_exist>
  std::pair<iterator, bool> emplace_impl(K key, V value) {
    int64_t slot_index = SlotIndex(key);
    while (true) {
      auto& slot = slots_[slot_index];
      if (slot.offset() == kUnOccupied) {
        slot.set_key(key);
        slot.set_payload(value);
        slot.set_offset(kOccupied);
        size_ += 1;
        return {iterator{&slot, this}, true};
      }

      if (slot.key() == key) {
        if constexpr (update_if_exist) {
          slot.set_payload(value);
        }
        return {iterator{&slot, this}, false};
      }

      if (load_factor() > 0.8) {
        rehash(capacity_ * 2);
        return emplace_impl<update_if_exist>(key, value);
      }

      slot_index += 1;
      slot_index &= capacity_ - 1;
    }
  }

  typename PolicyTraits::Hash hash_;
  typename PolicyTraits::Hash2Slot hash2slot_;

  slot_type* slots_;
  size_t size_;
  size_t capacity_;
};

}  // namespace neighbor
