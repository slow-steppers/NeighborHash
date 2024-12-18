#pragma once

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <atomic>
#include <iostream>
#include <iterator>
#include <map>
#include <new>
#include <set>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/prefetch.h"

#include "neighbor_hash/slot_type.h"

#ifdef NEIGHBOR_HASH_SIMD_FIND
#include <immintrin.h>
#include <xmmintrin.h>
#endif  // NEIGHBOR_HASH_SIMD_FIND

#ifdef NEIGHBOR_HASH_COROUTINE_FIND
#include <coroutine>
#endif  // NEIGHBOR_HASH_COROUTINE_FIND

namespace neighbor {

namespace detail {

#ifdef ABSL_HAVE_PREFETCH
template <class K, class slot_type, bool>
struct AMAC_State {
  K key;
  slot_type* slot;
};

template <class K, class slot_type>
struct AMAC_State<K, slot_type, true> {
  int key_index;
  K key;
  slot_type* slot;
};
#endif  // ABSL_HAVE_PREFETCH

}  // namespace detail

template <class K, class V, class PolicyTraits,
    template <class> class SlotType = detail::Slot>
class alignas(64) NeighborHashMap {
 public:
  static_assert(std::is_unsigned<K>::value, "Unsigned K required.");
  static_assert(std::is_unsigned<V>::value, "Unsigned V required.");
  static_assert(sizeof(V) * 8 > PolicyTraits::kPayloadBitCount,
      "kPayloadBitCount too big");

  static constexpr size_t kValueBitCount = sizeof(V) * 8;
  static constexpr size_t kOffsetBitCount =
      kValueBitCount - PolicyTraits::kPayloadBitCount;
  static constexpr size_t kPayloadBitCount = PolicyTraits::kPayloadBitCount;

  static constexpr V kPayloadMask = V(-1) >> kOffsetBitCount;
  static constexpr V kOffsetMask = (V(-1) >> PolicyTraits::kPayloadBitCount)
      << PolicyTraits::kPayloadBitCount;
  static constexpr int kInvalidOffset = -(1 << (kOffsetBitCount - 1));
  static constexpr int kOffsetUpperBound = (1 << (kOffsetBitCount - 1));
  static constexpr K kUnOccupiedKey = 0;
  static constexpr V kUnOccupiedValue = 0;

  static constexpr size_t hardware_constructive_interference_size = 64;

  using key_type = K;
  using mapped_type = V;

  using slot_type = SlotType<NeighborHashMap>;
  using payload_proxy_type = typename slot_type::payload_proxy_type;

  class iterator {
    friend class NeighborHashMap;

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
    iterator(slot_type* slot, const NeighborHashMap* container)
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
    const NeighborHashMap* container_ = nullptr;
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

#ifdef ABSL_HAVE_PREFETCH
  template <int kPipelineSize, class Fn>
  void amac_find_with_index(const K* keys, int64_t keys_size, Fn&& fn) {
    amac_find<kPipelineSize, Fn, true>(keys, keys_size, std::move(fn));
  }

  template <int kPipelineSize, class Fn, bool kNeedKeyIndex = false>
  void amac_find(const K* keys, int64_t keys_size, Fn&& fn) {
    detail::AMAC_State<K, slot_type, kNeedKeyIndex> states[kPipelineSize];
    int64_t keys_index = 0;

    if (kPipelineSize > keys_size) {
      while (keys_index < keys_size) {
        auto key = keys[keys_index];
        auto it = find(key);
        if (it == end()) {
        } else {
          if constexpr (kNeedKeyIndex) {
            fn(keys_index, key, it->second);
          } else {
            fn(key, it->second);
          }
        }
        keys_index += 1;
      }
      return;
    }

    // initial-fill
    for (int i = 0; i < kPipelineSize; ++i) {
      auto& state = states[i];
      auto key = keys[keys_index];
      if constexpr (kNeedKeyIndex) {
        state.key_index = keys_index;
      }
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
          if constexpr (kNeedKeyIndex) {
            fn(state.key_index, state.key, state.slot->payload());
          } else {
            fn(state.key, state.slot->payload());
          }
        } else {
          int offset = state.slot->offset();
          if (offset == 0) {
            // do nothing
          } else {
            auto* prev_slot = state.slot;
            state.slot += offset;
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
        if constexpr (kNeedKeyIndex) {
          state.key_index = keys_index;
        }
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
        int offset = state.slot->offset();
        if (offset == 0) {
          goto next;
        }
        state.slot += offset;
      }
      if constexpr (kNeedKeyIndex) {
        fn(state.key_index, state.key, state.slot->payload());
      } else {
        fn(state.key, state.slot->payload());
      }
    next:;
    }
  }
#endif  // ABSL_HAVE_PREFETCH

#ifdef NEIGHBOR_HASH_SIMD_FIND
  struct SIMD_AMAC_State {
    __m512i vkey;
    __m512i vslot_index;
  };

  __m512i V_SlotIndex(__m512i key) const {
    return hash2slot_.v_hash2slot(hash_.v_hash_64(key), capacity_);
  }

  void compress_state(SIMD_AMAC_State* state, __mmask8 not_end) {
    state->vkey = _mm512_maskz_compress_epi64(not_end, state->vkey);
    state->vslot_index =
        _mm512_maskz_compress_epi64(not_end, state->vslot_index);
  }

  void expand_state(
      SIMD_AMAC_State* dst_state, SIMD_AMAC_State* src_state, __mmask8 is_end) {
    dst_state->vkey =
        _mm512_mask_expand_epi64(dst_state->vkey, is_end, src_state->vkey);
    dst_state->vslot_index = _mm512_mask_expand_epi64(
        dst_state->vslot_index, is_end, src_state->vslot_index);
  }

  template <class Fn>
  void simd_find(const std::vector<K>& keys, const Fn& fn) {
    return simd_find<Fn>(&keys[0], keys.size(), fn);
  }

  template <class Fn>
  void simd_find(const K* keys, int64_t keys_size, const Fn& fn) {
    constexpr int kSIMDWidth = 8;
    int64_t keys_index = 0;

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

    alignas(64) uint64_t tmp_values[kSIMDWidth];

    constexpr __mmask8 mask_map[] = {0, 1, 3, 7, 15, 31, 63, 127};
    SIMD_AMAC_State remained;
    int remained_count = 0;

    SIMD_AMAC_State state;

    __m512i vmatch_payload{0};
    int match_payload_cnt = 0;

    state.vkey = _mm512_load_epi64(&keys[keys_index]);
    state.vslot_index = V_SlotIndex(state.vkey);
    keys_index += kSIMDWidth;
    while (true) {
      __m512i vslot_index_by_byte = _mm512_slli_epi64(state.vslot_index, 4);
      __m512i vslot_key =
          _mm512_i64gather_epi64(vslot_index_by_byte, slots_, 1);
      __m512i vslot_offset_and_value = _mm512_i64gather_epi64(
          vslot_index_by_byte, reinterpret_cast<uint64_t*>(slots_) + 1, 1);

      __mmask8 vequal = _mm512_cmpeq_epi64_mask(vslot_key, state.vkey);
      size_t match_count = __builtin_popcount(vequal);
      if (match_count) {
        __m512i vpayload = _mm512_and_si512(
            vslot_offset_and_value, _mm512_set1_epi64(kPayloadMask));
        if (match_count + match_payload_cnt >= kSIMDWidth) {
          // vmatch_payload -> vpayload
          vpayload = _mm512_mask_expand_epi64(
              vpayload, _knot_mask8(vequal), vmatch_payload);
          _mm512_store_epi64(tmp_values, vpayload);
          for (int x = 0; x < kSIMDWidth; ++x) {
            fn(0, tmp_values[x]);
          }
          int consume_cnt = kSIMDWidth - match_count;
          // compress vmatch_payload
          vmatch_payload = _mm512_maskz_compress_epi64(
              _kand_mask8(mask_map[match_payload_cnt],
                  _knot_mask8(mask_map[consume_cnt])),
              vmatch_payload);
          match_payload_cnt -= consume_cnt;
        } else {
          // compress payload
          vpayload = _mm512_maskz_compress_epi64(vequal, vpayload);
          vmatch_payload = _mm512_mask_expand_epi64(vmatch_payload,
              _knot_mask8(mask_map[match_payload_cnt]), vpayload);
          match_payload_cnt += match_count;
        }
      }

      __m512i voffset =
          _mm512_srai_epi64(vslot_offset_and_value, kPayloadBitCount);
      __mmask8 vend_of_chain =
          _mm512_cmpeq_epi64_mask(voffset, _mm512_set1_epi64(0));
      __mmask8 v_is_end = _kor_mask8(vequal, vend_of_chain);
      __mmask8 v_not_end = _knot_mask8(v_is_end);

      int refill_count = __builtin_popcount(v_is_end);
      if (keys_index + kSIMDWidth > keys_size) {
        alignas(64) uint64_t tmp_key[kSIMDWidth];
        state.vslot_index = _mm512_add_epi64(state.vslot_index, voffset);
        _mm512_mask_compressstoreu_epi64(tmp_key, v_not_end, state.vkey);
        for (int x = 0; x < kSIMDWidth - refill_count; ++x) {
          uint64_t key = tmp_key[x];
          auto it = find(key);
          if (it != end()) {
            fn(key, it->second);
          }
        }
        break;
      }

      if (refill_count == 0) {
        state.vslot_index = _mm512_add_epi64(state.vslot_index, voffset);
      } else if (refill_count == kSIMDWidth) {
        state.vkey = _mm512_load_epi64(&keys[keys_index]);
        state.vslot_index = V_SlotIndex(state.vkey);
        keys_index += kSIMDWidth;
      } else {
        state.vslot_index = _mm512_add_epi64(state.vslot_index, voffset);
        if (refill_count <= remained_count) {
          // remained to state
          expand_state(&state, &remained, v_is_end);
          compress_state(&remained,
              _kand_mask8(mask_map[remained_count],
                  _knot_mask8(mask_map[refill_count])));
          remained_count -= refill_count;
        } else {
          compress_state(&state, v_not_end);
          // state to remained
          expand_state(
              &remained, &state, _knot_mask8(mask_map[remained_count]));
          remained_count += (kSIMDWidth - refill_count);

          state.vkey = _mm512_load_epi64(&keys[keys_index]);
          state.vslot_index = V_SlotIndex(state.vkey);
          keys_index += kSIMDWidth;
        }
      }
    }

    _mm512_store_epi64(tmp_values, vmatch_payload);
    for (int i = 0; i < match_payload_cnt; ++i) {
      fn(0, tmp_values[i]);
    }

    if (remained_count) {
      alignas(64) uint64_t tmp_key[kSIMDWidth];
      _mm512_store_epi64(tmp_key, remained.vkey);
      for (int x = 0; x < remained_count; ++x) {
        uint64_t key = tmp_key[x];
        auto it = find(key);
        if (it != end()) {
          fn(key, it->second);
        }
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

#ifdef ABSL_HAVE_PREFETCH
  template <int kPipelineSize, class Fn>
  void simd_amac_find(const std::vector<K>& keys, const Fn& fn) {
    return simd_amac_find<kPipelineSize, Fn>(&keys[0], keys.size(), fn);
  }

  template <int kPipelineSize, class Fn>
  void simd_amac_find(const K* keys, int64_t keys_size, const Fn& fn) {
    constexpr int kSIMDWidth = 8;
    SIMD_AMAC_State states[kPipelineSize];
    int64_t keys_index = 0;

    // align
    while (intptr_t(&keys[keys_index]) % 64 != 0 && keys_index < keys_size) {
      auto key = keys[keys_index];
      auto it = find(key);
      if (it != end()) {
        fn(key, it->second);
      }
      keys_index += 1;
    }

    if (keys_index + kPipelineSize * kSIMDWidth > keys_size) {
      abort();
    }

    int state_circular_buffer_index = 0;
    int state_circular_buffer_stop_index = -1;

    alignas(64) uint64_t tmp_values[kSIMDWidth];
    alignas(64) uint64_t tmp_key[kSIMDWidth];
    alignas(64) uint64_t tmp_slot_index[kSIMDWidth];

    constexpr __mmask8 remained_mask[] = {0, 1, 3, 7, 15, 31, 63, 127};
    SIMD_AMAC_State remained;
    int remained_count = 0;

    constexpr bool kIMV = true;

    // initial-fill
    for (int i = 0; i < kPipelineSize; ++i) {
      auto& state = states[i];
      state.vkey = _mm512_load_epi64(&keys[keys_index]);
      state.vslot_index = V_SlotIndex(state.vkey);
      keys_index += kSIMDWidth;
      _mm512_store_epi64(tmp_slot_index, state.vslot_index);
      for (int x = 0; x < kSIMDWidth; ++x) {
        absl::PrefetchToLocalCache(slots_ + tmp_slot_index[x]);
      }
    }
    absl::PrefetchToLocalCache(&keys[keys_index]);

    __m512i match_values{0};
    int match_buffer_cnt = 0;

    while (true) {
      auto& state = states[state_circular_buffer_index & (kPipelineSize - 1)];

      __m512i vslot_index_by_byte = _mm512_slli_epi64(state.vslot_index, 4);
      __m512i vslot_key =
          _mm512_i64gather_epi64(vslot_index_by_byte, slots_, 1);
      __m512i vslot_offset_and_value = _mm512_i64gather_epi64(
          vslot_index_by_byte, reinterpret_cast<uint64_t*>(slots_) + 1, 1);

      __mmask8 vequal = _mm512_cmpeq_epi64_mask(vslot_key, state.vkey);
      size_t match_count = __builtin_popcount(vequal);
      if (match_count) {
        __m512i vpayload = _mm512_and_si512(
            vslot_offset_and_value, _mm512_set1_epi64(kPayloadMask));
        if (match_count + match_buffer_cnt >= kSIMDWidth) {
          vpayload = _mm512_mask_expand_epi64(
              vpayload, _knot_mask8(vequal), match_values);
          _mm512_store_epi64(tmp_values, vpayload);
          for (int x = 0; x < kSIMDWidth; ++x) {
            fn(0, tmp_values[x]);
          }
          int consume_cnt = kSIMDWidth - match_count;
          match_values = _mm512_maskz_compress_epi64(
              _kand_mask8(remained_mask[match_buffer_cnt],
                  _knot_mask8(remained_mask[consume_cnt])),
              match_values);
          match_buffer_cnt -= consume_cnt;
        } else {
          // compress payload
          vpayload = _mm512_maskz_compress_epi64(vequal, vpayload);
          match_values = _mm512_mask_expand_epi64(match_values,
              _knot_mask8(remained_mask[match_buffer_cnt]), vpayload);
          match_buffer_cnt += match_count;
        }
      }

      __m512i voffset =
          _mm512_srai_epi64(vslot_offset_and_value, kPayloadBitCount);
      __mmask8 vend_of_chain =
          _mm512_cmpeq_epi64_mask(voffset, _mm512_set1_epi64(0));
      __mmask8 v_is_end = _kor_mask8(vequal, vend_of_chain);
      __mmask8 v_not_end = _knot_mask8(v_is_end);

      int refill_count = __builtin_popcount(v_is_end);
      if (keys_index + kSIMDWidth >= keys_size) {
        state_circular_buffer_stop_index =
            state_circular_buffer_index & (kPipelineSize - 1);
        state.vslot_index = _mm512_add_epi64(state.vslot_index, voffset);
        _mm512_mask_compressstoreu_epi64(tmp_key, v_not_end, state.vkey);
        for (int x = 0; x < kSIMDWidth - refill_count; ++x) {
          uint64_t key = tmp_key[x];
          auto it = find(key);
          if (it != end()) {
            fn(key, it->second);
          }
        }
        break;
      }

      if (refill_count == 0) {
        state.vslot_index = _mm512_add_epi64(state.vslot_index, voffset);
        _mm512_store_epi64(tmp_slot_index, state.vslot_index);
      } else if (refill_count == kSIMDWidth) {
        state.vkey = _mm512_load_epi64(&keys[keys_index]);
        state.vslot_index = V_SlotIndex(state.vkey);
        keys_index += kSIMDWidth;
        absl::PrefetchToLocalCache(&keys[keys_index]);
        _mm512_store_epi64(tmp_slot_index, state.vslot_index);
      } else {
        state.vslot_index = _mm512_add_epi64(state.vslot_index, voffset);
        if constexpr (kIMV) {
          if (refill_count <= remained_count) {
            // remained to state
            expand_state(&state, &remained, v_is_end);
            compress_state(&remained,
                _kand_mask8(remained_mask[remained_count],
                    _knot_mask8(remained_mask[refill_count])));
            remained_count -= refill_count;
          } else {
            compress_state(&state, v_not_end);
            // state to remained
            expand_state(
                &remained, &state, _knot_mask8(remained_mask[remained_count]));
            remained_count += (kSIMDWidth - refill_count);

            state.vkey = _mm512_load_epi64(&keys[keys_index]);
            state.vslot_index = V_SlotIndex(state.vkey);
            keys_index += kSIMDWidth;
            absl::PrefetchToLocalCache(&keys[keys_index]);
          }
          _mm512_store_epi64(tmp_slot_index, state.vslot_index);
        } else {
          _mm512_mask_compressstoreu_epi64(tmp_key, v_not_end, state.vkey);
          _mm512_mask_compressstoreu_epi64(
              tmp_slot_index, v_not_end, state.vslot_index);

          for (int x = kSIMDWidth - refill_count; x < kSIMDWidth; ++x) {
            tmp_key[x] = keys[keys_index++];
            tmp_slot_index[x] = SlotIndex(tmp_key[x]);
          }
          state.vkey = _mm512_load_epi64(tmp_key);
          state.vslot_index = _mm512_load_epi64(tmp_slot_index);
        }
      }
      for (int x = 0; x < kSIMDWidth; ++x) {
        absl::PrefetchToLocalCache(slots_ + tmp_slot_index[x]);
      }

      state_circular_buffer_index += 1;
    }

    // finish all tasks
    for (int i = 0; i < kPipelineSize; ++i) {
      if (i == state_circular_buffer_stop_index) {
        continue;
      }
      auto& state = states[i];
      _mm512_store_epi64(tmp_key, state.vkey);
      for (int x = 0; x < kSIMDWidth; ++x) {
        uint64_t key = tmp_key[x];
        auto it = find(key);
        if (it != end()) {
          fn(key, it->second);
        }
      }
    }

    _mm512_store_epi64(tmp_values, match_values);
    for (int i = 0; i < match_buffer_cnt; ++i) {
      fn(0, tmp_values[i]);
    }

    if (remained_count) {
      _mm512_store_epi64(tmp_key, remained.vkey);
      for (int x = 0; x < remained_count; ++x) {
        uint64_t key = tmp_key[x];
        auto it = find(key);
        if (it != end()) {
          fn(key, it->second);
        }
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
#endif  // ABSL_HAVE_PREFETCH

  inline iterator vfind(K key) const {
    int64_t slot_index = SlotIndex(key);
    auto* slot = &slots_[slot_index];

    if (slot->key() == key) {
      return iterator{slot, this};
    }

    constexpr __mmask8 key_mask = 0b01010101;
    // issue a SIMD compare
    auto* slot_cacheline =
        reinterpret_cast<slot_type*>(intptr_t(slot) & kCachelineMask);
    __m512i this_cacheline = _mm512_load_epi64(slot_cacheline);
    this_cacheline = _mm512_maskz_compress_epi64(key_mask, this_cacheline);
    __m256i keys_in_this_cacheline = _mm512_castsi512_si256(this_cacheline);
    __mmask8 vequal = _mm256_cmpeq_epi64_mask(
        _mm256_set1_epi64x(key), keys_in_this_cacheline);
    if (vequal) {
      return iterator{slot_cacheline + __builtin_ctz(vequal), this};
    }

    if (slot->offset() == 0) {
      return end();
    }

    slot += slot->offset();
    while (slot->key() != key) {
      if (slot->offset() == 0) {
        return end();
      }

      slot += slot->offset();
    }
    return iterator{slot, this};
  }
#endif  // NEIGHBOR_HASH_SIMD_FIND

#ifdef ABSL_HAVE_PREFETCH
#ifdef NEIGHBOR_HASH_COROUTINE_FIND
  struct coro_find_task {
    struct find_promise;
    using handle_type = std::coroutine_handle<find_promise>;

    struct find_promise {
      coro_find_task get_return_object() {
        return {handle_type::from_promise(*this)};
      }

      constexpr std::suspend_never initial_suspend() noexcept { return {}; }

      constexpr std::suspend_always final_suspend() noexcept { return {}; }

      void unhandled_exception() {}

      void return_value(V v) { value = v; }

      V value;
    };

    using promise_type = find_promise;

    V result() { return handle.promise().value; }

    coro_find_task(handle_type h) : handle{h} {}

    coro_find_task(const coro_find_task&) = delete;
    coro_find_task& operator=(const coro_find_task&) = delete;

    coro_find_task(coro_find_task&& other) noexcept : handle{other.handle} {
      other.handle = {};
    }

    coro_find_task& operator=(coro_find_task&& other) noexcept {
      if (this != &other) {
        if (handle)
          handle.destroy();
        handle = other.handle;
        other.handle = {};
      }
      return *this;
    }

    ~coro_find_task() {
      if (handle)
        handle.destroy();
    }

    handle_type handle;
  };

  coro_find_task coro_find(K key) const {
    return coro_find(key, &slots_[SlotIndex(key)]);
  }

  static coro_find_task coro_find(K key, const slot_type* slot) {
    PrefetchToLocalCache(slot);
    co_await std::suspend_always{};

    while (slot->key() != key) {
      if (slot->offset() == 0) {
        co_return 0;
      } else {
        slot += slot->offset();
        PrefetchToLocalCache(slot);
        co_await std::suspend_always{};
      }
    }
    co_return slot->payload();
  }
#endif  // NEIGHBOR_HASH_COROUTINE_FIND
#endif  // ABSL_HAVE_PREFETCH

  inline iterator find(K key) const {
    int64_t slot_index = SlotIndex(key);
    auto* slot = &slots_[slot_index];

    if (slot->key() == key) {
      return iterator{slot, this};
    }

    if (slot->offset() == 0) {
      return end();
    }

    slot += slot->offset();
    if (slot->key() == key) {
      return iterator{slot, this};
    }

    if (slot->offset() == 0) {
      return end();
    }

    slot += slot->offset();
    while (slot->key() != key) {
      if (slot->offset() == 0) {
        return end();
      }

      slot += slot->offset();
    }
    return iterator{slot, this};
  }

  payload_proxy_type& operator[](K key) {
    return emplace(key, V()).first->second;
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
    NeighborHashMap new_hash(n);
    for (size_t i = 0; i < capacity_; ++i) {
      if (is_occupied_slot(i)) {
        new_hash.emplace(slots_[i].key(), slots_[i].payload());
      }
    }

    std::swap(size_, new_hash.size_);
    std::swap(capacity_, new_hash.capacity_);
    std::swap(slots_, new_hash.slots_);
    std::swap(invisible_slot_index_, new_hash.invisible_slot_index_);
  }

  void clear() {
    for (size_t i = 0; i < capacity_; ++i) {
      slots_[i].reset();
    }
    SetUnOccupiedKey();
    size_ = 0;
  }

  void erase(iterator it) {
    auto erased_key = it->first;
    int64_t slot_index = SlotIndex(erased_key);
    auto& slot = slots_[slot_index];

    if (slot.key() == erased_key) {
      if (erased_key == kUnOccupiedKey) {
        // assert slot_index == invisible_slot_index_;
        SetUnOccupiedKey();
      } else {
        EraseSlot(slot_index, slot_index);
      }
      size_ -= 1;
      return;
    }

    int offset = slot.offset();
    while (offset != 0) {
      slot_index += offset;
      auto slot = slots_[slot_index];
      if (slot.key() == erased_key) {
        EraseSlot(slot_index - offset, slot_index);
        size_ -= 1;
        return;
      }
      offset = slot.offset();
    }
  }

  size_t erase(K k) {
    auto it = find(k);
    if (it == end()) {
      return 0;
    }
    erase(it);
    return 1;
  }

  explicit NeighborHashMap(size_t n) {
    auto capacity = PolicyTraits::NormalizeCapacity(n);
    slot_type* slot = (slot_type*)PolicyTraits::template Allocate<
        hardware_constructive_interference_size>(
        sizeof(slot_type) * (capacity + 1));
    for (size_t i = 0; i < capacity; ++i) {
      new (&slot[i]) slot_type;
    }
    new (&slot[capacity]) slot_type;
    slot[capacity].set_sentinel();

    slots_ = slot;
    size_ = 0;
    capacity_ = capacity;
    SetUnOccupiedKey();
  }

  NeighborHashMap() : NeighborHashMap(8) {}

  ~NeighborHashMap() {
    if (slots_) {
      for (size_t i = 0; i < capacity_ + 1; ++i) {
        slots_[i].~slot_type();
      }
      PolicyTraits::Deallocate(slots_, sizeof(slot_type) * (capacity_ + 1));
    }
  }

  // PRECONDITION: it != end()
  void move_to_probing_chain_head(iterator it) {
    int64_t head_slot_index = SlotIndex(it->first);
    if (head_slot_index == invisible_slot_index_) {
      return;
    }

    int64_t current_slot_index = it.slot_ - slots_;
    if (head_slot_index == current_slot_index) {
      return;
    }

    K key = it->first;
    V value = it->second;
    auto* head_slot = &slots_[head_slot_index];
    slots_[current_slot_index].set_key(head_slot->key());
    slots_[current_slot_index].set_payload(head_slot->payload());
    head_slot->set_key(key);
    head_slot->set_payload(value);
  }

  void print_offset() const {
    for (size_t i = 0; i < capacity_; ++i) {
      if (SlotIndex(slots_[i].key()) == i) {
        std::cout << "slot:" << i;
        auto* slot = &slots_[i];
        while (slot->offset() != 0) {
          std::cout << " " << slot->offset();
          slot += slot->offset();
        }
        std::cout << std::endl;
      }
    }
  }

  static constexpr size_t kCachelineMask = ~(64UL - 1);

  bool is_one_cacheline_access(K key) const {
    int slot_index = SlotIndex(key);
    auto* slot = &slots_[slot_index];
    if (slot->key() == key) {
      return true;
    }
    auto cachline_id = ((size_t)slot & kCachelineMask);
    while (slot->key() != key && slot->offset() != 0) {
      slot += slot->offset();
      if (cachline_id != ((size_t)slot & kCachelineMask)) {
        return false;
      }
    }
    return true;
  }

  void count_hit_keys_cacheline_access() const {
    size_t total_cacheline_access = 0;
    size_t reprobed_keys = 0;
    size_t probing_count = 0;

    std::map<size_t, size_t> cacheline_sizes;
    for (size_t i = 0; i < capacity_; ++i) {
      if (SlotIndex(slots_[i].key()) == i) {
        probing_count += 1;
        total_cacheline_access += 1;
        std::set<size_t> cachelines;
        auto* slot = &slots_[i];
        cachelines.insert((size_t)slot & kCachelineMask);
        // onelines->push_back(slot->key());

        int probings = 1;
        while (slot->offset() != 0) {
          probings += 1;
          probing_count += probings;
          reprobed_keys += 1;
          slot += slot->offset();
          cachelines.insert((size_t)slot & kCachelineMask);
          total_cacheline_access += cachelines.size();
          cacheline_sizes[cachelines.size()] += 1;
          // if (cachelines.size() == 1) {
          // onelines->push_back(slot->key());
          // }
        }
      }
    }

    std::cout << "cachelines" << std::endl;
    for (auto& it : cacheline_sizes) {
      std::cout << it.first << ":" << it.second << std::endl;
    }
    std::cout << "size:" << size_
              << " total_cacheline_access:" << total_cacheline_access
              << " average:" << double(total_cacheline_access) / size_
              << " reprobed:" << reprobed_keys << " probings:" << probing_count
              << std::endl;
  }

 protected:
  iterator make_iterator(slot_type* slot) const { return iterator{slot, this}; }

  bool is_occupied_slot(int64_t slot_index) const {
    // (data_.first != kUnOccupiedKey && this != invisible_slot) or
    // (data_.first == kUnOccupiedKey && this == invisible_slot)
    return (slots_[slot_index].key() == kUnOccupiedKey) ==
        (slot_index == invisible_slot_index_);
  }

  bool is_available_slot(int64_t slot_index) const {
    // not occupied and not invisible
    return (slots_[slot_index].key() == kUnOccupiedKey) &&
        (slot_index != invisible_slot_index_);
  }

  void SetUnOccupiedKey() {
    int64_t slot_index = SlotIndex(kUnOccupiedKey);
    K invalid_key = kUnOccupiedKey;
    while (SlotIndex(invalid_key) == slot_index) {
      invalid_key += 1;
    }

    // This slot is intentionally set to an invalid state
    // where it's designed not to match any find operations.
    // We purposefully use 'kUnOccupiedKey' as the initial key's value
    // to avoid the need to check if the slot is occupied during the 'find' operation.
    slots_[slot_index].set_key(invalid_key);
    slots_[slot_index].set_payload(0);
    // NOTE: we should keep its `offset`.

    invisible_slot_index_ = slot_index;
  }

  template <bool update_if_exist>
  std::pair<iterator, bool> emplace_impl(K key, V value) {
    int64_t slot_index = SlotIndex(key);
    auto& slot = slots_[slot_index];

    if (is_available_slot(slot_index) ||
        (key == kUnOccupiedKey && slot.key() != key)) {
      slot.set_key(key);
      slot.set_payload(value);
      size_ += 1;
      return {iterator{&slot, this}, true};
    }

    if (slot.key() == key) {
      if constexpr (update_if_exist) {
        slot.set_payload(value);
      }
      return {iterator{&slot, this}, false};
    }

    int64_t head_slot_index = slot_index;
    int offset = slot.offset();
    while (offset != 0) {
      slot_index += offset;
      auto& slot = slots_[slot_index];
      if (slot.key() == key) {
        if constexpr (update_if_exist) {
          slot.set_payload(value);
        }
        return {iterator{&slot, this}, false};
      }
      offset = slot.offset();
    }

    size_ += 1;
    return do_insert_with_collision(head_slot_index, slot_index, key, value);
  }

  std::pair<iterator, bool> do_insert_with_collision(
      int64_t head_slot_index, int64_t tail_slot_index, K key, V value) {
    do {
      if (PolicyTraits::ShouldGrow(size_, capacity_)) {
        break;
      }

      int64_t occupant_head_slot = SlotIndex(slots_[head_slot_index].key());
      int64_t inserted_slot_index = head_slot_index;
      if (occupant_head_slot != head_slot_index &&
          head_slot_index != invisible_slot_index_) {
        // not the chain, kick out head, insert into head
        if (!KickOutOccupant(head_slot_index, occupant_head_slot)) {
          break;
        }
      } else {
        // insert after tail
        int64_t new_offset = FindAvailableSlot(tail_slot_index, head_slot_index,
            0);  // key < slots_[tail_slot_index].key() ? -1 : 1
        if (new_offset == kInvalidOffset) {
          break;
        }
        slots_[tail_slot_index].set_offset(new_offset);
        inserted_slot_index = tail_slot_index + new_offset;
      }

      auto& slot = slots_[inserted_slot_index];
      slot.set_key(key);
      slot.set_payload(value);
      slot.set_offset(0);
      return {iterator{&slot, this}, true};
    } while (0);

    // rehash
    rehash(PolicyTraits::GrowthSize(capacity_));
    return emplace_impl<true>(key, value);
  }

  void EraseSlot(int64_t prev_slot_index, int64_t slot_index) {
    auto& slot = slots_[slot_index];
    int offset = slot.offset();
    if (offset == 0) {
      slots_[prev_slot_index].set_offset(0);
      slot.reset();
    } else {
      auto* current_slot = &slot;
      do {
        // move slot forward
        slot_index += offset;
        auto& next_slot = slots_[slot_index];
        current_slot->set_key(next_slot.key());
        current_slot->set_payload(next_slot.payload());

        offset = next_slot.offset();
        if (offset == 0) {
          current_slot->set_offset(0);
          next_slot.reset();
          return;
        }
        current_slot = &next_slot;
      } while (true);
    }
  }

  bool KickOutOccupant(int64_t occupied_slot, int64_t occupant_head_slot) {
    int64_t slot_index = occupant_head_slot;
    do {
      int offset = slots_[slot_index].offset();
      // find occupant from it's home so we can get its previous slot
      if (slot_index + offset == occupied_slot) {
        int new_offset =
            FindAvailableSlot(slot_index, occupant_head_slot, 0);  //offset);
        if (new_offset == kInvalidOffset) {
          return false;
        }
        int64_t new_slot_index = slot_index + new_offset;

        int next_offset = slots_[occupied_slot].offset();
        if (next_offset != 0) {
          int new_next_offset = occupied_slot + next_offset - new_slot_index;
          if (new_next_offset <= kInvalidOffset ||
              new_next_offset >= kOffsetUpperBound) {
            return false;
          }
          slots_[new_slot_index].set_offset(new_next_offset);
        } else {
          slots_[new_slot_index].set_offset(0);
        }

        slots_[slot_index].set_offset(new_offset);
        slots_[new_slot_index].set_key(slots_[occupied_slot].key());
        slots_[new_slot_index].set_payload(slots_[occupied_slot].payload());
        return true;
      }
      slot_index += offset;
    } while (true);
  }

  template <int range_begin_bits, int range_end_bits, int... ranges>
  int FindAvailableNeighborRange(int64_t slot_begin, int64_t slot_end,
      int64_t slot_index, int64_t head_slot_index, int direction,
      std::integer_sequence<int, range_begin_bits, range_end_bits, ranges...>) {
    constexpr int64_t range_begin = 1 << range_begin_bits;
    constexpr int64_t range_end = 1 << range_end_bits;

    if (direction >= 0) {
      // right
      for (int64_t i = std::min(slot_index + range_begin, slot_end);
           i < std::min(slot_index + range_end, slot_end); ++i) {
        if (is_available_slot(i)) {
          return i - slot_index;
        }
      }
    }

    if (direction <= 0) {
      // left
      for (int64_t i = std::max(slot_begin, slot_index - range_begin);
           i > std::max(slot_begin, slot_index - range_end); --i) {
        if (is_available_slot(i)) {
          return i - slot_index;
        }
      }
    }

    return FindAvailableNeighborRange<range_end_bits, ranges...>(slot_begin,
        slot_end, slot_index, head_slot_index, direction,
        std::integer_sequence<int, range_end_bits, ranges...>());
  }

  template <int range_end>
  int FindAvailableNeighborRange(int64_t, int64_t, int64_t, int64_t, int,
      std::integer_sequence<int, range_end>) {
    return kInvalidOffset;
  }

  template <int kCachelineSize, bool kSubRange = false>
  bool kick_if_slot_not_within_cacheline(int64_t slot_index) {
    if (invisible_slot_index_ == slot_index) {
      return false;
    }
    auto* slot = &slots_[slot_index];
    if (slot->offset() != 0) {
      return false;
    }

    int64_t head_index = SlotIndex(slot->key());
    if (head_index == slot_index) {
      return false;
    }
    auto prev_index = head_index;
    while (prev_index + slots_[prev_index].offset() != slot_index) {
      prev_index += slots_[prev_index].offset();
    }

    constexpr size_t items_per_cacheline = kCachelineSize / sizeof(slot_type);
    if ((prev_index / items_per_cacheline) ==
        (slot_index / items_per_cacheline)) {
      return false;
    }

    int offset = FindAvailableSlot<false, kSubRange>(prev_index, head_index, 0);
    if (offset == kInvalidOffset) {
      return false;
    }
    // std::cout << "cacheline kick, before:" << head_index;
    // auto x = head_index;
    // while (slots_[x].offset() != 0) {
    //   std::cout << " index:" << x << " offset:" << slots_[x].offset();
    //   x += slots_[x].offset();
    // }
    // std::cout << " index:" << x << std::endl;
    int64_t new_slot_index = prev_index + offset;
    slots_[new_slot_index].set_offset(0);

    slots_[new_slot_index].set_key(slot->key());
    slots_[new_slot_index].set_payload(slot->payload());
    slots_[prev_index].set_offset(offset);
    slot->reset();

    // std::cout << "cacheline kick:" << slot_index << " head:" << head_index << std::endl;
    // x = head_index;
    // while (slots_[x].offset() != 0) {
    //   std::cout << " index:" << x << " offset:" << slots_[x].offset();
    //   x += slots_[x].offset();
    // }
    // std::cout << " index:" << x << std::endl;
    return true;
  }

  template <int kCachelineSize, bool kSubRange>
  int FindWithinCacheline(int64_t slot_index) {
    constexpr size_t items_per_cacheline = kCachelineSize / sizeof(slot_type);
    int64_t cacheline_start =
        int64_t(slot_index / items_per_cacheline) * items_per_cacheline;
    int64_t cacheline_end = cacheline_start + items_per_cacheline;
    for (int64_t i = cacheline_start; i < cacheline_end; ++i) {
      if (i != slot_index &&
          (is_available_slot(i) ||
              kick_if_slot_not_within_cacheline<kCachelineSize, kSubRange>(
                  i))) {
        return i - slot_index;
      }
    }
    // next cacheline
    // if (cacheline_end + items_per_cacheline <= capacity_) {
    //   for (int64_t i = 0; i < items_per_cacheline; ++i) {
    //     if (is_available_slot(i + cacheline_end)) {
    //       return i + cacheline_end - slot_index;
    //     }
    //   }
    // }
    return kInvalidOffset;
  }

  template <bool kPreferCacheline = true, bool kSubRange = false>
  int FindAvailableSlot(
      int64_t slot_index, int64_t head_slot_index, int direction) {
    // in this cacheline
    if (kPreferCacheline) {
      int offset = FindWithinCacheline<hardware_constructive_interference_size,
          kSubRange>(slot_index);
      if (offset != kInvalidOffset) {
        return offset;
      }
    }

    int64_t slot_begin = 0;
    int64_t slot_end = capacity_;
    if constexpr (kSubRange) {
      std::tie(slot_begin, slot_end) =
          PolicyTraits::SubRange(head_slot_index, capacity_);
    }

    return FindAvailableNeighborRange(slot_begin, slot_end, slot_index,
        head_slot_index, direction,
        std::make_integer_sequence<int, kOffsetBitCount>());
  }

  typename PolicyTraits::Hash hash_;
  typename PolicyTraits::Hash2Slot hash2slot_;

  slot_type* slots_;
  size_t size_;
  size_t capacity_;
  int64_t invisible_slot_index_;
};

template <class MapType>
class AtomicSlot {
 public:
  using key_type = typename MapType::key_type;
  using mapped_type = typename MapType::mapped_type;

  class payload_proxy_type {
    friend MapType;

   public:
    void operator=(mapped_type payload) {
      value_ =
          (payload & MapType::kPayloadMask) | (value_ & MapType::kOffsetMask);
    }

    operator mapped_type() const { return value_ & MapType::kPayloadMask; }

    payload_proxy_type(const payload_proxy_type&) = delete;
    payload_proxy_type& operator=(const payload_proxy_type&) = delete;

    explicit payload_proxy_type(mapped_type value) : value_(value) {}

    payload_proxy_type() {}

   private:
    mapped_type& raw() { return value_; }

    mapped_type value_{};
  };

  AtomicSlot() { reset(); }

  void set_key(
      key_type key, std::memory_order order = std::memory_order_relaxed) {
    data_.first.store(key, order);
  }

  void set_payload(mapped_type payload,
      std::memory_order order = std::memory_order_relaxed) {
    mapped_type new_value = (payload & MapType::kPayloadMask) |
        (data_.second.load(order) & MapType::kOffsetMask);
    data_.second.store(new_value, order);
  }

  void set_offset(
      int offset, std::memory_order order = std::memory_order_relaxed) {
    mapped_type new_value =
        static_cast<mapped_type>(
            static_cast<typename std::make_signed<mapped_type>::type>(offset)
            << MapType::kPayloadBitCount) |
        (data_.second.load(order) & MapType::kPayloadMask);
    data_.second.store(new_value, order);
  }

  key_type key(std::memory_order order = std::memory_order_relaxed) const {
    return data_.first.load(order);
  }

  int offset(std::memory_order order = std::memory_order_relaxed) const {
    return static_cast<typename std::make_signed<mapped_type>::type>(
               data_.second.load(order)) >>
        MapType::kPayloadBitCount;
  }

  mapped_type payload(
      std::memory_order order = std::memory_order_relaxed) const {
    return data_.second.load(order) & MapType::kPayloadMask;
  }

  void reset(std::memory_order order = std::memory_order_relaxed) {
    data_.first.store(MapType::kUnOccupiedKey, order);
    data_.second.store(0, order);
  }

  void set_sentinel() { set_offset(MapType::kInvalidOffset); }

  using value_type = std::pair<const key_type, payload_proxy_type>;

  value_type& value() { return value_; }

  const value_type& value() const { return value_; }

  AtomicSlot& operator=(const AtomicSlot& other) = delete;
  AtomicSlot(const AtomicSlot& other) = delete;

 private:
  using atomic_value_type =
      std::pair<std::atomic<key_type>, std::atomic<mapped_type>>;

  static_assert(
      sizeof(atomic_value_type) == sizeof(value_type), "invalid atomic size");
  static_assert(
      std::alignment_of<atomic_value_type>() == std::alignment_of<value_type>(),
      "invalid atomic alignment");

  union {
    atomic_value_type data_;
    value_type value_;
  };
};

template <class K, class V, class PolicyTraits>
class alignas(64) AtomicNeighborHashMap
    : public NeighborHashMap<K, V, PolicyTraits, AtomicSlot> {
 public:
  using Base = NeighborHashMap<K, V, PolicyTraits, AtomicSlot>;
  using iterator = typename Base::iterator;
  using slot_type = typename Base::slot_type;

  using Base::begin;
  using Base::end;

  inline iterator find(K key) const {
    int64_t slot_index = Base::SlotIndex(key);
    auto* slot = &Base::slots_[slot_index];

    if (slot->key(std::memory_order_acquire) == key) {
      return Base::make_iterator(slot);
    }

    int offset = slot->offset(std::memory_order_consume);
    if (offset == 0) {
      return end();
    }

    slot += offset;
    while (slot->key(std::memory_order_acquire) != key) {
      offset = slot->offset(std::memory_order_consume);
      if (offset == 0) {
        return end();
      }

      slot += offset;
    }
    return Base::make_iterator(slot);
  }

  enum class Status : int {
    kInserted = 0,
    kUpdated = 1,
    kHeadInsertPrepared = 2,
    kTailInsertPrepared = 3,
    kPrepareFailed = 4
  };

  template <class Fn>
  void atomic_foreach(const Fn& fn) {
    for (int64_t slot_index = 0; slot_index < int64_t(Base::capacity_);
         ++slot_index) {
      auto* slot = &Base::slots_[slot_index];
      if (Base::SlotIndex(slot->key(std::memory_order_acquire)) == slot_index ||
          slot_index == Base::invisible_slot_index_) {
        // head node
        if (slot_index != Base::invisible_slot_index_ ||
            slot->key() == Base::kUnOccupiedKey) {
          fn(slot->key(), slot->payload());
        }

        while (slot->offset(std::memory_order_consume) != 0) {
          slot += slot->offset();
          fn(slot->key(), slot->payload());
        }
      }
    }
  }

  template <bool kAllowTailInsert = false,
      bool kInsertAfterKickEpoch = kAllowTailInsert>
  Status atomic_insert_or_update(K key, V value) {
    int64_t slot_index = Base::SlotIndex(key);
    auto* slot = Base::slots_ + slot_index;

    if (kInsertAfterKickEpoch) {
      int64_t occupant_head_slot = Base::SlotIndex(slot->key());
      if (occupant_head_slot != slot_index &&
          slot_index != Base::invisible_slot_index_) {
        slot->reset();
      }
    }

    if (Base::is_available_slot(slot_index) ||
        (key == Base::kUnOccupiedKey && slot->key() != key)) {
      // insert key/value directly into head
      slot->set_payload(value);
      slot->set_offset(0);
      slot->set_key(key, std::memory_order_release);
      Base::size_ += 1;
      return Status::kInserted;
    }

    if (slot->key() == key) {
      slot->set_payload(value);
      return Status::kUpdated;
    }

    int64_t head_slot_index = slot_index;
    int offset = slot->offset();
    while (offset != 0) {
      slot_index += offset;
      auto* slot = &Base::slots_[slot_index];
      if (slot->key() == key) {
        slot->set_payload(value);
        return Status::kUpdated;
      }
      offset = slot->offset();
    }

    int64_t occupant_head_slot = Base::SlotIndex(slot->key());
    if (occupant_head_slot != head_slot_index &&
        head_slot_index != Base::invisible_slot_index_) {
      // just kick it out, don't insert
      if (!KickOutOccupantAtomic(head_slot_index, occupant_head_slot)) {
        return Status::kPrepareFailed;
      }
      return Status::kHeadInsertPrepared;
    } else if constexpr (kAllowTailInsert) {
      // insert into tail
      int new_offset = Base::template FindAvailableSlot<false>(
          slot_index, head_slot_index, 0);
      if (new_offset == Base::kInvalidOffset) {
        return Status::kPrepareFailed;
      }
      auto inserted_slot_index = slot_index + new_offset;
      Base::slots_[inserted_slot_index].set_payload(value);
      Base::slots_[inserted_slot_index].set_offset(0);
      Base::slots_[inserted_slot_index].set_key(key);
      Base::slots_[slot_index].set_offset(
          new_offset, std::memory_order_release);
      Base::size_ += 1;
      return Status::kInserted;
    } else {
      return Status::kTailInsertPrepared;
    }
  }

 private:
  bool KickOutOccupantAtomic(
      int64_t occupied_slot, int64_t occupant_head_slot) {
    int64_t slot_index = occupant_head_slot;
    do {
      int offset = Base::slots_[slot_index].offset();
      if (offset == 0) {
        // this may happen when this slot is already kick out, but not occupied
        return true;
      }
      // find occupant from it's home so we can get its previous slot
      if (slot_index + offset == occupied_slot) {
        int new_offset = Base::template FindAvailableSlot<false>(
            slot_index, occupant_head_slot, 0);
        if (new_offset == Base::kInvalidOffset) {
          return false;
        }
        int64_t new_slot_index = slot_index + new_offset;

        int next_offset = Base::slots_[occupied_slot].offset();
        if (next_offset != 0) {
          int new_next_offset = occupied_slot + next_offset - new_slot_index;
          if (new_next_offset <= Base::kInvalidOffset ||
              new_next_offset >= Base::kOffsetUpperBound) {
            return false;
          }
          Base::slots_[new_slot_index].set_offset(new_next_offset);
        } else {
          Base::slots_[new_slot_index].set_offset(0);
        }

        Base::slots_[new_slot_index].set_payload(
            Base::slots_[occupied_slot].payload());
        Base::slots_[new_slot_index].set_key(Base::slots_[occupied_slot].key());

        Base::slots_[slot_index].set_offset(
            new_offset, std::memory_order_release);
        return true;
      }
      slot_index += offset;
    } while (true);
  }
};

}  // namespace neighbor
