#pragma once

#include <type_traits>
#include <utility>

namespace neighbor {
namespace detail {

template <class MapType>
class PayloadProxy {
  friend MapType;

 public:
  using mapped_type = typename MapType::mapped_type;

  void operator=(mapped_type payload) {
    value_ =
        (payload & MapType::kPayloadMask) | (value_ & MapType::kOffsetMask);
  }

  operator mapped_type() const { return value_ & MapType::kPayloadMask; }

  PayloadProxy(const PayloadProxy&) = delete;
  PayloadProxy& operator=(const PayloadProxy&) = delete;

  explicit PayloadProxy(mapped_type value) : value_(value) {}

  PayloadProxy() {}

  mapped_type& raw() { return value_; }

 private:
  mapped_type value_{};
};

template <class MapType>
class Slot {
 public:
  using key_type = typename MapType::key_type;
  using mapped_type = typename MapType::mapped_type;

  using payload_proxy_type = PayloadProxy<MapType>;

  Slot() { reset(); }

  void set_key(key_type key) { data_.first = key; }

  void set_payload(mapped_type payload) {
    data_.second = (payload & MapType::kPayloadMask) |
        (data_.second & MapType::kOffsetMask);
  }

  void set_offset(int offset) {
    data_.second =
        static_cast<mapped_type>(
            static_cast<typename std::make_signed<mapped_type>::type>(offset)
            << MapType::kPayloadBitCount) |
        (data_.second & MapType::kPayloadMask);
  }

  key_type key() const { return data_.first; }

  int offset() const {
    return static_cast<typename std::make_signed<mapped_type>::type>(
               data_.second) >>
        MapType::kPayloadBitCount;
  }

  mapped_type payload() const { return data_.second & MapType::kPayloadMask; }

  void reset() {
    data_.first = MapType::kUnOccupiedKey;
    data_.second = MapType::kUnOccupiedValue;
  }

  void set_sentinel() { set_offset(MapType::kInvalidOffset); }

  using value_type = std::pair<const key_type, payload_proxy_type>;

  value_type& value() { return value_; }

  const value_type& value() const { return value_; }

  Slot& operator=(const Slot& other) {
    data_ = other.data_;
    return *this;
  }

  Slot(const Slot& other) { data_ = other.data_; }

 private:
  union {
    std::pair<key_type, mapped_type> data_;
    value_type value_;
  };
};

}  // namespace detail
}  // namespace neighbor
