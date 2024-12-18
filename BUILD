cc_library(
    name = "neighbor_hash",
    hdrs = [
        "neighbor_hash/bucketing_simd.h",
        "neighbor_hash/common_policy.h",
        "neighbor_hash/linear_probing.h",
        "neighbor_hash/neighbor_hash.h",
        "neighbor_hash/slot_type.h",
    ],
    linkstatic = True,
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:prefetch",
    ],
)
