# C++ test target with Google Test
cc_test(
    name = "neighbor_hash_test",
    srcs = ["neighbor_hash_test.cc"],
    deps = [
        "//:neighbor_hash",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest",  # Google Test dependency
    ],
)

# Benchmark target using Google Benchmark
cc_test(
    name = "hash_map_benchmark",
    srcs = ["hash_map_benchmark.cc"],
    deps = [
        "//:neighbor_hash",
        "@ankerl_unordered_dense//:unordered_dense",
        "@bytell",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_benchmark//:benchmark",
        "@emhash",
        "@tsl",
    ],
)

cc_test(
    name = "ankerl_bench",
    srcs = ["ankerl_bench.cc"],
    copts = [
        "-Iexternal/doctest/doctest",
        "-DDOCTEST_CONFIG_IMPLEMENT_WITH_MAIN",
    ],
    deps = [
        "//:neighbor_hash",
        "@ankerl_unordered_dense//:bench",
        "@ankerl_unordered_dense//:unordered_dense",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_binary(
    name = "mixed_multi_threading",
    srcs = ["mixed_multi_threading.cc"],
    deps = [
        "//:neighbor_hash",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings:str_format",
        "@folly",
    ],
)
