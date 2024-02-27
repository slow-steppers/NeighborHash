load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "bazel_skylib",
    sha256 = "08c0386f45821ce246bbbf77503c973246ed6ee5c3463e41efc197fa9bc3a7f4",
    strip_prefix = "bazel-skylib-288731ef9f7f688932bd50e704a91a45ec185f9b",
    urls = ["https://github.com/bazelbuild/bazel-skylib/archive/288731ef9f7f688932bd50e704a91a45ec185f9b.zip"],
)

http_archive(
    name = "com_github_storypku_rules_folly",
    patches = ["//third_party:rules_folly.patch"],
    sha256 = "16441df2d454a6d7ef4da38d4e5fada9913d1f9a3b2015b9fe792081082d2a65",
    strip_prefix = "rules_folly-0.2.0",
    urls = ["https://github.com/storypku/rules_folly/archive/v0.2.0.tar.gz"],
)

load("@com_github_storypku_rules_folly//bazel:folly_deps.bzl", "folly_deps")

folly_deps()

http_archive(
    name = "com_github_nelhage_rules_boost",
    patches = ["//third_party:rules_boost.patch"],
    sha256 = "046f774b185436d506efeef8be6979f2c22f1971bfebd0979bafa28088bf28d0",
    strip_prefix = "rules_boost-fb9f3c9a6011f966200027843d894923ebc9cd0b",
    urls = ["https://github.com/nelhage/rules_boost/archive/fb9f3c9a6011f966200027843d894923ebc9cd0b.tar.gz"],
)

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")

boost_deps()

http_archive(
    name = "ankerl_unordered_dense",
    build_file = "//third_party:ankerl.BUILD",
    sha256 = "300410dbcd32800f83b2113dfecbdfe8cd256caa4cfeb117d646021d6e3209ae",
    strip_prefix = "unordered_dense-4.1.2",
    urls = ["https://github.com/martinus/unordered_dense/archive/refs/tags/v4.1.2.tar.gz"],
)

http_archive(
    name = "bytell",
    build_file = "//third_party:bytell.BUILD",
    sha256 = "606aef45bb69d2fd8247b872919dc01607fe4477df28d6e0b4e686b1b6651980",
    strip_prefix = "flat_hash_map-2c4687431f978f02a3780e24b8b701d22aa32d9c",
    urls = ["https://github.com/skarupke/flat_hash_map/archive/2c4687431f978f02a3780e24b8b701d22aa32d9c.tar.gz"],
)

http_archive(
    name = "com_google_absl",
    sha256 = "987ce98f02eefbaf930d6e38ab16aa05737234d7afbab2d5c4ea7adbe50c28ed",
    strip_prefix = "abseil-cpp-20230802.1",
    urls = ["https://github.com/abseil/abseil-cpp/archive/20230802.1.tar.gz"],
)

http_archive(
    name = "com_google_benchmark",
    sha256 = "e4fbb85eec69e6668ad397ec71a3a3ab165903abe98a8327db920b94508f720e",
    strip_prefix = "benchmark-1.5.3",
    urls = ["https://github.com/google/benchmark/archive/v1.5.3.tar.gz"],
)

http_archive(
    name = "com_google_googletest",
    sha256 = "b4870bf121ff7795ba20d20bcdd8627b8e088f2d1dab299a031c1034eddc93d5",
    strip_prefix = "googletest-release-1.11.0",
    urls = ["https://github.com/google/googletest/archive/release-1.11.0.tar.gz"],
)

http_archive(
    name = "doctest",
    sha256 = "632ed2c05a7f53fa961381497bf8069093f0d6628c5f26286161fbd32a560186",
    strip_prefix = "doctest-2.4.11",
    urls = ["https://github.com/doctest/doctest/archive/refs/tags/v2.4.11.tar.gz"],
)

http_archive(
    name = "emhash",
    build_file = "//third_party:emhash.BUILD",
    sha256 = "f5f1fe855a63b2d6e964ac25baef48189668f4ce7ad2f35bbc4dac750f08ee84",
    strip_prefix = "emhash-e559ba34c3c5941f95d74f30868b1f7d4fac65ef",
    urls = ["https://github.com/ktprime/emhash/archive/e559ba34c3c5941f95d74f30868b1f7d4fac65ef.tar.gz"],
)

http_archive(
    name = "fmtlib",
    build_file = "//third_party:fmtlib.BUILD",
    sha256 = "78b8c0a72b1c35e4443a7e308df52498252d1cefc2b08c9a97bc9ee6cfe61f8b",
    strip_prefix = "fmt-10.1.1",
    urls = ["https://github.com/fmtlib/fmt/archive/refs/tags/10.1.1.tar.gz"],
)

http_archive(
    name = "tsl",
    build_file = "//third_party:tsl.BUILD",
    sha256 = "0a77f4835379e74bb7a1c043f3b3c498272acca1c70b03dd5a0444fddf28b316",
    strip_prefix = "hopscotch-map-2.3.1",
    urls = ["https://github.com/Tessil/hopscotch-map/archive/refs/tags/v2.3.1.zip"],
)
