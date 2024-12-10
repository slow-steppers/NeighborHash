cc_library(
    name = "unordered_dense",
    hdrs = ["include/ankerl/unordered_dense.h"],
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "bench",
    srcs = [
        "test/app/counter.cpp",
        "test/app/doctest.cpp",
        "test/app/nanobench.cpp",
    ],
    hdrs = [
        "test/app/counter.h",
        "test/app/doctest.h",
        "test/app/name_of_type.h",
        "test/app/print.h",
        "test/third-party/nanobench.h",
    ],
    copts = ["-Iexternal/doctest/doctest"],
    strip_include_prefix = "test",
    visibility = ["//visibility:public"],
    deps = [
        ":unordered_dense",
        "@doctest//doctest",
        "@fmtlib",
    ],
)
