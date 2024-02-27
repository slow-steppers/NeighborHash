cc_library(
    name = "unordered_dense",
    strip_include_prefix = "include",
    hdrs = ["include/ankerl/unordered_dense.h"],
    visibility = ["//visibility:public"]
)

cc_library(
    name = "bench",
    strip_include_prefix = "test",
    copts = ["-Iexternal/doctest/doctest"],
    hdrs = [
        "test/third-party/nanobench.h",
        "test/app/doctest.h",
        "test/app/counter.h",
        "test/app/print.h",
        "test/app/name_of_type.h"
    ],
    srcs = ["test/app/nanobench.cpp", "test/app/doctest.cpp", "test/app/counter.cpp"],
    deps = [":unordered_dense", "@fmtlib", "@doctest//doctest:doctest"],
    visibility = ["//visibility:public"]
)
