# neighbor\_hash\_map

[![build](https://github.com/slow-steppers/NeighborHash/actions/workflows/c-cpp.yml/badge.svg)](https://github.com/slow-steppers/NeighborHash/actions?query=workflow%3Abuild)
[![release](https://img.shields.io/github/v/release/slow-steppers/NeighborHash)](https://github.com/slow-steppers/NeighborHash/releases/latest)
[![issues](https://img.shields.io/github/issues/slow-steppers/NeighborHash?color=yellow)](https://github.com/slow-steppers/NeighborHash/issues?q=is%3Aopen+is%3Aissue)
[![license](https://img.shields.io/github/license/slow-steppers/NeighborHash)](https://github.com/slow-steppers/NeighborHash/blob/master/LICENSE)

----

This repo contains codes and steps necessary to reproduce the artifacts for research paper titled [**CIKM'24: An Enhanced Batch Query Architecture in Real-time Recommendation**](https://dl.acm.org/doi/10.1145/3627673.3680034).

## Setting up the hardware

We conducted tests in the paper using `Intel(R) Xeon(R) Gold 5218` CPUs. In fact, any hardware that supports AVX512 can be used for reproduction, but the results may not necessarily match the results presented in the paper.

## Software Requirements

* Linux x86\_64 kernel >= 4.9.0
* Hugepage (2 MiB) supported and has been configured, for example: `echo 8192 > /proc/sys/vm/nr_hugepages`
* Clang >= 16.0.6 or GCC >= 11.4.0
* Bazel >= 6.3.0
* Tested on Ubuntu 22.04

## Build benchmark

```bash
git clone https://github.com/slow-steppers/NeighborHash
cd NeighborHash
```

When using clang
```bash
bazel build --config=clang --config=neighbor_hugepage --config=neighbor_simd //testing:all
```

**Or**, when using gcc
```bash
bazel build --config=neighbor_hugepage --config=neighbor_simd //testing:all
```

## Run benchmarks

Our benchmark implementation utilizes the `google benchmark` framework, which allows selecting benchmarks to run using regular expressions.
The regular expression for running random access is: `BM_RandomAccess<.*HashMapType.*>/DS/SQR/LF`.
For instance, to run benchmarks for all scalar hashmaps with a 1M DatasetSize, SQR=90, and LF=80, execute the following code:

```bash
numactl --membind=0 --cpunodebind=0 ./bazel-bin/testing/hash_map_benchmark --benchmark_filter="BM_RandomAccess<.*Scalar.*uniform>/1048576/90/79"
```


We have integrated the following components into our benchmark:
- `NeighborHash`
- `std::unordered_map`
- `LinearProbing`
- `BucketingSIMD_16x16`
- `ankerl::unordered_dense::map`
- `absl::flat_hash_map`
- `ska::bytell_hash_map`
- `emhash7::HashMap`
- `tsl::hopscotch_map`
- Native array (Random Access)


For ease of comparison, we have categorized each hashmap into the following classes: Scalar, IntraVec, Vec, MultiThreading.
The candidate values for each component of the expression are as follows:
- SQR: 30/50/90/100
- LF: 79
- DS: 1024/16384/131072/1048576/2097152/16777216/67108864/134217728
- HashMapType: Scalar/Vec/IntraVec/MultiThreading/QFGO


The following presents the code for reproducing the results from the paper:

```bash
# all scalar hashmaps under all dataset size with SQR=90,LF=80
numactl --membind=0 --cpunodebind=0 ./bazel-bin/testing/hash_map_benchmark --benchmark_filter="BM_RandomAccess<.*Scalar.*>/.*/90/79"

# Neighbor/LinearProbing/RandomAccess with AMAC under all dataset size with SQR=90,LF=80
numactl --membind=0 --cpunodebind=0 ./bazel-bin/testing/hash_map_benchmark --benchmark_filter="BM_RandomAccess<(Neighbor|LinearProbing|Array).*uniform.*AMAC.*>/.*/90/79$"

# all intra-vectorization hashmaps under all dataset size with SQR=90,LF=80
numactl --membind=0 --cpunodebind=0 ./bazel-bin/testing/hash_map_benchmark --benchmark_filter="BM_RandomAccess<.*IntraVec.*>/.*/90/79"

# all vectorization hashmaps under all dataset size with SQR=90,LF=80
numactl --membind=0 --cpunodebind=0 ./bazel-bin/testing/hash_map_benchmark --benchmark_filter="BM_RandomAccess<.*Vec.*>/.*/90/79"

# NeighborHash with QFGO under all dataset size with SQR=90,LF=80
numactl --membind=0 --cpunodebind=0 ./bazel-bin/testing/hash_map_benchmark --benchmark_filter="BM_RandomAccess<.*QFGO.*>/.*/90/79"
```

To perform `Mixed insertions and lookups`, use the following code:
```bash
numactl --membind=0 --cpunodebind=0 ./bazel-bin/testing/mixed_multi_threading
```
