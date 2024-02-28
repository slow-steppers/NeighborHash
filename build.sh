#!/bin/bash

set -eu

ls -l /usr/bin/g++*
ls -l /usr/bin/clang++*

dpkg -l | fgrep libstdc++




exit

ABSOLUTE_ME="$(realpath "${0}")"
PROJECT_DIR="$(dirname "${ABSOLUTE_ME}")"

cd -P "${PROJECT_DIR}"

NEED_DOWNLOAD_BAZEL="1"
if [ "$(type -t "bazel")" == "file" ]; then
  SYSTEM_BAZEL_VERSION="$(bazel --version | head -n "1" | sed -n 's/^.*\s\+\(\([0-9]\+\.\)\+[0-9]\+[a-z]\?\)\(\s\+.*\)\?\s*$/\1/p')"
  if [ "$(echo -ne "${SYSTEM_BAZEL_VERSION}\n6.3.0\n" | sort -V | head -n "1")" == "6.3.0" ]; then
    NEED_DOWNLOAD_BAZEL="0"
  fi
fi

BAZEL_BINARY="bazel"
if [ "${NEED_DOWNLOAD_BAZEL}" -ne "0" ]; then
  mkdir -p "bin"
  curl -s -f -L -m "60" "https://github.com/bazelbuild/bazel/releases/download/6.3.2/bazel-6.3.2-linux-x86_64" > "bin/bazel"
  chmod +x "bin/bazel"
  BAZEL_BINARY="bin/bazel"
fi

ARGUMENTS=( --config="neighbor_hugepage" )

if lscpu | grep -iq "avx512"; then
  ARGUMENTS+=( --config="neighbor_simd" )
fi

if [ "$(type -t "clang")" == "file" ]; then
  ARGUMENTS+=( --config="clang" )
fi

exec "${BAZEL_BINARY}" build "${ARGUMENTS[@]}" "//testing:all" "${@}"
