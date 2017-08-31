#! /usr/bin/env bash

ARCH=$(uname -s)
if [ "$ARCH" == "Linux" ]; then
    NPROC="$(nproc)"
elif [ "$ARCH" == "Darwin" ]; then
    NPROC="$(sysctl -n hw.ncpu)"
else
    echo "$ARCH is not supported"
    exit 1
fi

if [ "$PIP" == "" ]; then
    PIP="pip"
fi

mkdir -p build
cd build
cmake ..
make -j $NPROC create_python_pkg
cd python/package
$PIP install -U --no-deps .
