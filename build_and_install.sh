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

mkdir -p build
cd build
cmake ..
make -j $NPROC create_python_pkg
cd python/package
pip install -U --no-deps .
