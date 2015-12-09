#! /usr/bin/env bash

mkdir -p build
cd build
cmake ..
make create_python_pkg
cd python/package
pip install .
