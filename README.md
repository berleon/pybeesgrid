# deepdecoder

This projects aims to train a neuronal network to decode the bees tags.

## build 

For now you have to set the `SOURCE_DIR` of the BioTracker dependency manually.

## build python module

```bash

$ mkdir build
$ cd build
$ cmake ..
$ make create_python_pkg
$ cd python/package
$ pip install .
```

The `create_python_pkg` target automatically copies the compiled c++ python
module to the right place.

Now you can import the python package with:

```python
import deepdecoder.generate_grids as gen_grids

for data, labels in gen_grids.batches(batch_size=32):
    pass
```


