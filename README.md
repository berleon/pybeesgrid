# pybeesgrid

This library can load Beesbook ground truth data and draw generated grids.


## build python module


```bash
$ build_and_install.sh
```

Now you can import the python package with:

```python
import deepdecoder.generate_grids as gen_grids

for data, labels in gen_grids.batches(batch_size=32):
    pass
```


