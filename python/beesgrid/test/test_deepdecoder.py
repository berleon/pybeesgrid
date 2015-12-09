# Copyright 2015 Leon Sixt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .. import TAG_SIZE, NUM_CELLS
from ..generate_grids import batches as gen_grid_batches
from ..gt_grids import batches as gt_batches

from timeit import Timer


def test_generate_grids():
    bs = 64
    grids, labels = next(gen_grid_batches(bs))
    assert grids.shape == (bs, 1, TAG_SIZE, TAG_SIZE)
    assert labels.shape == (bs, NUM_CELLS)


def test_generate_grids_scaled():
    bs = 64
    grids_1, grids_05, grids_025, labels = next(gen_grid_batches(bs, scales=[1, 0.5, 0.25]))
    assert grids_1.shape == (bs, 1, TAG_SIZE, TAG_SIZE)
    assert grids_05.shape == (bs, 1, TAG_SIZE//2, TAG_SIZE//2)
    assert grids_025.shape == (bs, 1, TAG_SIZE//4, TAG_SIZE//4)
    assert labels.shape == (bs, NUM_CELLS)


def test_gt_loader():
    bs = 64
    gt_files = ["../../src/test/testdata/Cam_0_20140804152006_3.tdat"] * 3
    for grids, labels in gt_batches(gt_files, batch_size=bs):
        print(TAG_SIZE)
        assert grids.shape == (bs, 1, TAG_SIZE, TAG_SIZE)
        assert labels.shape == (bs, NUM_CELLS)


def test_benchmark():
    bs = 6400
    n = 10
    t = Timer(lambda: next(gen_grid_batches(bs)))
    print("need {:.5f}s for {} grids".format(t.timeit(n) / n, bs))

