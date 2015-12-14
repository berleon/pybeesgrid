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
import os

import pytest
import scipy.misc
import time

import sys

from .. import TAG_SIZE, NUM_CONFIGS, CONFIG_LABELS, draw_grids, \
        generate_grids, gt_grids, NUM_MIDDLE_CELLS, CONFIG_ROTS

from timeit import Timer
import numpy as np


def test_generate_grids():
    bs = 64
    grids, labels = next(generate_grids(bs))
    assert grids.shape == (bs, 1, TAG_SIZE, TAG_SIZE)
    assert labels.shape == (bs, NUM_MIDDLE_CELLS)


def test_generate_grids_scaled():
    bs = 64
    grids_1, grids_05, grids_025, labels = next(generate_grids(bs, scales=[1, 0.5, 0.25]))
    assert grids_1.shape == (bs, 1, TAG_SIZE, TAG_SIZE)
    assert grids_05.shape == (bs, 1, TAG_SIZE//2, TAG_SIZE//2)
    assert grids_025.shape == (bs, 1, TAG_SIZE//4, TAG_SIZE//4)
    assert labels.shape == (bs, NUM_MIDDLE_CELLS)


def test_gt_loader_bs():
    bs = 64
    gt_files = ["../../src/test/testdata/Cam_0_20140804152006_3.tdat"] * 3
    last_iteration_occured = False
    for grids, bits, config in gt_grids(gt_files, batch_size=bs):
        if grids.shape[0] < bs:
            assert not last_iteration_occured
            last_iteration_occured = True
            bs = grids.shape[0]

        assert grids.shape == (bs, 1, TAG_SIZE, TAG_SIZE)
        assert bits.shape == (bs, NUM_MIDDLE_CELLS)
        assert config.shape == (bs, NUM_CONFIGS)

        z, y, x = CONFIG_ROTS
        assert ((-np.pi < config[:, z]) & (config[:, z] <= np.pi)).all()
        assert ((-np.pi < config[:, y]) & (config[:, y] <= np.pi)).all()
        assert ((-np.pi < config[:, x]) & (config[:, x] <= np.pi)).all()


def test_gt_loader_all():
    gt_files = ["../../src/test/testdata/Cam_0_20140804152006_3.tdat"] * 3
    for grids, bits, config in gt_grids(gt_files, all=True):
        assert grids.shape[0] >= 300


def test_benchmark():
    bs = 6400
    n = 10
    t = Timer(lambda: next(generate_grids(bs)))
    print("need {:.5f}s for {} grids".format(t.timeit(n) / n, bs))


def test_draw_grids_checks_shape():
    bs = 64
    too_many_dims = NUM_CONFIGS + 10
    bits = np.random.binomial(1, 0.5, (bs, NUM_MIDDLE_CELLS)).astype(np.float32)
    configs = np.zeros((bs, too_many_dims), dtype=np.float32)
    with pytest.raises(ValueError):
        draw_grids(bits, configs)


def test_draw_grids_checks_dims():
    bs = 64
    too_many_dims = bs
    bits = np.random.binomial(1, 0.5, (bs, too_many_dims, NUM_MIDDLE_CELLS)).astype(np.float32)
    configs = np.zeros((bs, too_many_dims, NUM_CONFIGS), dtype=np.float32)
    with pytest.raises(ValueError):
        draw_grids(bits, configs)


def test_draw_grids():
    bs = 256
    bits = np.random.binomial(1, 0.5, (bs, NUM_MIDDLE_CELLS)).astype(np.float32)
    configs = np.zeros((bs, NUM_CONFIGS), dtype=np.float32)
    configs[:, CONFIG_LABELS.index('center_x')] = TAG_SIZE // 2
    configs[:, CONFIG_LABELS.index('center_y')] = TAG_SIZE // 2
    configs[:, CONFIG_LABELS.index('radius')] = np.linspace(0, 32, num=bs)
    grids, = draw_grids(bits, configs)
    output_dir = "testout"
    os.makedirs(output_dir, exist_ok=True)
    for i in range(len(grids)):
        scipy.misc.imsave(output_dir + '/grid_{:03d}.png'.format(i), grids[i, 0])

