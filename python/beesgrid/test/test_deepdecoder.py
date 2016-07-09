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

import scipy.misc

from .. import TAG_SIZE, NUM_CONFIGS, draw_grids, \
        NUM_MIDDLE_CELLS, \
        MaskGridArtist, BadGridArtist, BlackWhiteArtist, DepthMapArtist, \
        dtype_tag_params

from timeit import Timer
import numpy as np
import pytest


def test_draw_benchmark():
    bs = 64
    n = 100
    params = np.zeros(bs, dtype_tag_params(with_structure=False))
    params['bits'] = np.random.binomial(1, 0.5, (bs, NUM_MIDDLE_CELLS))
    params['center'] = TAG_SIZE // 2
    params['center'] = TAG_SIZE // 2
    params['radius'] = np.linspace(0, 32, num=bs).reshape(-1, 1)
    t = Timer(lambda: draw_grids(params))
    print("need {:.5f}s for {} grids".format(t.timeit(n) / n, bs))


def test_warns_if_structure_is_zero():
    bs = 256
    with pytest.warns(UserWarning):
        params = np.zeros(bs, dtype_tag_params(with_structure=True))
        draw_grids(params)


def test_draw_grids_increasing_radius():
    bs = 256

    params = np.zeros(bs, dtype_tag_params(with_structure=False))
    params['bits'] = np.random.binomial(1, 0.5, (bs, NUM_MIDDLE_CELLS))
    params['center'] = 0
    params['radius'] = np.linspace(0, 32, num=bs).reshape(-1, 1)
    grids, = draw_grids(params)
    output_dir = "testout/draw_grids_increasing_radius"
    os.makedirs(output_dir, exist_ok=True)
    for i in range(len(grids)):
        scipy.misc.imsave(output_dir + '/grid_{:03d}.png'.format(i), grids[i, 0])


def test_draw_grids_paint_scale():
    bs = 256
    artists = [MaskGridArtist(), BlackWhiteArtist(0, 255, 200, 4), BadGridArtist()]
    for artist in artists:

        params = np.zeros(bs, dtype_tag_params())
        params['bits'] = np.random.binomial(1, 0.5, (bs, NUM_MIDDLE_CELLS))

        params['center'] = 0
        params['radius'] = np.linspace(0, 32, num=bs).reshape((-1, 1))
        scales = [2, 1, 0.5]
        grids = draw_grids(params, scales=scales, artist=artist)
        output_dir = "testout"
        os.makedirs(output_dir, exist_ok=True)
        for scale, grid in zip(scales, grids):
            output_dir_artist = output_dir + '/draw_grids_scale/{}_grid_{}'.format(
                type(artist).__name__, scale)
            os.makedirs(output_dir_artist, exist_ok=True)
            for i in range(len(grid)):
                scipy.misc.imsave(
                    output_dir_artist + '/grid_{:03d}.png'.format(i),
                    grid[i, 0])


def test_draw_grids_structure():
    bs = 256
    artist = BlackWhiteArtist(0, 255, 127, 4)
    params = np.zeros(bs, dtype_tag_params(with_structure=True))
    params['bits'] = np.random.binomial(1, 0.5, (bs, NUM_MIDDLE_CELLS))

    params['center'] = 0
    params['radius'] = 24

    params['inner_ring_radius'] = 0.5
    params['middle_ring_radius'] = 0.8
    params['outer_ring_radius'] = 1
    params['bulge_factor'] = 0.4
    params['focal_length'] = 1.0

    grid, = draw_grids(params, with_structure=True, artist=artist)

    output_dir = "testout"
    output_dir_artist = output_dir + '/structure'
    os.makedirs(output_dir_artist, exist_ok=True)
    for i in range(len(grid)):
        scipy.misc.imsave(
            output_dir_artist + '/grid_{:03d}.png'.format(i),
            grid[i, 0])


def test_draw_grids_depth_map():
    bs = 24
    artist = DepthMapArtist()

    params = np.zeros(bs, dtype_tag_params(with_structure=True))
    params['bits'] = np.random.binomial(1, 0.5, (bs, NUM_MIDDLE_CELLS))

    params['z_rotation'] = 0
    params['y_rotation'] = np.random.normal(0, np.pi/6, size=(bs, 1))
    params['x_rotation'] = np.random.normal(0, np.pi/6, size=(bs, 1))
    params['center'] = 0
    params['radius'] = 20

    params['inner_ring_radius'] = 0.45
    params['middle_ring_radius'] = 0.8
    params['outer_ring_radius'] = 1
    params['bulge_factor'] = 0.9
    params['focal_length'] = 4.0


    params_copy = params.copy()
    grid, = draw_grids(params, artist=artist)
    assert (params == params_copy).all()
    bw_artist = BlackWhiteArtist(0, 254, 120, 4)
    bw_artist = DepthMapArtist()
    bw_artist = MaskGridArtist()
    bw_artist = BadGridArtist()

    grid_bw, = draw_grids(params, artist=bw_artist)

    assert (params == params_copy).all()

    output_dir = "testout"
    output_dir_artist = output_dir + '/depth_map'
    os.makedirs(output_dir_artist, exist_ok=True)
    for i in range(len(grid)):
        scipy.misc.imsave(
            output_dir_artist + '/grid_{:03d}.png'.format(i),
            grid[i, 0])

        scipy.misc.imsave(
            output_dir_artist + '/grid_{:03d}_bw.png'.format(i),
            grid_bw[i, 0])
