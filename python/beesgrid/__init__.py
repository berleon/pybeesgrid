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
import collections
import glob

from .pybeesgrid import TAG_SIZE, NUM_CONFIGS, NUM_MIDDLE_CELLS
from .pybeesgrid import GridGenerator, BadGridArtist, BlackWhiteArtist, \
    MaskGridArtist, DepthMapArtist
from .pybeesgrid import drawGrids

from .pybeesgrid import INNER_BLACK_SEMICIRCLE, CELL_0_BLACK, CELL_1_BLACK, \
    CELL_2_BLACK, CELL_3_BLACK, CELL_4_BLACK, CELL_5_BLACK, CELL_6_BLACK, \
    CELL_7_BLACK, CELL_8_BLACK, CELL_9_BLACK, CELL_10_BLACK, CELL_11_BLACK, \
    IGNORE, CELL_0_WHITE, CELL_1_WHITE, CELL_2_WHITE, CELL_3_WHITE,  \
    CELL_4_WHITE, CELL_5_WHITE, CELL_6_WHITE, CELL_7_WHITE, CELL_8_WHITE, \
    CELL_9_WHITE, CELL_10_WHITE, CELL_11_WHITE, OUTER_WHITE_RING, \
    INNER_WHITE_SEMICIRCLE
from . import pybeesgrid as pybg

import numpy as np


__all__ = ["TAG_SIZE", "NUM_MIDDLE_CELLS", "CONFIG_LABELS", "NUM_CONFIGS",
           "draw_grids", "GridGenerator", "BadGridArtist", "BlackWhiteArtist",
           "MaskGridArtist", "DepthMapArtist", "MASK", "MASK_KEYS",
           "MASK_BLACK", "MASK_WHITE", "CELLS_BLACK", "CELLS_WHITE"]


CONFIG_LABELS = ('z_rotation', 'y_rotation', 'x_rotation',
                 'center_x', 'center_y', 'radius')

CONFIG_ROTS = (
    CONFIG_LABELS.index('z_rotation'),
    CONFIG_LABELS.index('y_rotation'),
    CONFIG_LABELS.index('x_rotation'),
)

CONFIG_CENTER = (
    CONFIG_LABELS.index('center_x'),
    CONFIG_LABELS.index('center_y'),
)

GRID_STRUCTURE_LABELS = (
    'inner_ring_radius',
    'middle_ring_radius',
    'outer_ring_radius',
    'bulge_factor',
    'focal_length'
)

GRID_STRUCTURE_POS = collections.OrderedDict([
    ('inner_ring_radius', GRID_STRUCTURE_LABELS.index('inner_ring_radius')),
    ('middle_ring_radius', GRID_STRUCTURE_LABELS.index('middle_ring_radius')),
    ('outer_ring_radius', GRID_STRUCTURE_LABELS.index('outer_ring_radius')),
    ('bulge_factor', GRID_STRUCTURE_LABELS.index('bulge_factor')),
    ('focal_length', GRID_STRUCTURE_LABELS.index('focal_length')),
])

CONFIG_RADIUS = CONFIG_LABELS.index('radius')

MASK = collections.OrderedDict([
    ("INNER_BLACK_SEMICIRCLE", INNER_BLACK_SEMICIRCLE),
    ("CELL_0_BLACK",           CELL_0_BLACK),
    ("CELL_1_BLACK",           CELL_1_BLACK),
    ("CELL_2_BLACK",           CELL_2_BLACK),
    ("CELL_3_BLACK",           CELL_3_BLACK),
    ("CELL_4_BLACK",           CELL_4_BLACK),
    ("CELL_5_BLACK",           CELL_5_BLACK),
    ("CELL_6_BLACK",           CELL_6_BLACK),
    ("CELL_7_BLACK",           CELL_7_BLACK),
    ("CELL_8_BLACK",           CELL_8_BLACK),
    ("CELL_9_BLACK",           CELL_9_BLACK),
    ("CELL_10_BLACK",          CELL_10_BLACK),
    ("CELL_11_BLACK",          CELL_11_BLACK),
    ("IGNORE",                 IGNORE),
    ("CELL_0_WHITE",           CELL_0_WHITE),
    ("CELL_1_WHITE",           CELL_1_WHITE),
    ("CELL_2_WHITE",           CELL_2_WHITE),
    ("CELL_3_WHITE",           CELL_3_WHITE),
    ("CELL_4_WHITE",           CELL_4_WHITE),
    ("CELL_5_WHITE",           CELL_5_WHITE),
    ("CELL_6_WHITE",           CELL_6_WHITE),
    ("CELL_7_WHITE",           CELL_7_WHITE),
    ("CELL_8_WHITE",           CELL_8_WHITE),
    ("CELL_9_WHITE",           CELL_9_WHITE),
    ("CELL_10_WHITE",          CELL_10_WHITE),
    ("CELL_11_WHITE",          CELL_11_WHITE),
    ("OUTER_WHITE_RING",       OUTER_WHITE_RING),
    ("INNER_WHITE_SEMICIRCLE", INNER_WHITE_SEMICIRCLE)
])

MASK_KEYS = list(MASK.keys())
CELLS_BLACK = MASK_KEYS[MASK_KEYS.index("CELL_0_BLACK"):MASK_KEYS.index("CELL_11_BLACK")+1]
MASK_BLACK = ["INNER_BLACK_SEMICIRCLE"] + CELLS_BLACK
CELLS_WHITE = MASK_KEYS[
              MASK_KEYS.index("CELL_0_WHITE"):
              MASK_KEYS.index("CELL_11_WHITE")+1]

MASK_WHITE = CELLS_WHITE + ["OUTER_WHITE_RING", "INNER_WHITE_SEMICIRCLE"]


def draw_grids(ids: np.ndarray, configs: np.ndarray, structure=None, scales=[1.], artist=None):
    if artist is None:
        artist = BlackWhiteArtist(0, 255, 0, 1)

    bits_and_config = np.concatenate((ids, configs), axis=1)
    if structure is not None:
        structure = np.ascontiguousarray(structure)
    grids = drawGrids(np.ascontiguousarray(bits_and_config), structure, artist, scales)
    return grids


def _normalize_angle(x):
    x %= 2*np.pi
    x = (x + 2*np.pi) % (2*np.pi)
    x[x > np.pi] -= 2*np.pi
    assert ((-np.pi <= x) & (x <= np.pi)).all()
    return x


def get_gt_files_in_dir(directory):
    """Returns all `*.tdat` files in the directory."""
    return glob.glob(directory + '/**/*.tdat', recursive=True)


def gt_grids(gt_files, batch_size=64, repeat=False, all=False,
             center='coordinates'):
    """
    Returns a `(images, id, config)` tuple, where `images` are the images of
    the bees, `id` is a (batch_size, 12) matrix of the bees' ids and config is
    a (batch_size, 6) matrix with the 'z_rotation', 'y_rotation', 'x_rotation',
    'center_x', 'center_y' and 'radius' in this order.

    If `all` is True, then all ground truth tags are returned.
    `center` returns the image coordinates for 'coordinates' or can be set to
    zeros with 'zeros'.
    """
    if all:
        batch_size = np.iinfo(np.int64).max
    gt_loader = pybg.GTDataLoader(gt_files)
    while True:
        batch = gt_loader.batch(batch_size, repeat)
        if batch is None:
            break
        else:
            gt, ids, configs = batch
            gt = gt.astype(np.float32) / 255.
            z, _, x = CONFIG_ROTS
            configs[:, z:x+1] = _normalize_angle(configs[:, z:x+1])
            if center.startswith('zero'):
                configs[:, CONFIG_CENTER] = 0
            yield gt, ids, configs


def generate_grids(batch_size=64, generator=None, with_gird_params=False,
                   artist=None, scales=[1.]):
    """
    Returns a tuple with `(b_1, b_2, .., b_m, label, grid_params)`,
    where `b_i` is the batch with scale `scales[i]`.
    """
    if generator is None:
        generator = GridGenerator()
    if artist is None:
        artist = BadGridArtist()
    while True:
        batch = pybg.generateBatch(generator, artist, batch_size, scales)
        if with_gird_params:
            yield batch
        else:
            yield batch[:len(scales)+1]

