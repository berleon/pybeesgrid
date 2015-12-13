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

import sys
from .pybeesgrid import TAG_SIZE, NUM_CONFIGS, NUM_MIDDLE_CELLS
from .pybeesgrid import GridGenerator, BadGridArtist, BlackWhiteArtist, \
    MaskGridArtist
from .pybeesgrid import drawGrids

from .pybeesgrid import INNER_BLACK_SEMICIRCLE, CELL_0_BLACK, CELL_1_BLACK, \
    CELL_2_BLACK, CELL_3_BLACK, CELL_4_BLACK, CELL_5_BLACK, CELL_6_BLACK, \
    CELL_7_BLACK, CELL_8_BLACK, CELL_9_BLACK, CELL_10_BLACK, CELL_11_BLACK, \
    BACKGROUND_RING, IGNORE, CELL_0_WHITE, CELL_1_WHITE, CELL_2_WHITE, \
    CELL_3_WHITE,  CELL_4_WHITE, CELL_5_WHITE, CELL_6_WHITE, CELL_7_WHITE, \
    CELL_8_WHITE, CELL_9_WHITE, CELL_10_WHITE, CELL_11_WHITE, \
    OUTER_WHITE_RING, INNER_WHITE_SEMICIRCLE
from . import  pybeesgrid as pybg

import numpy as np


__all__ = ["TAG_SIZE", "NUM_MIDDLE_CELLS", "CONFIG_LABELS", "NUM_CONFIGS",
           "draw_grids", "GridGenerator", "BadGridArtist", "BlackWhiteArtist",
           "MaskGridArtist", "MASK", "MASK_KEYS", "MASK_BLACK", "MASK_WHITE",
            "CELLS_BLACK", "CELLS_WHITE"]



CONFIG_LABELS = ('z_rotation', 'y_rotation', 'x_rotation',
                 'center_x', 'center_y', 'radius')


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
    ("BACKGROUND_RING",        BACKGROUND_RING),
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


def draw_grids(bits: np.ndarray, configs: np.ndarray, scales=[1.], artist=None):
    if artist is None:
        artist = BlackWhiteArtist()

    bits_and_config = np.concatenate((bits, configs), axis=1)
    grids = drawGrids(np.ascontiguousarray(bits_and_config), artist, scales)
    return grids


def gt_grids(gt_files, batch_size=64, repeat=False, all=False):
    """
    Returns a `(images, id, config)` tuple, where `images` are the images of
    the bees, `id` is a (batch_size, 12) matrix of the bees' ids and config is
    a (batch_size, 6) matrix with the 'z_rotation', 'y_rotation', 'x_rotation',
    'center_x', 'center_y' and 'radius' in this order.

    If `all` is True, then all ground truth tags are returned.
    """
    if all:
        batch_size = np.iinfo(np.int64).max
    gt_loader = pybg.GTDataLoader(gt_files)
    while True:
        batch = gt_loader.batch(batch_size, repeat)
        if batch is None:
            break
        else:
            yield batch


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

