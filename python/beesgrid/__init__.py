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


import numpy as np
import warnings

TAG_ID = ['bits']
TAG_CONFIG = ['z_rotation', 'y_rotation', 'x_rotation', 'center', 'radius']
TAG_STRUCTURE = ['inner_ring_radius', 'middle_ring_radius', 'outer_ring_radius', 'bulge_factor',
                 'focal_length']

TAG_LABEL_NAMES = TAG_ID + TAG_CONFIG + TAG_STRUCTURE

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


def dtype_tag_params(nb_bits=12, with_structure=False):
    keys = TAG_ID + TAG_CONFIG
    if with_structure:
        keys += TAG_STRUCTURE
    reps = {key: 1 for key in keys}
    reps['bits'] = nb_bits
    reps['center'] = 2
    return [(key, "({},)float32".format(n)) for key, n in reps.items()]


def draw_grids(params, with_structure='auto', scales=[1.], artist=None):
    def get_positions(keys):
        positions = {}
        i = 0
        for name in keys:
            positions[name] = i
            i += len(params[name][0])
        return positions, i

    def array_fill_by_keys(struct_arr, keys, positions, arr):
        for name in keys:
            b = positions[name]
            e = b + len(struct_arr[name][0])
            arr[:, b:e] = struct_arr[name]

    if artist is None:
        artist = BlackWhiteArtist(0, 255, 0, 1)

    batch_size = len(params['bits'])
    positions, size = get_positions(TAG_ID + TAG_CONFIG)
    bits_and_config = np.zeros((batch_size, size), dtype=np.float32)
    array_fill_by_keys(params, TAG_ID + TAG_CONFIG, positions, bits_and_config)

    if with_structure == 'auto':
        with_structure = all([struct_key in params.dtype.names for struct_key in TAG_STRUCTURE])
    if with_structure:
        struct_positions, struct_size = get_positions(TAG_STRUCTURE)
        structure = np.zeros((batch_size, struct_size), dtype=np.float32)
        array_fill_by_keys(params, TAG_STRUCTURE, struct_positions, structure)
        structure = np.ascontiguousarray(structure)
    else:
        structure = None
    bits_and_config = np.ascontiguousarray(bits_and_config)
    if structure is not None and (structure == 0).all():
        warnings.warn(
            "draw_grids got a structure that is all zero. Did you use "
            "`dtype_tag_params(with_structure=True)`"
            " and forgot to set the structure?")

    assert bits_and_config.dtype == np.float32
    assert bits_and_config.flags['C_CONTIGUOUS']
    return drawGrids(bits_and_config, structure, artist, scales)


def _normalize_angle(x):
    x %= 2*np.pi
    x = (x + 2*np.pi) % (2*np.pi)
    x[x > np.pi] -= 2*np.pi
    assert ((-np.pi <= x) & (x <= np.pi)).all()
    return x
