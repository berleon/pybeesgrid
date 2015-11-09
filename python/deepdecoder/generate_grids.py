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
#

from . import pydeepdecoder as pydd

from .pydeepdecoder import INNER_BLACK_SEMICIRCLE, CELL_0_BLACK, CELL_1_BLACK, \
     CELL_2_BLACK, CELL_3_BLACK, CELL_4_BLACK, CELL_5_BLACK, CELL_6_BLACK, \
     CELL_7_BLACK, CELL_8_BLACK, CELL_9_BLACK, CELL_10_BLACK, CELL_11_BLACK, \
     BACKGROUND_RING, IGNORE, CELL_0_WHITE, CELL_1_WHITE, CELL_2_WHITE, \
     CELL_3_WHITE,  CELL_4_WHITE, CELL_5_WHITE, CELL_6_WHITE, CELL_7_WHITE, \
     CELL_8_WHITE, CELL_9_WHITE, CELL_10_WHITE, CELL_11_WHITE, OUTER_WHITE_RING, \
     INNER_WHITE_SEMICIRCLE
import collections

MASK = collections.OrderedDict([
    ("INNER_BLACK_SEMICIRCLE", INNER_BLACK_SEMICIRCLE),
    ("CELL_0_BLACK", CELL_0_BLACK),
    ("CELL_1_BLACK", CELL_1_BLACK),
    ("CELL_2_BLACK", CELL_2_BLACK),
    ("CELL_3_BLACK", CELL_3_BLACK),
    ("CELL_4_BLACK", CELL_4_BLACK),
    ("CELL_5_BLACK", CELL_5_BLACK),
    ("CELL_6_BLACK", CELL_6_BLACK),
    ("CELL_7_BLACK", CELL_7_BLACK),
    ("CELL_8_BLACK", CELL_8_BLACK),
    ("CELL_9_BLACK", CELL_9_BLACK),
    ("CELL_10_BLACK", CELL_10_BLACK),
    ("CELL_11_BLACK", CELL_11_BLACK),
    ("BACKGROUND_RING", BACKGROUND_RING),
    ("IGNORE", IGNORE),
    ("CELL_0_WHITE", CELL_0_WHITE),
    ("CELL_1_WHITE", CELL_1_WHITE),
    ("CELL_2_WHITE", CELL_2_WHITE),
    ("CELL_3_WHITE", CELL_3_WHITE),
    ("CELL_4_WHITE", CELL_4_WHITE),
    ("CELL_5_WHITE", CELL_5_WHITE),
    ("CELL_6_WHITE", CELL_6_WHITE),
    ("CELL_7_WHITE", CELL_7_WHITE),
    ("CELL_8_WHITE", CELL_8_WHITE),
    ("CELL_9_WHITE", CELL_9_WHITE),
    ("CELL_10_WHITE", CELL_10_WHITE),
    ("CELL_11_WHITE", CELL_11_WHITE),
    ("OUTER_WHITE_RING", OUTER_WHITE_RING),
    ("INNER_WHITE_SEMICIRCLE", INNER_WHITE_SEMICIRCLE)
])

MASK_KEYS = list(MASK.keys())
CELLS_BLACK = MASK_KEYS[MASK_KEYS.index("CELL_0_BLACK"):MASK_KEYS.index("CELL_11_BLACK")+1]
MASK_BLACK = ["INNER_BLACK_SEMICIRCLE"] + CELLS_BLACK
CELLS_WHITE = MASK_KEYS[
              MASK_KEYS.index("CELL_0_WHITE"):
              MASK_KEYS.index("CELL_11_WHITE")+1]

MASK_WHITE = CELLS_WHITE + ["OUTER_WHITE_RING", "INNER_WHITE_SEMICIRCLE"]

GridGenerator = pydd.GridGenerator
BadGridArtist = pydd.BadGridArtist
BlackWhiteArtist = pydd.BlackWhiteArtist
MaskGridArtist = pydd.MaskGridArtist


def batches(batch_size=64, generator=None, with_gird_params=False,
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
        batch = pydd.generateBatch(generator, artist, batch_size, scales)
        if with_gird_params:
            yield batch
        else:
            yield batch[:len(scales)+1]

