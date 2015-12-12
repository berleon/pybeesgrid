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

from .pybeesgrid import TAG_SIZE as CXX_TAG_SIZE
from .pybeesgrid import GridGenerator, BadGridArtist, BlackWhiteArtist, MaskGridArtist
from .pybeesgrid import drawGrids

import numpy as np

TAG_SIZE = CXX_TAG_SIZE
NUM_CELLS = 12

CONFIG_LABELS = ('z_rotation', 'y_rotation', 'x_rotation',
                 'center_x', 'center_y', 'radius')

NUM_CONFIG = len(CONFIG_LABELS)


def draw_grids(bits: np.ndarray, configs: np.ndarray, scales=[1.], artist=None):
    if artist is None:
        artist = BlackWhiteArtist()

    bits_and_config = np.concatenate((bits, configs), axis=1)
    return drawGrids(bits_and_config, artist, scales)