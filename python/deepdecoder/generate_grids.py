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

GridGenerator = pydd.GridGenerator


def batches(batch_size=64, generator=None, with_gird_params=False):
    if generator is None:
        generator = GridGenerator()
    while True:
        batch = pydd.generateBatch(generator, batch_size)
        if with_gird_params:
            yield batch
        else:
            yield batch[:2]

