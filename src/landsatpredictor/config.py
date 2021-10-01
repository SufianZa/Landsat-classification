# =================================================================
# Copyright (C) 2021-2021 52°North Spatial Information Research GmbH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# =================================================================
from matplotlib import patches, pyplot as plt

selected_classes = ['no_change', 'water', 'coniferous', 'herbs']

original_classes = dict(no_change=0,
                        water=20,
                        snow_ice=31,
                        rock_rubble=32,
                        exposed_barren_land=33,
                        bryoids=40,
                        shrubland=50,
                        wetland=80,
                        wetlandtreed=81,
                        herbs=100,
                        coniferous=210,
                        broadleaf=220,
                        mixedwood=230)


if len(selected_classes) == 0:
    selected_classes = list(original_classes.keys())
model_classes = {c: idx for idx, c in enumerate(original_classes) if c in selected_classes}

colors = [(0, 0, 0)] + list(plt.cm.get_cmap('Paired').colors)
colors_legend = [patches.Patch(color=colors[i], label=c) for i, c in enumerate(original_classes) if
                 c in selected_classes]
colors = [colors[i] for i, c in enumerate(original_classes) if c in selected_classes]

LANDSAT8_REFLECTANCE_BAND_MAX_VALUE = 65455
"""
Max value for Landsat 8 reflectance bands.

See https://www.usgs.gov/core-science-systems/nli/landsat/landsat-collection-2-level-2-science-products
"""

PADDING_EDGE = 100

# folder s
LAND_COVER_FILE = "./CA_forest_VLCE_2015/CA_forest_VLCE_2015.tif"
TRAIN_DATASETS = 'train'
TEST_DATASETS = 'test'

SUPPORTED_BANDS = [2, 3, 4, 5, 6, 7]
