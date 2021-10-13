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
from pathlib import Path
import rasterio
from src.landsatpredictor.u_net import UNET
from src.landsatpredictor.preprocessing.image_registration import get_multi_spectral

model = UNET(batch_size=24, epochs=50)

landsat_file_path = Path('test', 'LC08_L2SP_035024')
with rasterio.open(landsat_file_path) as dataset:
    input_landsat_bands_normalized, visual_light_reflectance_mask, metadata = get_multi_spectral(dataset)

model.estimate_raw_landsat(input_landsat_bands_normalized, visual_light_reflectance_mask, metadata, trim=5)
