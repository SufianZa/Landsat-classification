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
from __future__ import annotations

import json
import logging
import os

import time
from typing import Tuple, Any
from urllib.error import HTTPError

import requests
from landsatpredictor.u_net import UNET
from pygeoapi.process.base import (BaseProcessor, ProcessorExecuteError)

BASE_URL = "https://17.testbed.dev.52north.org/geodatacube/collections/{}/coverage?f=NetCDF&bbox={}"

LOGGER = logging.getLogger(__name__)

#
# LINKS
#
# Process inputs
#   https://github.com/opengeospatial/ogcapi-processes/blob/master/core/examples/json/ProcessDescription.json#L14
#   http://docs.ogc.org/DRAFTS/18-062.html#sc_process_inputs
#   Bbox:
#   http://docs.ogc.org/DRAFTS/18-062.html#bbox-schema
#   https://github.com/opengeospatial/ogcapi-coverages#query-parameters-optional-conformance-classes
#
# Process outputs
#   https://github.com/opengeospatial/ogcapi-processes/blob/master/core/examples/json/ProcessDescription.json#L199
#   Image
#   https://github.com/opengeospatial/ogcapi-processes/blob/master/core/examples/json/ProcessDescription.json#L318-L325
#
# Implementation
# Async processing pygeoapi:
#       https://docs.pygeoapi.io/en/latest/data-publishing/ogcapi-processes.html#asynchronous-support
#
PROCESS_METADATA = {
    'version': '0.1.0',
    'id': 'landcover-prediction',
    'title': 'Landcover prediction',
    'description': 'Landcover prediction with landsat',
    'keywords': ['landcover prediction', 'landsat', 'tb-17'],
    'jobControlOptions': 'async-execute',
    'outputTransmission': ['value'],
    'links': [
        {
            'type': 'text/html',
            'rel': 'canonical',
            'title': 'Processor Repository',
            'href': 'https://github.com/52North/Landsat-classification/blob/main/README.md',
            'hreflang': 'en-US'
        },
        {
            'type': 'text/html',
            'rel': 'canonical',
            'title': 'Landsat 8 Collection 2 Level 2',
            'href': 'https://www.usgs.gov/core-science-systems/nli/landsat/landsat-collection-2-level-2-science-products',
            'hreflang': 'en-US'
        }
    ],
    'inputs': {
        'landsat-collection-id': {
            'title': 'Name',
            'description': 'id of the OGC API coverages collection providing the landsat data',
            'schema': {
                'type': 'string'
            },
            'minOccurs': 1,
            'maxOccurs': 1,
            'metadata': None,  # TODO how to use?
            'keywords': ['landsat']
        },
        'bbox': {
            'title': 'Spatial bounding box',
            'description': 'Spatial bounding box in WGS84',
            'schema': {
                'type': 'string'
            },
            'minOccurs': 1,
            'maxOccurs': 1,
            'metadata': None,
            'keywords': ['bbox']
        }
    },
    'outputs': {
        'prediction': {
            'title': 'Landcover prediction',
            'description':
                'Landcover prediction with Landsat 8 Collection 2 Level 2 for no change, water, herbs and coniferous',
            'schema': {
                'type': 'string',
                'format': 'byte',
                'contentMediaType': 'image/tiff; application=geotiff'
            }
        }
    },
    'example': {
        "inputs": {
            "landsat-collection-id": "landsat8_c2_l2",
            "bbox": "-111.0,64.99,-110.99,65.0"
        }
    }
}


class ModelCache:
    """
    Stores not changing trained model to be used by each instance of the LandcoverPredictionProcessor.
    Implementation follows:

        https://python-patterns.guide/gang-of-four/singleton/
    """

    _instance = None

    def __init__(self):
        raise RuntimeError('Call instance() instead')

    @classmethod
    def instance(cls) -> ModelCache:
        LOGGER.debug("instance() called")
        #
        #   Init model and its data as global pickled singleton
        #
        if cls._instance is None:
            LOGGER.debug("Creating instance of class '{}'...".format(UNET))
            cls._instance = UNET()
            LOGGER.debug('...DONE.')
        else:
            LOGGER.debug("Instance of class '{}' already existing".format(UNET))
        return cls._instance


class LandcoverPredictionProcessor(BaseProcessor):
    """Landcover Prediction Processor"""

    def __init__(self, processor_def):
        """
        Initialize object

        :param processor_def: provider definition

        :returns: odcprovider.processes.LandcoverPredictionProcessor
        """

        super().__init__(processor_def, PROCESS_METADATA)
        self.model = ModelCache.instance()

    def execute(self, data: dict) -> Tuple[str, Any]:
        # Workflow:
        bbox, collection_id = self._parse_inputs(data)

        # 2) Get array to use for the prediction with the correct bbox
        #    a) either using open data cube directly or
        #    b) making a coverage request (may be slower but enables usage of external collections)
        # ToDo add rangeset subset parameter to specify the required bands
        request = BASE_URL.format(collection_id, bbox)
        LOGGER.debug("Requesting coverage from '{}'".format(request))
        try:
            with requests.get(request, verify=False, stream=True) as request:
                request.raise_for_status()
                # ToDo use correct temp file and use tmp file name as input for unet.estimate_raw
                with open('/tmp/temp.geotiff', 'wb') as file:
                    for chunk in request.iter_content(chunk_size=8192):
                        file.write(chunk)
        except HTTPError as err:
            msg = 'Requesting input data failed: {}'.format(request)
            LOGGER.error(msg)
            raise ProcessorExecuteError(msg)
        # write response to temporary file used as input for prediction/estimation function
        # ToDo replace next line with correct tempfile from above
        tmp_file = os.getcwd() + '/tests/data/LC08_L2SP_035024_20150813_20200909_02_T1_merged_1-6.tif'

        # 3) If necessary adapt this function https://github.com/SufianZa/Landsat-classification/blob/main/u_net.py#L208
        #       to use, e.g., array input instead of path
        # 4) Make the prediction using this method https://github.com/SufianZa/Landsat-classification/blob/main/test.py
        # 5) Correctly encode the result of 4) as process output (geotiff)
        LOGGER.debug('Requesting prediction for file "{}"'.format(tmp_file))
        result_file_path = self.model.estimate_raw_landsat(path=tmp_file, trim=20)
        LOGGER.debug('Prediction received. Result in "{}"'.format(result_file_path))

        mimetype = 'image/tiff; application=geotiff'
        with open(result_file_path, 'r+b') as file:
            return mimetype, file.read()

    def _parse_inputs(self, data):
        LOGGER.debug("RAW Inputs:\n{}".format(json.dumps(data, indent=4)))
        # 1) Parse process inputs
        collection_id = data.get('landsat-collection-id', None)
        bbox = data.get('bbox', None)
        if collection_id is None:
            raise ProcessorExecuteError('Cannot process without a collection_id')
        if bbox is None:
            raise ProcessorExecuteError('Cannot process without a bbox')
        LOGGER.debug('Parsed Process inputs')
        LOGGER.debug('collection_id: {}'.format(collection_id))
        LOGGER.debug('bbox         : {}'.format(bbox))
        bbox_coords = [s.strip() for s in bbox.split(",")]
        if len(bbox_coords) != 4:
            raise ProcessorExecuteError("Received bbox '{}' could not be split into four (4) elements by ','."
                                        .format(bbox))
        bbox_float_coords = list(map(float, bbox_coords))
        if not all(isinstance(x, float) for x in bbox_float_coords):
            raise ProcessorExecuteError("Received bbox '{}' could not be converted completly to integer."
                                        .format(bbox))
        return bbox, collection_id

    def __repr__(self):
        return '<LandcoverPredictionProcessor> {}'.format(self.name)
