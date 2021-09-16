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

import logging

from pygeoapi.process.base import (BaseProcessor, ProcessorExecuteError)

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
    'links': [{
        'type': 'text/html',
        'rel': 'canonical',
        'title': 'information',
        'href': 'https://github.com/52North/Landsat-classification/blob/main/README.md',
        'hreflang': 'en-US'
    }],
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
        'echo': {
            'title': 'Landcover prediction',
            'description': 'Landcover prediction with Landsat 8 Collection 2 Level 2 for water, herbs and coniferous',
            'schema': {
                'type': 'object',
                'contentMediaType': 'application/json'
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


class LandcoverPredictionProcessor(BaseProcessor):
    """Landcover Prediction Processor"""

    def __init__(self, processor_def):
        """
        Initialize object

        :param processor_def: provider definition

        :returns: odcprovider.processes.LandcoverPredictionProcessor
        """

        super().__init__(processor_def, PROCESS_METADATA)

    def execute(self, data):
        # Workflow:
        bbox, collection_id = self.parse_inputs(data)

        # 2) Get array to use for the prediction with the correct bbox
        #    a) either using open data cube directly or
        #    b) making a coverage request (may be slower but enables usage of external collections)
        # 3) If necessary adapt this function https://github.com/SufianZa/Landsat-classification/blob/main/u_net.py#L208
        #       to use, e.g., array input instead of path
        # 4) Make the prediction using this method https://github.com/SufianZa/Landsat-classification/blob/main/test.py
        # 5) Correctly encode the result of 4) as process output (geotiff)

        outputs = [{
            'id': 'echo',
            'collection_id': collection_id,
            'bbox': bbox
        }]
        mimetype = 'application/json'
        return mimetype, outputs

    def parse_inputs(self, data):
        # 1) Parse process inputs
        collection_id = data.get('landsat-collection-id', None)
        bbox = data.get('bbox', None)
        if collection_id is None:
            raise ProcessorExecuteError('Cannot process without a collection_id')
        if bbox is None:
            raise ProcessorExecuteError('Cannot process without a bbox')
        LOGGER.debug('Process inputs:\n - collection_id: {}\n - bbox: {}'.format(collection_id, bbox))
        LOGGER.debug(type(bbox))
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
