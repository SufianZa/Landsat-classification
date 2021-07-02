import json
from landsatxplore.api import API
from landsatxplore.earthexplorer import EarthExplorer
import os

username, password = os.environ['UN_EarthE'], os.environ['PW_EarthE']
file = "C:\\Users\\Sufian\\Downloads\\CA_forest_VLCE_2015\\CA_forest_VLCE_2015.tif"

# Initialize a new API instance and get an access key
api = API(username, password)

# search area
west, south, east, north = -127.4439, 48.9225, -84.1984, 58.9953

start_date = '2015-07-15'
end_date = '2015-08-30'

# Search for Landsat 8 https://pypi.org/project/landsatxplore/
scenes = api.search(
    dataset='landsat_ot_c2_l2',
    bbox=(west, south, east, north),
    start_date=start_date,
    end_date=end_date,
    max_cloud_cover=1
)

print(f"{len(scenes)} scenes found.")
api.logout()

ee = EarthExplorer(username, password)
# Process the result
for scene in scenes:
    print(scene['entity_id'])
    try:
        ee.download(identifier=scene['entity_id'], output_dir='./data')
    except Exception as e:
        print(e)
ee.logout()
