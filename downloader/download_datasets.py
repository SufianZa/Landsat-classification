from pathlib import Path

from landsatxplore.api import API
from landsatxplore.earthexplorer import EarthExplorer
import os
from tqdm import tqdm
import requests

# download land cover dataset
land_cover_url = 'https://opendata.nfis.org/downloads/forest_change/CA_forest_VLCE_2015.zip'
response = requests.get(land_cover_url, stream=True)
Path('./landcover').mkdir(parents=True, exist_ok=True)
with open(str(Path('./landcover', 'land_cover.zip')), "wb") as handle:
    for data in tqdm(response.iter_content()):
        handle.write(data)

# download landsat dataset
username, password = os.getenv('UN_EarthE'), os.getenv('PW_EarthE')
api = API(username, password)

# search area
west, south, east, north = -127.4439, 48.9225, -84.1984, 58.9953

# search time range
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

for scene in scenes:
    print(scene['entity_id'])
    try:
        ee.download(identifier=scene['entity_id'], dataset='landsat_ot_c2_l2', output_dir='../train')
    except Exception as e:
        print(e)
ee.logout()
