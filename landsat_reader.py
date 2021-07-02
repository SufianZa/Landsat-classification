import json
from landsatxplore.api import API
from landsatxplore.earthexplorer import EarthExplorer
import os
username, password = 'szaabalawi', '.x3a9qQEH7CiCRf'
file = "C:\\Users\\Sufian\\Downloads\\CA_forest_VLCE_2015\\CA_forest_VLCE_2015.tif"
os.system('gdalwarp %s %s -t_srs "+proj=longlat +ellps=WGS84"' % (file, 'new_' + file))

# Initialize a new API instance and get an access key
api = API(username, password)

# lat, lon = 55.697639, -105.997849
west = -103.675537
south = 39.911194
east = -74.731979
north = 51.327511
# start_date = '2015-07-01'
# end_date = '2015-08-30'
# # Search for Landsat TM scenes
# scenes = api.search(
#     dataset='landsat_etm_c2_l2',
#     # latitude=lat,
#     # longitude=lon,
#     bbox=(west, south, east, north),
#     start_date=start_date,
#     end_date=end_date,
#     max_cloud_cover=10
# )
#
# print(f"{len(scenes)} scenes found.")
# ee = EarthExplorer(username, password)
# # Process the result
# for scene in scenes:
#     print(scene['acquisition_date'])
#     print(scene.keys())
#     # Write scene footprints to disk
#     fname = f"{scene['landsat_product_id']}.geojson"
#     print(scene['display_id'])
#     print(scene['ordering_id'])
#     print(scene['entity_id'])
#     print('-')
#     try:
#         ee.download(identifier=scene['entity_id'], output_dir='.')
#     except Exception as e:
#         pass
# ee.logout()
# api.logout()

# dat = ['LC80240222015196LGN01', 'LC80240232015196LGN01', 'LC80240242015196LGN01', 'LC80240252015196LGN01',
#        'LC80240262015196LGN01']
# for scene in dat:
#     print(scene)
#     try:
#         ee.download(identifier=scene, output_dir='./data')
#     except Exception as e:
#         pass
from string import Template
from pathlib import Path
import rasterio
import matplotlib.pyplot as plt
from rasterio.warp import calculate_default_transform, reproject, Resampling, aligned_target

dst_crs = 'EPSG:32616'
with rasterio.open(file) as ds:
    dst_crs = ds.crs
t = Template('data/LC08_L1TP_024022_20150715_20170226_01_T1_B$band.TIF')

with rasterio.open(t.substitute(dict(band=1))) as src:
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    with rasterio.open('result.tif', 'w', **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)
print('sdsd')
# for i in range(1, 12):
#     file_path = Path('data', t.substitute(dict(band=i)))
#     ds = rasterio.open(file_path)
#     arr = ds.read(1)
#     # landcover_window = ds.read(1, window=window, masked=True)
#     # plt.imshow(arr)
#     # plt.show()
#     lon, lat = 60.047639, -124.997849
#     px, py = ds.index(lon, lat)
#     print(ds.bounds)

import rioxarray
import fiona

# open the rasters
# rds1 = rioxarray.open_rasterio("data/LC08_L1TP_024022_20150715_20170226_01_T1_B1.TIF")
rds2 = rioxarray.open_rasterio(file)

# clip the rasters
with fiona.open("data/LC08_L1TP_024022_20150715_20170226_01_T1_B1.TIF", layer='County_Land_Parcels_IGIO_IN_Apr2018') as src:
    geom_crs = src.crs_wkt
    geoms = [feature["geometry"] for feature in src]

# rds1_clipped = rds1.rio.clip(geoms, geom_crs)
# rds2_clipped = rds2.rio.clip(geoms, geom_crs)
#
# # ensure the rasters have the same projection/transform/shape
# rds2_match = rds2_clipped.rio.reproject_match(rds2_clipped)
#
# # write to file
# rds1_clipped.rio.to_raster("LC08_L1TP_021032_20160728_20170221_01_T1_sr_band3__clipped.tif")
# rds2_match.rio.to_raster("CDL_2018_18__clipped_reprojected.tif")