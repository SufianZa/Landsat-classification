import xarray as xr
import numpy as np
import os
from osgeo import gdal
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import matplotlib.pyplot as plt

file = "C:\\Users\\Sufian\\Downloads\\CA_forest_VLCE_2015\\CA_forest_VLCE_2015.tif"
# ds = gdal.Open(file)
# band = ds.GetRasterBand(1)
# arr = band.ReadAsArray(60000, 60000, 2000 , 5000)
# plt.imshow(arr)
# plt.show()
# print('Sd')

import rasterio
import rasterio.features
from rasterio.plot import show, show_hist
import rasterio.warp
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from datetime import datetime, timedelta

date = datetime(2020, 8, 1)
# lon=-124.997849 lat=62.997243
# lon=-103.298564 lat=62.997243
# lon=-103.298564 lat=60.047639
# lon=-124.997849 lat=60.047639

# request API
# api = SentinelAPI('szaabalawi', 'CH5E@8_ukpr34RJ')
# REQUEST_AREA = 'POLYGON ((-126.675537 59.911194,-122.220840 60.327511,-121.731979 58.711685,-125.979378 58.303020,-126.675537 59.911194))'
# products = api.query(
#     area=REQUEST_AREA,
#     date=('20150801', '20150807'))
#
# pid = list(products.keys())[0]
# print(pid)
# api.download(pid)

with rasterio.open(file) as ds:
    # Use pyproj to convert point coordinates
    lon, lat = 60.047639, -124.997849
    px, py = ds.index(lon, lat)
    print(px, py)
    lon2, lat2 = 63.047639, -40.997849
    px2, py2 = ds.index(lon2, lat2)
    print(px2, py2)
    window = rasterio.windows.Window(px, py, 50,50)
    # west = -150.675537
    # south = 55.911194
    # east = -121.731979
    # north = 60.327511
    # print(ds.index(west, north))
    # print(ds.index(east, south))
    # window = from_bounds(west, south, east, north, ds.transform, 500, 500)
    landcover_window = ds.read(1, window=window)
    # landcover_window = ds.read(1, window=window, masked=True)
    plt.imshow(landcover_window)
    plt.show()
