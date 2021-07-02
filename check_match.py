from string import Template
from pathlib import Path
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
import matplotlib.pyplot as plt
from rasterio.windows import from_bounds
import numpy as np
import cv2
file = "C:\\Users\\Sufian\\Downloads\\CA_forest_VLCE_2015\\CA_forest_VLCE_2015.tif"

from string import Template

t = Template('data/LC08_L1TP_024022_20150715_20170226_01_T1_B$band.TIF')
landsat = t.substitute(dict(band=2))
with rasterio.open(file) as ds:
    with rasterio.open('result.tif') as l_sat:
        dst_crs = ds.crs.data
        lat, lon = 755175, 632651  # -84.709031, 54.508761
        lon, lat = 637153, 748161

        px, py = ds.index(lon, lat)
        print(px, py)
        window = rasterio.windows.Window(px, py, 2000, 2000)

        landcover_window = ds.read(1, window=from_bounds(603459, 739532, 648658, 754736, transform=ds.transform))
        px, py = l_sat.index(lon, lat)
        print(px, py)
        window = rasterio.windows.Window(px, py, 2000, 2000)
        bands = []
        red = np.array(l_sat.read(4, window=from_bounds(603459, 739532, 648658, 754736, transform=l_sat.transform)))
        red = (red - red.min()) / (red.max() - red.min())

        green = np.array(l_sat.read(3, window=from_bounds(603459, 739532, 648658, 754736, transform=l_sat.transform)))
        green = (green - green.min()) / (green.max() - green.min())

        blue = np.array(l_sat.read(2, window=from_bounds(603459, 739532, 648658, 754736, transform=l_sat.transform)))
        blue = (blue - blue.min()) / (blue.max() - blue.min())

        l_sat_window = (np.stack((blue, green, red), axis=2) * 255).astype(np.uint8)

        l_sat_window = cv2.cvtColor(l_sat_window, cv2.COLOR_RGB2YUV)

        # equalize the histogram of the Y channel
        l_sat_window[:, :, 0] = cv2.equalizeHist(l_sat_window[:, :, 0])

        # convert the YUV image back to RGB format
        img_output = cv2.cvtColor(l_sat_window, cv2.COLOR_YUV2RGB)
        # l_sat_window = l_sat.read(2, window=window)
        # west = -84.709031
        # south = 54.508761
        # east = -83.709031
        # north = 55.508761
        # window = from_bounds(west, south, east, north, ds.transform, 500, 500)
        # landcover_window = ds.read(1, window=window)

        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(l_sat_window)
        ax[1].imshow(landcover_window, cmap='nipy_spectral')
        plt.show()
