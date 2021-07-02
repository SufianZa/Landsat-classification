from string import Template
from pathlib import Path
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
import matplotlib.pyplot as plt
from rasterio.windows import from_bounds
import numpy as np
import cv2 as cv

file = "CA_forest_VLCE_2015\\CA_forest_VLCE_2015.tif"
from skimage.exposure import equalize_hist
from string import Template

t = Template('data/LC08_L1TP_024022_20150715_20170226_01_T1_B$band.TIF')

with rasterio.open(file) as ds:
    with rasterio.open('result.tif') as l_sat:
        dst_crs = ds.crs.data
        #### TODO
        x = l_sat.read(2, masked=True)
        coords = np.column_stack(np.where(x > 0))
        angle = cv.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle

        # rotated = cv.warpAffine(x, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
        #### TOREMOVE
        west, north = l_sat.transform * (l_sat.width // 2, l_sat.height // 2)
        east, south = l_sat.transform * ((l_sat.width // 2) + 500, (l_sat.height // 2) + 500)

        west, north = l_sat.transform * (2000, 2000)
        east, south = l_sat.transform * (3000, 3000)
        # west, south, east, north = 603459, 739532, 648658, 754736
        print(west, south, east, north)
        landcover_window = ds.read(1, window=from_bounds(west, south, east, north, transform=ds.transform))
        window = from_bounds(west, south, east, north, transform=l_sat.transform)
        n = l_sat.read_masks(2)
        red = equalize_hist(l_sat.read(4, window=window))
        red = (red - np.min(red)) / (np.max(red) - np.min(red))

        green = equalize_hist(l_sat.read(3, window=window))
        green = (green - np.min(green)) / (np.max(green) - np.min(green))

        blue = equalize_hist(l_sat.read(2, window=window))
        blue = (blue - np.min(blue)) / (np.max(blue) - np.min(blue))

        l_sat_window = np.dstack((blue, green, red))
        w, h, _ = l_sat_window.shape
        center = (w // 2, h // 2)
        M = cv.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv.warpAffine(l_sat_window, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_DEFAULT)

        w, h = landcover_window.shape
        center = (w // 2, h // 2)
        M = cv.getRotationMatrix2D(center, angle, 1.0)

        landcover_window_rotated = cv.warpAffine(landcover_window, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_DEFAULT)

        fig, ax = plt.subplots(nrows=2, ncols=2)
        ax[0][0].imshow(l_sat_window)
        ax[1][0].imshow(landcover_window, cmap='nipy_spectral')

        ax[0][1].imshow(rotated)
        ax[1][1].imshow(landcover_window_rotated, cmap='nipy_spectral')
        plt.show()
