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
from skimage.exposure import equalize_adapthist as equalize_hist
# from skimage.exposure import equalize_adapthist, equalize_hist, match_histograms
from string import Template

PADDING_EDGE = 100
t = Template('data/LC08_L1TP_024022_20150715_20170226_01_T1_B$band.TIF')
with rasterio.open(file) as ds:
    with rasterio.open('result.tif') as l_sat:
        west, south, east, north = l_sat.bounds
        # reading a window oo landcover dataset according to landsat boundries
        lc_original = ds.read(1, window=from_bounds(west, south, east, north, transform=ds.transform))

        red = l_sat.read(4)
        m_red = red != 0
        red = (red - np.min(red)) / (np.max(red) - np.min(red))

        green = l_sat.read(3)
        m_green = green != 0
        green = (green - np.min(green)) / (np.max(green) - np.min(green))

        blue = l_sat.read(2)
        m_blue = blue != 0
        blue = (blue - np.min(blue)) / (np.max(blue) - np.min(blue))

        # stacking Multi-spectral image containing -> (Blue, Green, Red, NIR, SWIR 1, SWIR 2)
        ls_original = np.dstack((blue, green, red))

        # extract mask from the bands
        mask = np.mean(np.dstack((m_blue, m_green, m_red)), axis=2)
        mask[mask > 0] = 1
        mask[mask <= 0] = 0

        # calculate the angle to perform the affine transformation of the (rotated) dataset
        coords = np.column_stack(np.where(mask))
        angle = cv.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        w, h = lc_original.shape
        center = (w // 2, h // 2)
        M = cv.getRotationMatrix2D(center, angle, 1.0)

        # perform affine transformation of landcover
        lc_rotated = cv.warpAffine(lc_original, M, (h, w),
                                   flags=cv.INTER_NEAREST,
                                   borderMode=cv.BORDER_CONSTANT)
        # perform affine transformation of landsat
        ls_rotated = cv.warpAffine(ls_original, M, (h, w),
                                   flags=cv.INTER_NEAREST,
                                   borderMode=cv.BORDER_CONSTANT)

        mask = np.mean(ls_rotated, axis=2)
        mask[mask > 0] = 1
        mask[mask <= 0] = 0
        lc_masked = lc_rotated * mask

        # crop
        x, y = np.nonzero(mask)
        ls_cropped = ls_rotated[np.ix_(np.unique(x), np.unique(y))]
        lc_cropped = lc_masked[np.ix_(np.unique(x), np.unique(y))]

        # remove padding on the edges
        ls_cropped = ls_cropped[PADDING_EDGE:-PADDING_EDGE, PADDING_EDGE:-PADDING_EDGE]
        lc_cropped = lc_cropped[PADDING_EDGE:-PADDING_EDGE, PADDING_EDGE:-PADDING_EDGE]

        # enhance colors
        ls_cropped[:, :, 0] = equalize_hist(ls_cropped[:, :, 0])
        ls_cropped[:, :, 1] = equalize_hist(ls_cropped[:, :, 1])
        ls_cropped[:, :, 2] = equalize_hist(ls_cropped[:, :, 2])

        # show steps
        fig, ax = plt.subplots(nrows=2, ncols=4)
        ax[0][0].imshow(ls_original)
        ax[1][0].imshow(lc_original, cmap='nipy_spectral')
        ax[0][0].title.set_text('original')
        ax[0][0].set_axis_off()
        ax[1][0].set_axis_off()

        ax[0][1].imshow(ls_rotated)
        ax[1][1].imshow(lc_rotated, cmap='nipy_spectral')
        ax[0][1].title.set_text('rotated')
        ax[0][1].set_axis_off()
        ax[1][1].set_axis_off()

        ax[0][2].imshow(ls_rotated)
        ax[1][2].imshow(lc_masked, cmap='nipy_spectral')
        ax[0][2].title.set_text('masked')
        ax[0][2].set_axis_off()
        ax[1][2].set_axis_off()

        ax[0][3].imshow(ls_cropped)
        ax[1][3].imshow(lc_cropped, cmap='nipy_spectral')
        ax[0][3].title.set_text('cropped and enhanced')
        ax[0][3].set_axis_off()
        ax[1][3].set_axis_off()
        plt.show()
