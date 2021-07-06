from string import Template
from pathlib import Path
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

file = "CA_forest_VLCE_2015\\CA_forest_VLCE_2015.tif"

from string import Template

t = Template('ndata/LC08_L2SP_042021_20150830_20200908_02_T1/LC08_L2SP_042021_20150830_20200908_02_T1_SR_B$band.TIF')

with rasterio.open(file) as ds:
    dst_src = ds.crs

    with rasterio.open(t.substitute(dict(band=2)), read='r+') as band:
        transform, width, height = calculate_default_transform(
            band.crs, dst_src, band.width, band.height, *band.bounds, resolution=ds.res)
        kwargs = band.meta.copy()
        kwargs.update({
            'crs': dst_src,
            'transform': transform,
            'width': width,
            'height': height,
            'count': 7
        })

with rasterio.open('result.tif', 'w', **kwargs) as dst:
    for i in range(1, 8):
        print('Band %s' % i)
        with rasterio.open(t.substitute(dict(band=i))) as band:
                reproject(
                    source=rasterio.band(band, 1),
                    destination=rasterio.band(dst, i),
                    src_transform=band.transform,
                    src_crs=band.crs,
                    dst_transform=transform,
                    dst_crs=dst_src,
                    resampling=Resampling.nearest)