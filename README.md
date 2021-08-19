# TB17-landsat ML model for land cover classification 

An Instant Segmentation model to classify different landcover classes using raw satellite imagery.


### Datasets

Input:  of Landsat 8 images Level 2 collection 2 using https://earthexplorer.usgs.gov/ web interface. \
The Multi-spectral Image consists of Blue, Green, Red, NIR, SWIR 1 and SWIR 2 corresponding to bands numbers (2, 3, 4, 5, 6, 7), respectively.


Label: Landcover 2015 from [National Forest Information System](https://www.google.com/url?q=https://opendata.nfis.org/mapserver/nfis-change_eng.html&sa=D&source=editors&ust=1629370818190000&usg=AOvVaw2KuSW6Mkyf05elJAda43O7)
\
The following classes were included in label data:
- no_change
- water 
- snow_ice 
- rock_rubble 
- exposed_barren_land 
- bryoids 
- shrubland 
- wetland 
- wetlandtreed 
- herbs 
- coniferous 
- broadleaf 
- mixedwood 


### Preprocessing

The preparation of the train data consists of extracting pairs of input und output of the train and label data. This requires the datasets to be projected in the same spatial reference. Therefore, the landsat images were reprojected to match the same spatial reference of landcover dataset. After Datasets-registration patches with fixed size were extracted to prepare the train and label data.

<img alt="Preprocessing" align="middle" src="./img/preprocessing.png"/>

### Training
The model has u-net architecture consisting of 5 convolution and deconvolution layers. The model is trained to classify 4 different classes (water, herbs, coniferous and other) using the dice coefficient to evaluate accuracy.
The model has reached total accuracy of 89% after learning for 120 epochs.

### Testing or using the model

After the model loads the weights it can estimate raw bands images of landsat 8 using ```model.estimate_raw_landsat(path)``` as demonstrated in test.py. \
The raw landsat bands should be in one folder named as their originial _Landsat Product Identifier L2_ followed by the SR\_B<band\_number>.TIF (e.g. LC08\_L2SP\_196024\_20210330\_20210409\_02\_T1\_SR\_B4.TIF is band 4 of the landsat product LC08\_L2SP\_196024\_20210330\_20210409\_02\_T1) \

The result ```classified_landcover.tiff``` is saved as a geo-referenced 1 band GeoTiff in the same folder.
