import sys
import pickle
from pathlib import Path

from dd import show_learning_curves
from preprocessing.image_registration import rotate_datasets
from u_net import UNET

#
# with open('his.json', 'rb') as f:
#     history = pickle.load(f)
# #
# show_learning_curves(history, 'Land cover segmentation')

model = UNET(batch_size=24, epochs=50)
# model.estimate_raw_landsat(path='muenster2', trim=20)
model.test()
