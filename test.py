import sys
import pickle
from u_net import UNET

model = UNET(batch_size=16, epochs=50)
model.estimate_raw_landsat(path='test/LC08_L2SP_016021', trim=50)
