import sys
import pickle
from u_net import UNET

model = UNET(batch_size=24, epochs=100)
model.train()
