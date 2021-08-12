import sys
import pickle
from u_net import UNET

model = UNET(batch_size=16, epochs=300)
model.train()
