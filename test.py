from pathlib import Path

from u_net import UNET
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model = UNET(batch_size=24, epochs=50)
model.estimate_raw_landsat(path=Path('test', 'LC08_L2SP_035024'), trim=5)
