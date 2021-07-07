import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import distance_transform_edt
import os
import numpy as np
from sklearn.model_selection import train_test_split
from image_registration import image_registration

original_classes = dict(no_change=0,
                        water=20,
                        snow_ice=31,
                        rock_rubble=32,
                        exposed_barren_land=33,
                        bryoids=40,
                        shrubland=50,
                        wetland=80,
                        wetlandtreed=81,
                        herbs=100,
                        coniferous=210,
                        broadleaf=220,
                        mixedwood=230)
classes_names = list(original_classes.keys())
model_classes = {c: idx for idx, c in enumerate(classes_names)}


def generate_patches(train_image, label_image, train=True, bands=None,
                     class_assignment=None, data_id='',
                     patch_size=256,
                     patches_per_map=15):
    # get all classes if no specific were given
    if class_assignment is None:
        class_assignment = classes_names

    # calculate most important indices
    sampling_weights = label_image[patch_size // 2:-patch_size // 2, patch_size // 2:-patch_size // 2].astype(np.float)
    linear = np.cumsum(sampling_weights)
    linear /= linear[-1]
    indices = np.searchsorted(linear, np.random.random_sample(patches_per_map), side='right')

    if train:
        # splitting the image into patches 80% train,15% validation, %5 test
        train, validation_test = train_test_split(indices, test_size=0.20, shuffle=True)
        validation, test = train_test_split(validation_test, test_size=0.25, shuffle=True)
        data = dict(train=train, validation=validation, test=test)
    else:
        data = dict(test=indices)

    for subset_name, subset in data.items():
        for i, idx in enumerate(subset):
            x = idx % sampling_weights.shape[1]
            y = idx // sampling_weights.shape[1]
            input_patch = train_image[y:y + patch_size, x:x + patch_size, :]
            label_patch = label_image[y:y + patch_size, x:x + patch_size]
            label_patch_converted = np.zeros_like(label_patch)

            # create categorical mask
            for c in class_assignment:
                label_patch_converted[label_patch == original_classes[c]] = model_classes[c]

            # create directories
            Path('dataset', subset_name, 'inputs', 'input').mkdir(parents=True, exist_ok=True)
            Path('dataset', subset_name, 'labels', 'label').mkdir(parents=True, exist_ok=True)

            # save patches
            Image.fromarray((input_patch * 255).astype(np.uint8)).save(
                os.path.join('dataset/{0}/{1}s/{1}'.format(subset_name, 'input'),
                             "{}img-{}.tiff".format(data_id, i)))
            Image.fromarray(label_patch_converted.astype(np.uint8)).save(
                os.path.join('dataset/{0}/{1}s/{1}'.format(subset_name, 'label'),
                             "{}img-{}.tiff".format(data_id, i)))


if __name__ == '__main__':
    generate_patches(*image_registration('result.tif'), train=True,
                     class_assignment=['water', 'wetland', 'shrubland', 'wetlandtreed'], patches_per_map=2500) 
