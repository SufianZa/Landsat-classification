from pathlib import Path
from patches_generator import generate_patches
from image_registration import merge_reprojected_bands, rotate_dataset

# folder of the train/test datasets
TRAIN_DATASETS = 'train'
TEST_DATASETS = 'test'

# reproject each dataset then obtain a list of paths
reprojected_train_datasets = merge_reprojected_bands(TRAIN_DATASETS)
reprojected_test_datasets = merge_reprojected_bands(TEST_DATASETS)

# create patches from the training datasets
for i, dataset in enumerate(reprojected_train_datasets):
    classes = ['water']
    generate_patches(*rotate_dataset(Path(TRAIN_DATASETS,'%s.tif' % dataset)), train_flag=True, data_id=i.__str__(),
                     class_assignment=classes, patches_per_map=1500)

# create patches from the testing datasets
for i, dataset in enumerate(reprojected_test_datasets):
    classes = ['water']
    generate_patches(*rotate_dataset(Path(TEST_DATASETS,'%s.tif' % dataset)), train_flag=False, data_id=i.__str__(),
                     class_assignment=classes, patches_per_map=1500)
