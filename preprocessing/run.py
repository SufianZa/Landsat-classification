from pathlib import Path
from patches_generator import generate_patches
from image_registration import merge_reprojected_bands, rotate_datasets

# folder of the train/test datasets
TRAIN_DATASETS = 'train'
TEST_DATASETS = 'test'

# reproject each dataset then obtain a list of paths
reprojected_train_datasets = merge_reprojected_bands(TRAIN_DATASETS)
reprojected_test_datasets = merge_reprojected_bands(TEST_DATASETS)
classes = ['water', 'coniferous', 'herbs']

# create patches from the training datasets
for i, dataset in enumerate(reprojected_train_datasets):
    generate_patches(*rotate_datasets(Path(TRAIN_DATASETS, '%s.tif' % dataset)), train_flag=True, data_id=i.__str__(),
                     class_assignment=classes, patches_per_map=1300)

# create patches from the testing datasets
for i, dataset in enumerate(reprojected_test_datasets):
    generate_patches(*rotate_datasets(Path(TEST_DATASETS, '%s.tif' % dataset)), train_flag=False, data_id=i.__str__(),
                     class_assignment=classes, patches_per_map=200)
