from pathlib import Path
from config import selected_classes, TRAIN_DATASETS, TEST_DATASETS
from patches_generator import generate_patches
from image_registration import merge_reprojected_bands, rotate_datasets

# reproject each dataset then obtain a list of paths
reprojected_train_datasets = merge_reprojected_bands(TRAIN_DATASETS)
reprojected_test_datasets = merge_reprojected_bands(TEST_DATASETS)

# create patches from the training datasets
for i, dataset in enumerate(reprojected_train_datasets):
    generate_patches(*rotate_datasets(Path(TRAIN_DATASETS, '%s.tif' % dataset)), train_flag=True, data_id=i.__str__(),
                     class_assignment=selected_classes, patches_per_map=1300)

# create patches from the testing datasets
for i, dataset in enumerate(reprojected_test_datasets):
    generate_patches(*rotate_datasets(Path(TEST_DATASETS, '%s.tif' % dataset)), train_flag=False, data_id=i.__str__(),
                     class_assignment=selected_classes, patches_per_map=200)
