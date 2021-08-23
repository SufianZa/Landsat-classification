from matplotlib import patches, pyplot as plt

selected_classes = ['no_change', 'water', 'coniferous', 'herbs']

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


if len(selected_classes) == 0:
    selected_classes = list(original_classes.keys())
model_classes = {c: idx for idx, c in enumerate(original_classes) if c in selected_classes}

colors = [(0, 0, 0)] + list(plt.cm.get_cmap('Paired').colors)
colors_legend = [patches.Patch(color=colors[i], label=c) for i, c in enumerate(original_classes) if
                 c in selected_classes]
colors = [colors[i] for i, c in enumerate(original_classes) if c in selected_classes]

REFLECTANCE_MAX_BAND = 65535
PADDING_EDGE = 100

# folder s
LAND_COVER_FILE = "./CA_forest_VLCE_2015/CA_forest_VLCE_2015.tif"
TRAIN_DATASETS = 'train'
TEST_DATASETS = 'test'

SUPPORTED_BANDS = [2, 3, 4, 5, 6, 7]
