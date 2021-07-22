import os
import pickle
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt, patches
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.python.keras.backend import concatenate
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers import Conv2DTranspose
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

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


class UNET:
    def __init__(self, batch_size=64, epochs=30, window_size=256):
        self.batch_size = batch_size
        self.window_size = window_size
        self.epochs = epochs
        self.weight_file = str(Path('best_weight.hdf5'))
        self.model = self.init_network((window_size, window_size, 6))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def init_network(self, input_size):
        """
        This method initiates the u-network which takes the input_size as initial size of the Input layer
        """
        inputs = Input(input_size)
        conv_1 = Conv2D(32, (3, 3), padding="same", strides=1, activation="relu")(inputs)
        conv_1 = Conv2D(32, (3, 3), padding="same", strides=1, activation="relu")(conv_1)
        pool = MaxPooling2D((2, 2))(conv_1)
        conv_2 = Conv2D(64, (3, 3), padding="same", strides=1, activation="relu")(pool)
        conv_2 = Conv2D(64, (3, 3), padding="same", strides=1, activation="relu")(conv_2)
        pool = MaxPooling2D((2, 2))(conv_2)
        conv_3 = Conv2D(128, (3, 3), padding="same", strides=1, activation="relu")(pool)
        conv_3 = Conv2D(128, (3, 3), padding="same", strides=1, activation="relu")(conv_3)
        pool = MaxPooling2D((2, 2))(conv_3)
        bn = Conv2D(256, (3, 3), padding="same", strides=1, activation="relu")(pool)
        bn = Conv2D(256, (3, 3), padding="same", strides=1, activation="relu")(bn)
        de_conv_3 = Conv2DTranspose(128, (3, 3), strides=2, padding='same')(bn)
        concat = concatenate([de_conv_3, conv_3])
        de_conv_3 = Conv2D(128, (3, 3), padding="same", activation="relu")(concat)
        de_conv_3 = Conv2D(128, (3, 3), padding="same", activation="relu")(de_conv_3)
        de_conv_2 = Conv2DTranspose(64, (3, 3), strides=2, padding='same')(de_conv_3)
        concat = concatenate([de_conv_2, conv_2])
        de_conv_2 = Conv2D(64, (3, 3), padding="same", activation="relu")(concat)
        de_conv_2 = Conv2D(64, (3, 3), padding="same", activation="relu")(de_conv_2)
        de_conv_1 = Conv2DTranspose(32, (3, 3), strides=2, padding='same')(de_conv_2)
        concat = concatenate([de_conv_1, conv_1])
        de_conv_1 = Conv2D(32, (3, 3), padding="same", activation="relu")(concat)
        de_conv_1 = Conv2D(32, (3, 3), padding="same", activation="relu")(de_conv_1)
        outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(de_conv_1)
        return Model(inputs=inputs, outputs=[outputs])

    def multi_spectral_image_generator(self, mode='train'):
        """
        This method provides pairs of input RGB images and NIR images and labels as generators
        and can be used for train as well as validation data
        :param mode: str
                can be set for "train" or "validation" data
        """
        # same seed should be in all generators
        SEED = 213
        data_gen_args = dict(rescale=1. / 255)
        # apply augmentation only for train-data
        if mode == 'train':
            data_gen_args = dict(
                rescale=1. / 255,
                horizontal_flip=True,
                vertical_flip=True
            )

        X_train_RGB = ImageDataGenerator(**data_gen_args).flow_from_directory(
            str(Path('dataset', mode, 'RGBinputs', 'input').parent), batch_size=self.batch_size, color_mode='rgb',
            seed=SEED)

        X_train_NIR = ImageDataGenerator(**data_gen_args).flow_from_directory(
            str(Path('dataset', mode, 'NIRinputs', 'input').parent), batch_size=self.batch_size, color_mode='rgb',
            seed=SEED)

        # don't rescale masks
        del data_gen_args['rescale']

        y_train = ImageDataGenerator(**data_gen_args).flow_from_directory(
            str(Path('dataset', mode, 'labels', 'label').parent), batch_size=self.batch_size,
            class_mode='input', color_mode='grayscale',
            seed=SEED)

        while True:
            yield np.concatenate((next(X_train_RGB)[0], next(X_train_NIR)[0]), axis=3), next(y_train)[0]

    def train(self):
        checkpoint = ModelCheckpoint(self.weight_file, verbose=1, monitor='val_loss', save_best_only=True, mode='min')
        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=0,
                                   patience=3,
                                   verbose=0, mode='auto')

        train_gen = self.multi_spectral_image_generator('train')
        val_gen = self.multi_spectral_image_generator('validation')

        _, _, num_of_train = next(os.walk(str(Path('dataset', 'train', 'RGBinputs', 'input'))))
        _, _, num_of_val = next(os.walk(str(Path('dataset', 'validation', 'RGBinputs', 'input'))))

        print('Start training with %d images and %d images for validation' % (len(num_of_train), len(num_of_val)))
        self.history = self.model.fit(train_gen,
                                      steps_per_epoch=len(num_of_train) // self.batch_size,
                                      epochs=self.epochs,
                                      validation_steps=len(num_of_val) // self.batch_size,
                                      validation_data=val_gen,
                                      callbacks=[checkpoint, early_stop])

        with open('history.json', 'wb') as file_pi:
            pickle.dump(self.history.history, file_pi)

    def test(self):
        """
            Tests the model on the test images in the pre-defined paths in global variables
            then plots a comparison of the prediction and ground truth patches
        """
        x = []
        y = []
        colors = [(0, 0, 0)] + list(plt.cm.get_cmap('Paired').colors)
        colors_legend = [patches.Patch(color=colors[i], label=classes_names[i]) for i in range(len(colors))]

        for img_path in list(Path('dataset/test/RGBinputs/input').glob('*.*'))[::10]:
            name = os.path.basename(img_path)
            img_rgb = np.array(Image.open(img_path))
            img_nir = np.array(Image.open(str(img_path).replace('RGBinputs', 'NIRinputs')))
            mask = np.array(
                Image.open(os.path.join(Path('dataset/test/labels/label'), name)))  # read image
            img = np.dstack((img_rgb, img_nir)) * 1.0 / 255
            x.append(img)
            y.append(mask)
        x = np.array(x)
        y = np.array(y)
        self.model.load_weights(self.weight_file)
        output = np.squeeze(self.model.predict(x, verbose=0))
        n_rows = 4
        for i in np.arange(0, output.shape[0], n_rows):
            fig, ax = plt.subplots(n_rows, 3)
            for row in range(n_rows):
                inp = x[i+row, :, :, :3]
                pre = output[i+row, ...]
                pre = (pre - np.amin(pre)) * 1.0 / (np.amax(pre) - np.amin(pre))
                ori = y[i+row, ...]
                pre[pre > 0.7] = 1
                pre[pre <= 0.7] = 0

                fig.suptitle('Estimation {}'.format(i))
                ax[row][0].imshow(inp)
                ax[row][1].imshow(np.array([[colors[int(val)]] for val in pre.reshape(-1)]).reshape(*pre.shape, 3))
                ax[row][2].imshow(np.array([[colors[int(val)]] for val in ori.reshape(-1)]).reshape(*ori.shape, 3))

                ax[row][0].title.set_text('input')
                ax[row][1].title.set_text('Estimated')
                ax[row][2].title.set_text('Ground truth')

                ax[row][0].axis('off')
                ax[row][1].axis('off')
                ax[row][2].axis('off')

            plt.legend(handles=colors_legend, borderaxespad=-15, fontsize='x-small')
            plt.show()
