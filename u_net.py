from pathlib import Path

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.python.keras.backend import concatenate
from tensorflow.python.keras.layers import Conv2DTranspose
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


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
        concat = concatenate([de_conv_1, conv_2])
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