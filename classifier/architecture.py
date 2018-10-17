from keras import layers
from keras.models import Model, Input
from keras.applications.xception import Xception
from keras.preprocessing.image import ImageDataGenerator
from random import randint
import numpy as np
import os


def create_generator_from_stash(
        stash_dir,
        batch_size=12,
        demo_gen=False,
        include_extra=False
):
    x = np.load(os.path.join(stash_dir, 'images.npy'))
    y = np.load(os.path.join(stash_dir, 'ohelabels.npy'))
    image_datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        dtype=np.uint8,
        rotation_range=90,
        zoom_range=0.1,
        rescale=1. / 255.0
    )

    if demo_gen:
        index = randint(0, x.shape[1])
        x = np.expand_dims(x[index, :], axis=0)
        y = np.expand_dims(y[index, :], axis=0)
    image_datagen.fit(x)

    if include_extra:
        extra_x = np.load(os.path.join(stash_dir, 'extra.npy'))
        x = [x, extra_x]
    return image_datagen.flow(x, y, batch_size=batch_size)


def build_simple_fm_model(class_count, feature_count):
    feature_input = Input(shape=(feature_count,))
    x = layers.Dense(96)(feature_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(12)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    # x = layers.Dense(24, activation='softmax')(x)
    # x = layers.Dropout(0.1)(x)
    predictions = layers.Dense(class_count, activation='softmax')(x)
    m = Model(inputs=feature_input, outputs=predictions)
    m.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return m


def build_xception_model(class_count):
    base_model = Xception(
        weights=None,
        include_top=False,
        input_shape=(99, 99, 3)
    )
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    predictions = layers.Dense(class_count, activation='softmax')(x)

    m = Model(inputs=base_model.input, outputs=predictions)
    m.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return m


def build_cnn_model(class_count, extra_feature_count=0):
    img_input = Input(shape=(99, 99, 3))

    # 1st Conv layer
    x = layers.Conv2D(
        32, (7, 7),
        padding='same',
        use_bias=False)(img_input)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # 2nd Conv layer with BN
    # x = layers.Conv2D(
    #     32, (5, 5),
    #     border_mode='same',
    #     use_bias=False)(x)
    # x = layers.Activation('relu')(x)
    # # x = layers.BatchNormalization()(x)
    # x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    # x = layers.Dropout(0.1)(x)

    # pool & flatten Conv outputs
    x = layers.Flatten()(x)

    # add fully-connected layer
    extra_input = Input(shape=(extra_feature_count,))
    x = layers.Concatenate()([x, extra_input])
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(12)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)

    predictions = layers.Dense(class_count, activation='softmax')(x)
    m = Model(inputs=[img_input, extra_input], outputs=predictions)
    m.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return m
