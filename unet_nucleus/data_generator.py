import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

x = np.load('data/nucleus/xmat.npy')
y = np.load('data/nucleus/ymat.npy')

image_datagen = ImageDataGenerator(
    vertical_flip=True,
    horizontal_flip=True
)
mask_datagen = ImageDataGenerator(
    vertical_flip=True,
    horizontal_flip=True
)
image_generator = image_datagen.flow(x)
mask_generator = mask_datagen.flow(y)

train_generator = zip(image_generator, mask_generator)