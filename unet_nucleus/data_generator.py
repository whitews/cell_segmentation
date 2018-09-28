import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

x = np.load('data/nucleus/xmat.npy')
y = np.load('data/nucleus/ymat.npy')

seed = 820
image_datagen = ImageDataGenerator(
    vertical_flip=True,
    horizontal_flip=True
)
mask_datagen = ImageDataGenerator(
    vertical_flip=True,
    horizontal_flip=True
)
image_generator = image_datagen.flow(x, seed=seed)
mask_generator = mask_datagen.flow(y, seed=seed)

def adjustData(img,mask):
    img = img / 255
    mask = mask / 255
    mask[mask > 0] = 1
    return (img, mask)


def trainGenerator():
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = adjustData(img,mask)
        yield (img, mask)