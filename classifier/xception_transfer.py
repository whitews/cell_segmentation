from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
import argparse
import numpy as np
from utils.data import create_generator_from_stash
import matplotlib.pyplot as plt
from classifier.architecture import build_model_double, build_model, weighted_categorical_crossentropy, build_model_scratch

gen = create_generator_from_stash('data/train_numpy')

from keras import backend as K


checkpoint = ModelCheckpoint(
    'classifier/model_1.hdf5',
    monitor='loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    period=1
)

model = build_model_scratch()

h = model.fit_generator(
    gen,
    steps_per_epoch=70,
    epochs=10,
    callbacks=[checkpoint]
)

print(h.history.keys())
# summarize history for accuracy
plt.plot(h.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(h.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model2 = build_model_double('classifier/model_1.hdf5')

checkpoint2 = ModelCheckpoint(
    'classifier/model_2.hdf5',
    monitor='acc',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    period=1
)

h2 = model.fit_generator(
    gen,
    steps_per_epoch=70,
    epochs=10,
    callbacks=[checkpoint2]
)
plt.plot(h2.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(h2.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()