from utils.data import get_training_data_for_image_set, get_imageset_in_memory, clean_and_stash_numpys, create_generator_from_stash
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
from classifier.architecture import build_model_double, build_model, weighted_categorical_crossentropy

gen = create_generator_from_stash('data/test_numpy', batch_size=200)
x,y = next(gen)

model = load_model('classifier/model_1.hdf5',
                   custom_objects = {'loss':  weighted_categorical_crossentropy((1, 6, 6, 13, 13))})

pred = model.predict(x)


gen = create_generator_from_stash('data/test_numpy', batch_size=400)
x,y = next(gen)


pred2 = model.predict(x)