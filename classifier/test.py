from utils.data import create_generator_from_stash
from keras.models import load_model
from classifier.architecture import weighted_categorical_crossentropy

gen = create_generator_from_stash('data/test_numpy', batch_size=200)
x, _ = next(gen)

model = load_model(
    'classifier/model_1.hdf5',
    custom_objects={'loss':  weighted_categorical_crossentropy((1, 6, 6, 13, 13))}
)

pred = model.predict(x)

gen = create_generator_from_stash('data/test_numpy', batch_size=400)
x, _ = next(gen)

pred2 = model.predict(x)
