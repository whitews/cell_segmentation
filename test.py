from utils.data import create_generator_from_stash
from keras.models import load_model
from classifier.architecture import weighted_categorical_crossentropy
import numpy as np
import pandas as pd

gen = create_generator_from_stash('model_data/test_numpy', batch_size=200)
x, y = next(gen)

model = load_model(
    'model_data/model_1.hdf5',
    custom_objects={'loss':  weighted_categorical_crossentropy((1, 6, 6, 13, 13))}
)

pred = model.predict(x)

pred_label = pred.argmax(axis=1)
true_label = y.argmax(axis=1)

df = pd.DataFrame(np.vstack((true_label, pred_label)).T, columns=['true', 'predicted'])
df.to_csv('model_data/results.csv')

true_pred = pred_label == true_label
accuracy = np.sum(true_pred) / true_pred.shape[0]
print (accuracy)
