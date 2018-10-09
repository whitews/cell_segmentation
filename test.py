from utils.data import create_generator_from_stash
from keras.models import load_model
from classifier.architecture import weighted_categorical_crossentropy
import numpy as np
import pandas as pd
import json

gen = create_generator_from_stash('model_data/test_numpy', batch_size=200)
x, y = next(gen)

model = load_model(
    'model_data/model_2.hdf5',
    custom_objects={'loss':  weighted_categorical_crossentropy((1, 6, 6, 13, 13))}
)

pred = model.predict(x)

pred_code = pred.argmax(axis=1)
true_code = y.argmax(axis=1)

df = pd.DataFrame(np.vstack((true_code, pred_code)).T, columns=['true', 'predicted'])

label_lut = json.load(open('model_data/test_numpy/ohe_meta.json'))

for code, label in label_lut.items():
    df.loc[df.true == int(code), 'true_label'] = label
    df.loc[df.predicted == int(code), 'pred_label'] = label

df.to_csv('model_data/results.csv')

true_pred = pred_code == true_code
accuracy = np.sum(true_pred) / true_pred.shape[0]
print(accuracy)
