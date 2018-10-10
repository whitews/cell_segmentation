from utils.data import create_generator_from_stash
from keras.models import load_model
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

gen = create_generator_from_stash('model_data/test_numpy', batch_size=200)
x, y = next(gen)

model = load_model('model_data/model_1_plain.hdf5')

pred = model.predict(x)

pred_code = pred.argmax(axis=1)
true_code = y.argmax(axis=1)

df = pd.DataFrame(np.vstack((true_code, pred_code)).T, columns=['true', 'predicted'])

label_lut = json.load(open('model_data/test_numpy/ohe_meta.json'))

for code, label in label_lut.items():
    df.loc[df.true == int(code), 'true_label'] = label
    df.loc[df.predicted == int(code), 'pred_label'] = label

df.to_csv('model_data/results_model1.csv')

true_pred = pred_code == true_code
accuracy = np.sum(true_pred) / true_pred.shape[0]
print(accuracy)

results = pd.crosstab(true_code, pred_code)

checkindex = np.where(np.not_equal(true_code, pred_code))

for i in checkindex[0]:
    yhat = np.argmax(pred[i], axis=-1)
    yhat_lab = label_lut[str(yhat)]
    truth = np.argmax(y[i], axis=-1)
    truth_lab = label_lut[str(truth)]
    plt.title('Truth: {}, Prediction {}'.format(truth_lab, yhat_lab))
    plt.imshow(x[i, :, :, :])
    plt.savefig('classifier_missed' + str(i) + '.png')
