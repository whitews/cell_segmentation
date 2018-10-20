from keras.models import load_model
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import os

mask_opt = 'masked'
color_opt = 'full'
balance_opt = 'unbalanced'

parent_dir = 'model_data'
opt_str = "_".join([mask_opt, color_opt, balance_opt])
test_dir = "_".join(['test', 'numpy', opt_str])
test_path = os.path.join(parent_dir, test_dir)
model_file = "_".join(['model', opt_str]) + '_simple_fm.hdf5'
model_cache_path = os.path.join(parent_dir, model_file)
results_file = "_".join(['results', opt_str]) + '.csv'
results_path = os.path.join(parent_dir, results_file)
misclassified_img_path = os.path.join(parent_dir, 'misclassified')

x = np.load(os.path.join(test_path, 'images.npy'))
x = x.astype(np.float32)
x = x / 255.0
extra = np.load(os.path.join(test_path, 'extra.npy'))
y = np.load(os.path.join(test_path, 'ohelabels.npy'))

model = load_model(model_cache_path)

#pred = model.predict([x, extra])
#pred = model.predict(x)
pred = model.predict(extra)

pred_code = pred.argmax(axis=1)
true_code = y.argmax(axis=1)

df = pd.DataFrame(np.vstack((true_code, pred_code)).T, columns=['true', 'predicted'])

label_lut = json.load(open(os.path.join(test_path, 'ohe_meta.json')))

for code, label in label_lut.items():
    df.loc[df.true == int(code), 'true_label'] = label
    df.loc[df.predicted == int(code), 'pred_label'] = label

df.to_csv(results_path)

true_pred = pred_code == true_code
accuracy = np.sum(true_pred) / true_pred.shape[0]
print(accuracy)

results = pd.crosstab(true_code, pred_code)

checkindex = np.where(np.not_equal(true_code, pred_code))

for i in checkindex[0]:
    if not os.path.exists(misclassified_img_path):
        os.makedirs(misclassified_img_path)

    yhat = np.argmax(pred[i], axis=-1)
    yhat_lab = label_lut[str(yhat)]
    truth = np.argmax(y[i], axis=-1)
    truth_lab = label_lut[str(truth)]
    plt.title('Truth: {}, Prediction {}'.format(truth_lab, yhat_lab))
    plt.imshow(x[i, :, :, :])
    plt.savefig(
        os.path.join(
            misclassified_img_path,
            "_".join(['classifier_missed', opt_str, str(i)]) + '.png'
        )
    )
