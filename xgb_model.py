import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb


mask_opt = 'masked'
color_opt = 'full'
balance_opt = 'unbalanced'

parent_dir = 'model_data'
opt_str = "_".join([mask_opt, color_opt, balance_opt])
train_dir = "_".join(['train', 'numpy', opt_str])
train_path = os.path.join(parent_dir, train_dir)

test_dir = "_".join(['test', 'numpy', opt_str])
test_path = os.path.join(parent_dir, test_dir)

label_lut = json.load(open(os.path.join(test_path, 'ohe_meta.json')))
misclassified_img_path = os.path.join(parent_dir, 'misclassified')

x = np.load(os.path.join(test_path, 'images.npy'))
x = x.astype(np.float32)
x = x / 255.0

train_x = np.load(os.path.join(train_path, 'extra.npy'))
train_y_ohe = np.load(os.path.join(train_path, 'ohelabels.npy'))
train_y_codes = [y.argmax() for y in train_y_ohe]

val_x = np.load(os.path.join(test_path, 'extra.npy'))
val_y_ohe = np.load(os.path.join(test_path, 'ohelabels.npy'))
val_y_codes = np.array([y.argmax() for y in val_y_ohe])

d_train = xgb.DMatrix(train_x, label=train_y_codes)
d_test = xgb.DMatrix(val_x, label=val_y_codes)

param = {
    'silent': 1,
    'max_depth': 6,
    'eta': 0.2,
    'min_child_weight': 0.1,
    'objective': 'multi:softprob',
    'num_class': 6
}

num_round = 100
bst = xgb.train(param, d_train, num_round)

predictions = bst.predict(d_test)
pred_codes = np.asarray([np.argmax(line) for line in predictions])

true_predictions = pred_codes == val_y_codes
accuracy = np.sum(true_predictions) / true_predictions.shape[0]
print(accuracy)

results = pd.crosstab(val_y_codes, pred_codes)

checkindex = np.where(np.not_equal(val_y_codes, pred_codes))

for i in checkindex[0]:
    if not os.path.exists(misclassified_img_path):
        os.makedirs(misclassified_img_path)

    yhat = np.argmax(predictions[i], axis=-1)
    yhat_lab = label_lut[str(yhat)]
    truth = np.argmax(val_y_ohe[i], axis=-1)
    truth_lab = label_lut[str(truth)]
    plt.title('Truth: {}, Prediction {}'.format(truth_lab, yhat_lab))
    plt.imshow(x[i, :, :, :])
    plt.savefig(
        os.path.join(
            misclassified_img_path,
            "_".join(['classifier_missed_xgb', opt_str, str(i)]) + '.png'
        )
    )
