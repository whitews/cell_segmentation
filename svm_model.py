import numpy as np
import os
from utils import data

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

classifier = SVC(
    kernel='rbf',
    probability=True,
    gamma='scale',
    class_weight='balanced',
    # degree=2
)
pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('classification', classifier)
    ]
)

feature_names = data.include_features

feature_categories = [
    'area_mean',
    'area_variance',
    'contour_count',
    'distance_mean',
    'distance_variance',
    'har_mean',
    'har_variance',
    'largest_contour_area',
    'largest_contour_circularity',
    'largest_contour_convexity',
    'largest_contour_eccentricity',
    'largest_contour_har',
    'largest_contour_saturation_mean',
    'largest_contour_saturation_variance',
    'largest_contour_value_mean',
    'largest_contour_value_variance',
    'percent',
    'perimeter_mean',
    'perimeter_variance',
    'region_area',
    'region_saturation_mean',
    'region_saturation_variance',
    'region_value_mean',
    'region_value_variance'
]

color_categories = [
    'black',
    'white',
    'gray',
    'red',
    'green',
    'blue',
    'violet',
    'yellow',
    'cyan',
    'white_blue'
]

mask_opt = 'masked'
color_opt = 'full'
balance_opt = 'unbalanced'

parent_dir = 'model_data'
opt_str = "_".join([mask_opt, color_opt, balance_opt])
train_dir = "_".join(['train', 'numpy', opt_str])
train_path = os.path.join(parent_dir, train_dir)

test_dir = "_".join(['test', 'numpy', opt_str])
test_path = os.path.join(parent_dir, test_dir)

train_x = np.load(os.path.join(train_path, 'extra.npy'))
train_y = np.load(os.path.join(train_path, 'ohelabels.npy'))
y_ints = [y.argmax() for y in train_y]

val_x = np.load(os.path.join(test_path, 'extra.npy'))
val_y = np.load(os.path.join(test_path, 'ohelabels.npy'))
y_ints_val = [y.argmax() for y in val_y]

pipe.fit(
    train_x,
    y_ints
)

accuracy = pipe.score(val_x, y_ints_val)

print("Baseline all features:", accuracy)

for i, f in enumerate(feature_names):
    i_train_x = train_x[:, i].reshape(train_x.shape[0], 1)

    # train the SVM model with the ground truth
    pipe.fit(
        i_train_x,
        y_ints
    )

    i_val_x = val_x[:, i].reshape(val_x.shape[0], 1)

    accuracy = pipe.score(i_val_x, y_ints_val)

    print("Single out", f, ":", accuracy)

for i, f in enumerate(feature_names):
    i_train_x = np.column_stack((train_x[:, :i], train_x[:, i+1:]))

    # train the SVM model with the ground truth
    pipe.fit(
        i_train_x,
        y_ints
    )

    i_val_x = np.column_stack((val_x[:, :i], val_x[:, i+1:]))

    accuracy = pipe.score(i_val_x, y_ints_val)

    print("Excluding", f, ":", accuracy)

for cat_name in feature_categories:
    feature_indices = []
    other_indices = []
    for i, f in enumerate(feature_names):
        if f.startswith(cat_name):
            feature_indices.append(i)
        else:
            other_indices.append(i)

    # train the SVM model with the ground truth
    pipe.fit(
        train_x[:, feature_indices],
        y_ints
    )

    accuracy = pipe.score(val_x[:, feature_indices], y_ints_val)

    print("Only feature cat", cat_name, ":", accuracy)

    # train the SVM model with the ground truth
    pipe.fit(
        train_x[:, other_indices],
        y_ints
    )

    accuracy = pipe.score(val_x[:, other_indices], y_ints_val)

    print("Excluding feature cat", cat_name, ":", accuracy)

for color_cat in color_categories:
    feature_indices = []
    other_indices = []
    for i, f in enumerate(feature_names):
        if f.endswith(color_cat + ')'):
            feature_indices.append(i)
        else:
            other_indices.append(i)

    # train the SVM model with the ground truth
    pipe.fit(
        train_x[:, feature_indices],
        y_ints
    )

    accuracy = pipe.score(val_x[:, feature_indices], y_ints_val)

    print("Only features with color", color_cat, ":", accuracy)

    # train the SVM model with the ground truth
    pipe.fit(
        train_x[:, other_indices],
        y_ints
    )

    accuracy = pipe.score(val_x[:, other_indices], y_ints_val)

    print("Excluding features with color", color_cat, ":", accuracy)
