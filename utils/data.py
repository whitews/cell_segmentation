import os
import json
from PIL import Image
# weird import style to un-confuse PyCharm
try:
    from cv2 import cv2
except ImportError:
    import cv2
from operator import itemgetter
import numpy as np
from random import randint

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from keras.preprocessing.image import ImageDataGenerator
import skimage.transform as trans
from collections import Counter
import itertools


def create_generator_from_stash(stash_dir, batch_size=12, demo_gen=False):
    x = np.load(os.path.join(stash_dir, 'images.npy'))
    y = np.load(os.path.join(stash_dir, 'ohelabels.npy'))
    image_datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True
    )

    if demo_gen:
        index = randint(0, x.shape[1])
        x = np.expand_dims(x[index, :], axis=0)
        y = np.expand_dims(y[index, :], axis=0)
    image_datagen.fit(x)
    return image_datagen.flow(x, y, batch_size=batch_size)


def clean_and_stash_numpys(
        imglist,
        strlist,
        numpy_save_dir,
        imgshape=(299, 299, 3)
):
    """
    Take a list of numpy arrays and a list of text and return one generator
    """
    new_mask = np.zeros((len(imglist),) + imgshape)
    for i, si in enumerate(imglist):
        img = trans.resize(si, imgshape)
        new_mask[i, :, :, :] = img
    le = LabelEncoder()
    strlist_le = le.fit_transform(strlist)
    ohe = OneHotEncoder(sparse=False)
    strlist_ohe = ohe.fit_transform(strlist_le.reshape(-1, 1))
    ohe_meta = np.column_stack((ohe.categories_[0].astype(np.int), le.classes_))
    np.savetxt(
        os.path.join(numpy_save_dir, 'ohemeta.txt'),
        ohe_meta,
        delimiter=" ",
        fmt="%s"
    )
    np.save(os.path.join(numpy_save_dir, 'images.npy'), new_mask)
    np.save(os.path.join(numpy_save_dir, 'ohelabels.npy'), strlist_ohe)
    print('Numpys stashed!')


def get_imageset_in_memory(
        image_set_directory,
        test_image_name,
        balance_train=True,
        balance_test=False
):
    """
    Takes an image set directory and a test image file name.
    Returns 4 numpy arrays: train_regions, train_labels, test_regions, test_labels
    """
    image_set_metadata = get_training_data_for_image_set(image_set_directory)
    test_metadata = image_set_metadata.pop(test_image_name)

    images_train = []
    labels_train = []
    images_test = []
    labels_test = []

    for key, value in image_set_metadata.items():
        hsv_img = value['hsv_img']
        rgb_img = cv2.cvtColor(
            hsv_img,
            cv2.COLOR_HSV2RGB
        )
        for region in value['regions']:
            labels_train.append(region['label'])
            images_train.append(
                extract_contour_bounding_box(rgb_img, region['points'])
            )

    hsv_img = test_metadata['hsv_img']
    rgb_img = cv2.cvtColor(
        hsv_img,
        cv2.COLOR_HSV2RGB
    )
    for region in test_metadata['regions']:
        labels_test.append(region['label'])
        images_test.append(
            extract_contour_bounding_box(rgb_img, region['points'])
        )
    if balance_train:
        images_train, labels_train = make_balanced(images_train, labels_train)
    if balance_test:
        images_test, labels_test = make_balanced(images_test, labels_test)
    assert len(images_train) == len(labels_train)
    return images_train, labels_train, images_test, labels_test


def make_balanced(xarray, ylist):
    counts = Counter(ylist)
    bringup = np.max(list(counts.values()))
    bringupdict = {key: bringup//value for key, value in counts.items()}
    newx = []
    newy = []
    for x in set(ylist):
        indices = [i for i, val in enumerate(ylist) if val == x]
        newindices = list(itertools.repeat(indices, bringupdict[x]))
        newindicesflat = [item for sublist in newindices for item in sublist]
        for index in newindicesflat:
            newx.append(xarray[index])
            newy.append(ylist[index])
    return newx, newy


def get_training_data_for_image_set(image_set_dir):
    # Each image set directory will have a 'regions.json' file. This regions
    # file has keys of the image file names in the image set, and the value
    # for each image is a list of segmented polygon regions.
    # First, we will read in this file and get the file names for our images
    regions_file = open(os.path.join(image_set_dir, 'regions.json'))
    regions_json = json.load(regions_file)
    regions_file.close()

    # output will be a dictionary of training data, where the
    # polygon points dict is a numpy array.
    # The keys will still be the image names
    training_data = {}

    for image_name, sub_regions in regions_json.items():
        # noinspection PyUnresolvedReferences
        tmp_image = Image.open(os.path.join(image_set_dir, image_name))
        tmp_image = np.asarray(tmp_image)

        # noinspection PyUnresolvedReferences
        tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_RGB2HSV)

        training_data[image_name] = {
            'hsv_img': tmp_image,
            'regions': []
        }

        for region in sub_regions:
            points = np.empty((0, 2), dtype='int')

            for point in sorted(region['points'], key=itemgetter('order')):
                points = np.append(points, [[point['x'], point['y']]], axis=0)

            training_data[image_name]['regions'].append(
                {
                    'label': region['anatomy'],
                    'points': points
                }
            )

    return training_data


def compute_bbox(contour):
    # noinspection PyUnresolvedReferences
    x1, y1, w, h = cv2.boundingRect(contour)

    return [x1, y1, x1 + w, y1 + h]


def make_binary_mask(contour, img_dims):
    mask = np.zeros(img_dims, dtype=np.uint8)
    # noinspection PyUnresolvedReferences
    cv2.drawContours(
        mask,
        [contour],
        0,
        1,
        cv2.FILLED
    )

    # return boolean array
    return mask


def extract_contour_bounding_box(image, contour):
    x, y, x2, y2 = compute_bbox(contour)
    sub = image[y:y2, x:x2]
    return sub


def extract_contour_bounding_box_masked(image, contour):
    zz = make_binary_mask(contour, (image.shape[0], image.shape[1]))
    zzz = cv2.bitwise_and(image, image, mask=zz)
    x, y, x2, y2 = compute_bbox(contour)
    sub = zzz[y:y2, x:x2]
    return sub


def extract_contour_masked(image, contour):
    zz = make_binary_mask(contour, (image.shape[0], image.shape[1]))
    zzz = cv2.bitwise_and(image, image, mask=zz)
    sub = zzz
    return sub
