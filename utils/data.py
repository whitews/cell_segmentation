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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import skimage.transform as trans
from collections import Counter
import itertools
from lung_map_utils import utils as lm_utils


rgb_color_lut = {
    'white': (255, 255, 255),
    'red': (223, 7, 7),
    'gray': (63, 63, 63),
    'blue': (7, 7, 255),
    'yellow': (255, 255, 0),
    'green': (0, 223, 0),
    'cyan': (0, 255, 255),
    'violet': (255, 0, 255),
    'black': (0, 0, 0)
}

rgb2_color_lut = {
    'dark_blue': (0, 0, 160),
    'white_blue': (160, 160, 255)
}

include_features = [
    'region_area',
    'region_saturation_mean',
    'region_saturation_variance',
    'region_value_mean',
    'region_value_variance',
    'region_eccentricity',
    'region_circularity',
    'region_convexity',
    'percent (black)',
    'percent (white)',
    'percent (gray)',
    'percent (red)',
    'percent (green)',
    'percent (blue)',
    'percent (violet)',
    'percent (yellow)',
    'percent (cyan)',
    'percent (white_blue)',
    'largest_contour_area (black)',
    'largest_contour_area (white)',
    'largest_contour_area (gray)',
    'largest_contour_area (red)',
    'largest_contour_area (green)',
    'largest_contour_area (blue)',
    'largest_contour_area (violet)',
    'largest_contour_area (yellow)',
    'largest_contour_area (cyan)',
    'largest_contour_area (white_blue)',
    'largest_contour_saturation_mean (black)',
    'largest_contour_saturation_mean (white)',
    'largest_contour_saturation_mean (gray)',
    'largest_contour_saturation_mean (red)',
    'largest_contour_saturation_mean (green)',
    'largest_contour_saturation_mean (blue)',
    'largest_contour_saturation_mean (violet)',
    'largest_contour_saturation_mean (yellow)',
    'largest_contour_saturation_mean (cyan)',
    'largest_contour_saturation_mean (white_blue)',
    'largest_contour_saturation_variance (black)',
    'largest_contour_saturation_variance (white)',
    'largest_contour_saturation_variance (gray)',
    'largest_contour_saturation_variance (red)',
    'largest_contour_saturation_variance (green)',
    'largest_contour_saturation_variance (blue)',
    'largest_contour_saturation_variance (violet)',
    'largest_contour_saturation_variance (yellow)',
    'largest_contour_saturation_variance (cyan)',
    'largest_contour_saturation_variance (white_blue)',
    'largest_contour_value_mean (black)',
    'largest_contour_value_mean (white)',
    'largest_contour_value_mean (gray)',
    'largest_contour_value_mean (red)',
    'largest_contour_value_mean (green)',
    'largest_contour_value_mean (blue)',
    'largest_contour_value_mean (violet)',
    'largest_contour_value_mean (yellow)',
    'largest_contour_value_mean (cyan)',
    'largest_contour_value_mean (white_blue)',
    'largest_contour_value_variance (black)',
    'largest_contour_value_variance (white)',
    'largest_contour_value_variance (gray)',
    'largest_contour_value_variance (red)',
    'largest_contour_value_variance (green)',
    'largest_contour_value_variance (blue)',
    'largest_contour_value_variance (violet)',
    'largest_contour_value_variance (yellow)',
    'largest_contour_value_variance (cyan)',
    'largest_contour_value_variance (white_blue)',
    'largest_contour_eccentricity (black)',
    'largest_contour_eccentricity (white)',
    'largest_contour_eccentricity (gray)',
    'largest_contour_eccentricity (red)',
    'largest_contour_eccentricity (green)',
    'largest_contour_eccentricity (blue)',
    'largest_contour_eccentricity (violet)',
    'largest_contour_eccentricity (yellow)',
    'largest_contour_eccentricity (cyan)',
    'largest_contour_eccentricity (white_blue)',
    'largest_contour_circularity (black)',
    'largest_contour_circularity (white)',
    'largest_contour_circularity (gray)',
    'largest_contour_circularity (red)',
    'largest_contour_circularity (green)',
    'largest_contour_circularity (blue)',
    'largest_contour_circularity (violet)',
    'largest_contour_circularity (yellow)',
    'largest_contour_circularity (cyan)',
    'largest_contour_circularity (white_blue)',
    'largest_contour_convexity (black)',
    'largest_contour_convexity (white)',
    'largest_contour_convexity (gray)',
    'largest_contour_convexity (red)',
    'largest_contour_convexity (green)',
    'largest_contour_convexity (blue)',
    'largest_contour_convexity (violet)',
    'largest_contour_convexity (yellow)',
    'largest_contour_convexity (cyan)',
    'largest_contour_convexity (white_blue)',
    'largest_contour_har (black)',
    'largest_contour_har (white)',
    'largest_contour_har (gray)',
    'largest_contour_har (red)',
    'largest_contour_har (green)',
    'largest_contour_har (blue)',
    'largest_contour_har (violet)',
    'largest_contour_har (yellow)',
    'largest_contour_har (cyan)',
    'largest_contour_har (white_blue)',
    'distance_mean (black)',
    'distance_mean (white)',
    'distance_mean (gray)',
    'distance_mean (red)',
    'distance_mean (green)',
    'distance_mean (blue)',
    'distance_mean (violet)',
    'distance_mean (yellow)',
    'distance_mean (cyan)',
    'distance_mean (white_blue)',
    'distance_variance (black)',
    'distance_variance (white)',
    'distance_variance (gray)',
    'distance_variance (red)',
    'distance_variance (green)',
    'distance_variance (blue)',
    'distance_variance (violet)',
    'distance_variance (yellow)',
    'distance_variance (cyan)',
    'distance_variance (white_blue)',
    'perimeter_mean (black)',
    'perimeter_mean (white)',
    'perimeter_mean (gray)',
    'perimeter_mean (red)',
    'perimeter_mean (green)',
    'perimeter_mean (blue)',
    'perimeter_mean (violet)',
    'perimeter_mean (yellow)',
    'perimeter_mean (cyan)',
    'perimeter_mean (white_blue)',
    'perimeter_variance (black)',
    'perimeter_variance (white)',
    'perimeter_variance (gray)',
    'perimeter_variance (red)',
    'perimeter_variance (green)',
    'perimeter_variance (blue)',
    'perimeter_variance (violet)',
    'perimeter_variance (yellow)',
    'perimeter_variance (cyan)',
    'perimeter_variance (white_blue)',
    'area_mean (black)',
    'area_mean (white)',
    'area_mean (gray)',
    'area_mean (red)',
    'area_mean (green)',
    'area_mean (blue)',
    'area_mean (violet)',
    'area_mean (yellow)',
    'area_mean (cyan)',
    'area_mean (white_blue)',
    'area_variance (black)',
    'area_variance (white)',
    'area_variance (gray)',
    'area_variance (red)',
    'area_variance (green)',
    'area_variance (blue)',
    'area_variance (violet)',
    'area_variance (yellow)',
    'area_variance (cyan)',
    'area_variance (white_blue)',
    'har_mean (black)',
    'har_mean (white)',
    'har_mean (gray)',
    'har_mean (red)',
    'har_mean (green)',
    'har_mean (blue)',
    'har_mean (violet)',
    'har_mean (yellow)',
    'har_mean (cyan)',
    'har_mean (white_blue)',
    'har_variance (black)',
    'har_variance (white)',
    'har_variance (gray)',
    'har_variance (red)',
    'har_variance (green)',
    'har_variance (blue)',
    'har_variance (violet)',
    'har_variance (yellow)',
    'har_variance (cyan)',
    'har_variance (white_blue)',
    'contour_count (black)',
    'contour_count (white)',
    'contour_count (gray)',
    'contour_count (red)',
    'contour_count (green)',
    'contour_count (blue)',
    'contour_count (violet)',
    'contour_count (yellow)',
    'contour_count (cyan)',
    'contour_count (white_blue)',
]


def clean_and_stash_numpys(
        imglist,
        extra_data,
        strlist,
        numpy_save_dir,
        imgshape=(99, 99, 3)
):
    """
    Take a list of numpy arrays and a list of text and return one generator
    """
    new_mask = np.zeros((len(imglist),) + imgshape, dtype=np.uint8)
    for i, si in enumerate(imglist):
        max_dim = max(si.shape[:2])
        si_square = np.zeros((max_dim, max_dim, si.shape[2]), dtype=np.uint8)
        si_square[:si.shape[0], :si.shape[1], :] = si

        img = trans.resize(
            si_square,
            imgshape,
            mode='constant',
            anti_aliasing=False,
            preserve_range=True)
        new_mask[i, :, :, :] = img.astype(np.uint8)
    le = LabelEncoder()
    strlist_le = le.fit_transform(strlist)
    ohe = OneHotEncoder(sparse=False, categories='auto')
    strlist_ohe = ohe.fit_transform(strlist_le.reshape(-1, 1))
    ohe_lut = {}
    for cat in ohe.categories_[0].astype(np.int):
        ohe_lut[str(cat)] = le.classes_[cat]

    fh = open(os.path.join(numpy_save_dir, 'ohe_meta.json'), 'w')
    json.dump(ohe_lut, fh, indent=2)
    fh.close()

    np.save(os.path.join(numpy_save_dir, 'images.npy'), new_mask)
    np.save(os.path.join(numpy_save_dir, 'extra.npy'), extra_data)
    np.save(os.path.join(numpy_save_dir, 'ohelabels.npy'), strlist_ohe)
    print('Numpys stashed!')


def get_imageset_in_memory(
        image_set_directory,
        test_image_name,
        balance_train=True,
        masked=False,
        pseudo_color=False
):
    """
    Takes an image set directory and a test image file name.
    Returns 4 numpy arrays: train_regions, train_labels, test_regions, test_labels
    """
    image_set_metadata = get_training_data_for_image_set(image_set_directory)
    test_metadata = image_set_metadata.pop(test_image_name)

    images_train = []
    extra_train = []
    labels_train = []
    images_test = []
    extra_test = []
    labels_test = []

    for key, value in image_set_metadata.items():
        hsv_img = value['hsv_img']
        rgb_img = cv2.cvtColor(
            hsv_img,
            cv2.COLOR_HSV2RGB
        )

        for region in value['regions']:
            labels_train.append(region['label'])

            features = lm_utils.generate_features(hsv_img, region['points'])

            extra_train.append(
                [features[x] for x in include_features]
            )

            if masked:
                tmp_img = extract_contour_bounding_box_masked(
                    rgb_img,
                    region['points']
                )
            else:
                tmp_img = extract_contour_bounding_box(rgb_img, region['points'])

            if pseudo_color:
                tmp_img = generate_pseudo_color_image(tmp_img)

            images_train.append(tmp_img)

    hsv_img = test_metadata['hsv_img']
    rgb_img = cv2.cvtColor(
        hsv_img,
        cv2.COLOR_HSV2RGB
    )
    for region in test_metadata['regions']:
        labels_test.append(region['label'])

        features = lm_utils.generate_features(hsv_img, region['points'])

        extra_test.append(
            [features[x] for x in include_features]
        )

        if masked:
            tmp_img = extract_contour_bounding_box_masked(
                rgb_img,
                region['points']
            )
        else:
            tmp_img = extract_contour_bounding_box(rgb_img, region['points'])

        if pseudo_color:
            tmp_img = generate_pseudo_color_image(tmp_img)

        images_test.append(tmp_img)

    extra_train = np.array(extra_train)
    max_area = extra_train[:, 0].max()

    extra_train[:, 0] = extra_train[:, 0] / max_area
    extra_test = np.array(extra_test)
    extra_test[:, 0] = extra_test[:, 0] / max_area

    if balance_train:
        images_train, extra_train, labels_train = make_balanced(
            images_train,
            extra_train,
            labels_train
        )

    assert len(images_train) == len(labels_train)

    processed_data = (
        images_train,
        extra_train,
        labels_train,
        images_test,
        extra_test,
        labels_test
    )

    return processed_data


def make_balanced(xarray, extra_array, ylist):
    counts = Counter(ylist)
    bringup = np.max(list(counts.values()))
    bringupdict = {key: bringup//value for key, value in counts.items()}
    newx = []
    new_extra = []
    newy = []
    for x in set(ylist):
        indices = [i for i, val in enumerate(ylist) if val == x]
        newindices = list(itertools.repeat(indices, bringupdict[x]))
        newindicesflat = [item for sublist in newindices for item in sublist]
        for index in newindicesflat:
            newx.append(xarray[index])
            new_extra.append(extra_array[index])
            newy.append(ylist[index])
    return newx, new_extra, newy


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
        tmp_image = Image.open(os.path.join(image_set_dir, image_name))
        tmp_image = np.asarray(tmp_image)

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
    x1, y1, w, h = cv2.boundingRect(contour)

    return [x1, y1, x1 + w, y1 + h]


def make_binary_mask(contour, img_dims):
    mask = np.zeros(img_dims, dtype=np.uint8)
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


def generate_pseudo_color_image(rgb_image):
    pseudo_color_img = np.zeros(rgb_image.shape, dtype=np.uint8)
    for color, value in rgb_color_lut.items():
        color_mask = lm_utils.create_mask(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV), [color])

        pseudo_color_img[color_mask > 0] = value

    for color, value in rgb2_color_lut.items():
        color_mask = lm_utils.create_mask(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV), [color])

        pseudo_color_img[color_mask > 0] = value

    return pseudo_color_img
