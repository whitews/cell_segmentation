import os
import json
from PIL import Image
import cv2
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt

from skimage.segmentation import felzenszwalb, slic, quickshift, random_walker
from skimage.segmentation import mark_boundaries
from skimage import data, img_as_float
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)


def get_training_data_for_image_set(image_set_dir):
    # Each image set directory will have a 'regions.json' file. This regions file
    # has keys of the image file names in the image set, and the value for each image
    # is a list of segmented polygon regions.
    # First, we will read in this file and get the file names for our images
    regions_file = open(os.path.join(image_set_dir, 'regions.json'))
    regions_json = json.load(regions_file)
    regions_file.close()

    # output will be a dictionary of training data, were the polygon points dict is a
    # numpy array. The keys will still be the image names
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

def k_means_segments(image):
    h, w, d = image.shape
    cell_area = max(h, w)
    n_segments = w * h // 200
    segments = slic(image, n_segments=n_segments, compactness=10, sigma=10,
                    multichannel=True, convert2lab=True)
    marked_image = mark_boundaries(image, segments)
    return marked_image

def watershed_grey(img):
    image = np.copy(img)
    greyimg = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(greyimg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]
    return thresh, image

def hsv_thresholding(image):
    hsvimg = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    green_mask = cv2.inRange(hsvimg, HSV_RANGES['green'][0]['lower'], HSV_RANGES['green'][0]['upper'])
    white_mask = cv2.inRange(hsvimg, HSV_RANGES['white'][0]['lower'], HSV_RANGES['white'][0]['upper'])
    _, contours, hierarchy = cv2.findContours(green_mask, 1, 2)
    a = cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
    _, contours, hierarchy = cv2.findContours(white_mask, 1, 2)
    b = cv2.drawContours(a, contours, -1, (255, 255, 255), 1)
    return b

def create_mask(hsv_img, colors):
    """
    Creates a binary mask from HSV image using given colors.
    """

    # noinspection PyUnresolvedReferences
    mask = np.zeros((hsv_img.shape[0], hsv_img.shape[1]), dtype=np.uint8)

    for color in colors:
        for color_range in HSV_RANGES[color]:
            # noinspection PyUnresolvedReferences
            mask += cv2.inRange(
                hsv_img,
                color_range['lower'],
                color_range['upper']
            )

    return mask

def lungmap_custom(image):
    color_blur_kernel = (27, 27)
    img_blur_c = cv2.GaussianBlur(
        image,
        color_blur_kernel,
        0
    )
    img_blur_c_hsv = cv2.cvtColor(img_blur_c, cv2.COLOR_RGB2HSV)
    mask = create_mask(
        img_blur_c_hsv,
        [
            'cyan',
            'gray',
            'green',
            'red',
            'violet',
            'white',
            'yellow'
        ]
    )
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    mask = fill_holes(mask)
    pass


def scott_method(image):
    """
    https://www.researchgate.net/publication/239587584_Watershed_Segmentation_of_Cervical_Images_Using_Multiscale_Morphological_Gradient_and_HSI_Color_Space
    :param image:
    :return:
    """
    hsv_image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2HSV
    )
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


HSV_RANGES = {
    # red is a major color
    'red': [
        {
            'lower': np.array([0, 39, 64]),
            'upper': np.array([20, 255, 255])
        },
        {
            'lower': np.array([161, 39, 64]),
            'upper': np.array([180, 255, 255])
        }
    ],
    # yellow is a minor color
    'yellow': [
        {
            'lower': np.array([21, 39, 64]),
            'upper': np.array([40, 255, 255])
        }
    ],
    # green is a major color
    'green': [
        {
            'lower': np.array([41, 39, 64]),
            'upper': np.array([80, 255, 255])
        }
    ],
    # cyan is a minor color
    'cyan': [
        {
            'lower': np.array([81, 39, 64]),
            'upper': np.array([100, 255, 255])
        }
    ],
    # blue is a major color
    'blue': [
        {
            'lower': np.array([101, 39, 64]),
            'upper': np.array([140, 255, 255])
        }
    ],
    # violet is a minor color
    'violet': [
        {
            'lower': np.array([141, 39, 64]),
            'upper': np.array([160, 255, 255])
        }
    ],
    # next are the monochrome ranges
    # black is all H & S values, but only the lower 10% of V
    'black': [
        {
            'lower': np.array([0, 0, 0]),
            'upper': np.array([180, 255, 63])
        }
    ],
    # gray is all H values, lower 15% of S, & between 11-89% of V
    'gray': [
        {
            'lower': np.array([0, 0, 64]),
            'upper': np.array([180, 38, 228])
        }
    ],
    # white is all H values, lower 15% of S, & upper 10% of V
    'white': [
        {
            'lower': np.array([0, 0, 229]),
            'upper': np.array([180, 38, 255])
        }
    ]
}

def fill_holes(mask):
    """
    Fills holes in a given binary mask.
    """
    # noinspection PyUnresolvedReferences
    ret, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    # noinspection PyUnresolvedReferences
    new_mask, contours, hierarchy = cv2.findContours(
        thresh,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt in contours:
        # noinspection PyUnresolvedReferences
        cv2.drawContours(new_mask, [cnt], 0, 255, -1)

    return new_mask

def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store

def morphological_chan_vese(img):
    image = img_as_float(img)

    # Initial level set
    init_ls = checkerboard_level_set(image.shape, 6)
    # List with intermediate results for plotting the evolution
    evolution = []
    callback = store_evolution_in(evolution)
    ls = morphological_chan_vese(image, 35, init_level_set=init_ls, smoothing=3,
                                 iter_callback=callback)






