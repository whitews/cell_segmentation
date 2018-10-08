"""
Mask R-CNN
Display and Visualization Functions.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
import random
import colorsys
from utils.data import compute_bbox, make_binary_mask

import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Polygon

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library


############################################################
#  Visualization
############################################################

def display_images(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interporlation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                alpha=0.7,
                linestyle="dashed",
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            x = random.randint(x1, (x1 + x2) // 2)
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()


def display_class_prediction_overlaps(
        image,
        segments,
        true_regions,
        test_regions,
        figsize=(16, 16),
        ax=None,
        show_mask=True,
        show_bbox=True,
        colors=None,
        captions=None
):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    for key, x in segments.items():
        # If no axis is passed, create one and automatically call show()
        ax = None
        if not ax:
            _, ax = plt.subplots(1, figsize=figsize)
            auto_show = True

        # Generate random colors
        # Number of color segments (choosing three to match tp, fp, fn
        # colors = colors or random_colors(3)
        colors = [(0.0, 1.0, 0.40000000000000036),
                  (1.0, 0.0, 1.0),
                  (1.0, 1.0, 0.0)]
        color_labels = ['tp', 'fn', 'fp']
        # Show area outside image boundaries.
        height, width = image.shape[:2]
        ax.set_ylim(height + 10, -10)
        ax.set_xlim(-10, width + 10)
        ax.axis('off')
        ax.set_title(key)
        masked_image = image.astype(np.uint32).copy()

        for typekey, type_val in x.items():
            color = colors[color_labels.index(typekey)]
            for seg in type_val:
                if 'gt_ind' in list(seg.keys()):
                    contour = true_regions['regions'][seg['gt_ind']]['points']
                    seglabel = 'gt'
                elif 'test_ind' in list(seg.keys()):
                    contour = test_regions[seg['test_ind']]['contour']
                    if 'prob' in list(seg.keys()):
                        seglabel = 'IOU: {0:.2}, PROB: {0:.2%}'.format(seg['iou'], seg['prob'])
                    else:
                        seglabel = 'IOU: {0:.2}'.format(seg['iou'])

                x1, y1, x2, y2 = compute_bbox(contour)
                if show_bbox:
                    p = patches.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        linewidth=2,
                        alpha=0.7,
                        linestyle="dashed",
                        edgecolor=color,
                        facecolor='none'
                    )
                    ax.add_patch(p)
                ax.text(x1, y1 + 8, seglabel,
                        color='w', size=15, backgroundcolor="none")
                # Mask
                mask = make_binary_mask(contour, (masked_image.shape[0], masked_image.shape[1]))
                if show_mask:
                    masked_image = apply_mask(masked_image, mask, color)

        ax.imshow(masked_image.astype(np.uint8))
        if auto_show:
            plt.show()


def display_instances_segments(
        image,
        segments,
        class_names,
        scores=None,
        title="",
        figsize=(16, 16),
        ax=None,
        show_mask=True,
        show_bbox=True,
        colors=None,
        captions=None
):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = len(segments)

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(len(class_names))

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    color = colors[0]
    caption = "{}".format('Automatic Segmentation')
    for x in segments:
        if 'prob' in list(x.keys()):
            max_key = max(x['prob'], key=lambda k: x['prob'][k])
            max_value = "{0:.2%}".format(x['prob'][max_key])
            index = class_names.index(max_key)
            color = colors[index]
            caption = "{} {}".format(max_key, max_value)

        try:
            x1, y1, x2, y2 = compute_bbox(x['contour'])
        except:
            x1, y1, x2, y2 = compute_bbox(x['points'])
        if show_bbox:
            p = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                alpha=0.7,
                linestyle="dashed",
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(p)

        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        try:
            mask = make_binary_mask(
                x['contour'],
                (masked_image.shape[0], masked_image.shape[1])
            )
        except:
            mask = make_binary_mask(
                x['points'],
                (masked_image.shape[0], masked_image.shape[1])
            )
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()
