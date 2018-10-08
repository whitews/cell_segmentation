from utils.data import (
    get_training_data_for_image_set,
    extract_contour_bounding_box_masked,
)
import matplotlib.pyplot as plt
import cv2
from utils.visualize import display_instances_segments
from skimage.segmentation import chan_vese, mark_boundaries
from skimage import img_as_float
from skimage.segmentation import (morphological_chan_vese, inverse_gaussian_gradient)


a = get_training_data_for_image_set('data/image_set_90')

# dat = []
# for x in a['2015-04-029_20X_C57Bl6_E16.5_LMM.14.24.4.46_SOX9_SFTPC_ACTA2_003.tif']['regions']:
#     print(x['label'])
#     if x['label']=='distal acinar tubule bud':
#         dat.append(x)

full_image = cv2.cvtColor(
        a['2015-04-029_60X_C57Bl6_E16.5_LMM.14.24.4.46_SOX9_SFTPC_ACTA2_003.tif']['hsv_img'],
        cv2.COLOR_HSV2RGB
)

display_instances_segments(
    full_image,
    [a['2015-04-029_60X_C57Bl6_E16.5_LMM.14.24.4.46_SOX9_SFTPC_ACTA2_003.tif']['regions'][2]],
    ['raw']
)

# sub = extract_contour_bounding_box(
#     full_image,
#     a['2015-04-029_60X_C57Bl6_E16.5_LMM.14.24.4.46_SOX9_SFTPC_ACTA2_003.tif']['regions'][2]['points']
# )

sub = extract_contour_bounding_box_masked(
    full_image,
    a['2015-04-029_60X_C57Bl6_E16.5_LMM.14.24.4.46_SOX9_SFTPC_ACTA2_003.tif']['regions'][2]['points']
)
sub_gray = cv2.cvtColor(sub,cv2.COLOR_RGB2GRAY)
plt.imshow(sub)
plt.show()

sbg = sub_gray
segments = chan_vese(sbg, mu=0.1)
plt.imshow(mark_boundaries(sbg, segments))
plt.show()


from skimage.segmentation import morphological_geodesic_active_contour
sbgf = img_as_float(sbg)
gimage = inverse_gaussian_gradient(sbgf)
plt.imshow(gimage)
plt.show()
gimage_mg = morphological_geodesic_active_contour(gimage, 230)
plt.imshow(gimage_mg)
plt.show()

# kmeans = k_means_segments(sub)
# plt.imshow(kmeans)
# plt.show()
#
# b_and_w, enclosed = watershed_grey(sub)
# plt.imshow(enclosed)
# plt.show()
# fig, axes = plt.subplots(1,2,figsize=(10,4))
# axes[0].imshow(b_and_w, cmap=plt.cm.gray)
# axes[1].imshow(enclosed)
# plt.show()
# fig.close()
# plt.clf()
# plt.cla()


# custom = hsv_thresholding(sub)
# # fig2, axes2 = plt.subplots(1,2,figsize=(10,4))
# # axes2[0].imshow(sub)
# # axes2[1].imshow(custom)
# # plt.show()
# plt.imshow(custom)
# plt.show()

mcv = morphological_chan_vese(sub)