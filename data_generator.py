from utils.data import get_imageset_in_memory, clean_and_stash_numpys
import os

cwd = os.getcwd()
train_dir = os.path.join(cwd, 'model_data/train_numpy')
test_dir = os.path.join(cwd, 'model_data/test_numpy')
test_img_name = '2015-04-029_20X_C57Bl6_E16.5_LMM.14.24.4.46_SOX9_SFTPC_ACTA2_001.tif'

if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

image_set_dir = os.path.join(cwd, 'data/image_set_73')
x, y, test_x, test_y = get_imageset_in_memory(image_set_dir, test_img_name)
clean_and_stash_numpys(x, y, train_dir)
clean_and_stash_numpys(test_x, test_y, test_dir)
