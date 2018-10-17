from utils.data import get_imageset_in_memory, clean_and_stash_numpys
import os

mask_opt = 'masked'
color_opt = 'full'
balance_opt = 'unbalanced'

cwd = os.getcwd()
parent_dir = 'model_data'
opt_str = "_".join([mask_opt, color_opt, balance_opt])

train_dir = "_".join(['train', 'numpy', opt_str])
train_path = os.path.join(cwd, parent_dir, train_dir)
test_dir = "_".join(['test', 'numpy', opt_str])
test_path = os.path.join(cwd, parent_dir, test_dir)

image_set_dir = os.path.join(cwd, 'data/image_set_73')
test_img_name = '2015-04-029_20X_C57Bl6_E16.5_LMM.14.24.4.46_SOX9_SFTPC_ACTA2_001.tif'

if not os.path.exists(train_path):
    os.makedirs(train_path)
if not os.path.exists(test_path):
    os.makedirs(test_path)

if mask_opt == 'masked':
    is_masked = True
else:
    is_masked = False

if balance_opt == 'balanced':
    is_balanced = True
else:
    is_balanced = False

if color_opt == 'pseudo':
    is_pseudo = True
else:
    is_pseudo = False

x, extra_x, y, test_x, test_extra_x, test_y = get_imageset_in_memory(
    image_set_dir,
    test_img_name,
    balance_train=is_balanced,
    masked=is_masked,
    pseudo_color=is_pseudo
)
clean_and_stash_numpys(x, extra_x, y, train_path)
clean_and_stash_numpys(test_x, test_extra_x, test_y, test_path)
