from utils.data import get_imageset_in_memory, clean_and_stash_numpys, \
    create_generator_from_stash
import matplotlib.pyplot as plt
import os

cwd = os.getcwd()
train_dir = os.path.join(cwd, 'data/train_numpy')

image_set_dir = os.path.join(cwd, 'data/image_set_73')
x, y, testx, testy = get_imageset_in_memory(image_set_dir)
clean_and_stash_numpys(x, y, train_dir)
clean_and_stash_numpys(testx, testy, train_dir)

gen = create_generator_from_stash(train_dir)
x, y = next(gen)

plt.imshow(x[0, :, :, :])
plt.show()
