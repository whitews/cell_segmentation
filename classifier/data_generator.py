from utils.data import get_imageset_in_memory, clean_and_stash_numpys, \
    create_generator_from_stash
import matplotlib.pyplot as plt


image_set_dir = 'data/image_set_73'
x, y, testx, testy = get_imageset_in_memory(image_set_dir)
clean_and_stash_numpys(x, y, 'data/train_numpy')
clean_and_stash_numpys(testx, testy, 'data/test_numpy')

gen = create_generator_from_stash('data/train_numpy')
x, y = next(gen)

plt.imshow(x[0, :, :, :])
plt.show()
