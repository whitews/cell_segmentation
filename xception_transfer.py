from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import matplotlib.pyplot as plt
from classifier import architecture as arch
import os
import numpy as np

mask_opt = 'masked'
color_opt = 'full'
balance_opt = 'unbalanced'

parent_dir = 'model_data'
opt_str = "_".join([mask_opt, color_opt, balance_opt])
train_dir = "_".join(['train', 'numpy', opt_str])
train_path = os.path.join(parent_dir, train_dir)
model_file = "_".join(['model', opt_str]) + '.hdf5'
model_cache_path = os.path.join(parent_dir, model_file)

test_dir = "_".join(['test', 'numpy', opt_str])
test_path = os.path.join(parent_dir, test_dir)

x = np.load(os.path.join(test_path, 'images.npy'))
x = x.astype(np.float32)
x = x / 255.0
extra = np.load(os.path.join(test_path, 'extra.npy'))
y = np.load(os.path.join(test_path, 'ohelabels.npy'))

gen = arch.create_generator_from_stash(
    train_path,
    batch_size=20,
    include_extra=True
)

checkpoint = ModelCheckpoint(
    model_cache_path,
    monitor='loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    period=1
)

if os.path.exists(model_cache_path):
    model = load_model(model_cache_path)
    print('Loading model from cache...')
else:
    model = arch.build_model(5, extra.shape[1])
    #model = arch.build_xception_model(5)

h = model.fit_generator(
    gen,
    steps_per_epoch=80,
    epochs=15,
    callbacks=[checkpoint],
    validation_data=([x, extra], y),
    #initial_epoch=10
)

print(h.history.keys())

# summarize history for accuracy
plt.plot(h.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(h.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
