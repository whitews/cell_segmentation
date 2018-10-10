from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from utils.data import create_generator_from_stash
import matplotlib.pyplot as plt
from classifier.architecture import build_model
import os

gen = create_generator_from_stash('model_data/train_numpy_masked')
model_cache = 'model_data/model_1_plain_masked.hdf5'

checkpoint = ModelCheckpoint(
    model_cache,
    monitor='loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    period=1
)

if os.path.exists(model_cache):
    model = load_model(model_cache)
    print('Loading model from cache...')
else:
    model = build_model()

h = model.fit_generator(
    gen,
    steps_per_epoch=70,
    epochs=10,
    callbacks=[checkpoint]
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
