from keras.callbacks import ModelCheckpoint
from utils.data import create_generator_from_stash
import matplotlib.pyplot as plt
from classifier.architecture import build_model_double, build_model

gen = create_generator_from_stash('data/train_numpy')

checkpoint = ModelCheckpoint(
    'classifier/model_1.hdf5',
    monitor='loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    period=1
)

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

model2 = build_model_double('classifier/model_1.hdf5')

checkpoint2 = ModelCheckpoint(
    'classifier/model_2.hdf5',
    monitor='acc',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    period=1
)

h2 = model.fit_generator(
    gen,
    steps_per_epoch=70,
    epochs=10,
    callbacks=[checkpoint2]
)

plt.plot(h2.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(h2.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
