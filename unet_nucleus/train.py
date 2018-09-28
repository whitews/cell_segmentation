from unet_nucleus.data_generator import train_generator
from unet_nucleus.architecture import build_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler


model_checkpoint = ModelCheckpoint('unet_nucleus.hdf5', monitor='loss',verbose=1, save_best_only=True)

model = build_model()

model.fit(train_generator, steps_per_epoch=2000, epochs=5, callbacks=[model_checkpoint])