from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.applications.xception import Xception


def build_model():
    base_model = Xception(weights=None, include_top=False, input_shape=(299, 299, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(5, activation='softmax')(x)
    m = Model(inputs=base_model.input, outputs=predictions)
    m.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return m
