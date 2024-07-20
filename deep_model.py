import keras
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = np.stack([x_train]*3, axis=-1)
x_test = np.stack([x_test]*3, axis=-1)

x_train = tf.image.resize(x_train, (224, 224)).numpy()
x_test = tf.image.resize(x_test, (224, 224)).numpy()

x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

input_shape = (224, 224, 3)

resnet = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

# freeze
for layer in resnet.layers:
    layer.trainable = False

# tunable layers
inputs = Input(shape=input_shape)
x = resnet(inputs, training=False)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs, outputs)

model.compile(optimizer=Adam(learning_rate=0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
              
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))




