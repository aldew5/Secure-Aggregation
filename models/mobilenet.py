
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from train import *

df_x_train = pd.read_csv('./mnist/x_train_partition_1.csv', header=None)
df_y_train = pd.read_csv('./mnist/y_train_partition_1.csv', header=None)
df_x_test = pd.read_csv('./mnist/x_test.csv', header=None)
df_y_test = pd.read_csv('./mnist/y_test.csv', header=None)

# Convert DataFrames to NumPy arrays
x_train = df_x_train.values
y_train = df_y_train.values
x_test = df_x_test.values
y_test = df_y_test.values

num_samples_train = x_train.shape[0]
num_samples_test = x_test.shape[0]
img_height, img_width, num_channels = 28, 28, 1  # MNIST dimensions

x_train = x_train.reshape(num_samples_train, img_height, img_width, num_channels)
x_test = x_test.reshape(num_samples_test, img_height, img_width, num_channels)

# Resize images to 32x32x3
def resize_images(images):
    images_resized = np.zeros((images.shape[0], 32, 32, 3), dtype=np.float32)
    for i in range(images.shape[0]):
        img = images[i]
        # Convert single channel to 3 channels by repeating
        img_rgb = np.stack([img.squeeze()] * 3, axis=-1)
        img_resized = tf.image.resize(img_rgb, [32, 32])
        images_resized[i] = img_resized
    return images_resized

x_train_resized = resize_images(x_train)
x_test_resized = resize_images(x_test)

# Convert labels to categorical
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build MobileNet model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# freeze layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    x_train_resized, y_train,
    validation_data=(x_test_resized, y_test),
    epochs=10,  # Number of epochs
    batch_size=32
)

loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {accuracy}')
