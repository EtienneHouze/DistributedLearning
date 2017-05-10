# For a single-input model with 2 classes (binary classification):
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Reshape, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
from keras.optimizers import SGD
from keras.utils import plot_model
import pydot_ng
import numpy as np
import h5py
import keras
from helpers import preprocess


# Generate dummy data
x_train = np.random.random((100, 256, 512, 3))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
x_test = np.random.random((20, 256, 512, 3))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 512, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.summary()
model.fit(x_train, y_train, epochs=20, verbose=2 , batch_size=10)
#model.save(filepath = 'model.hdf5')
plot_model(model, to_file='model16.png')