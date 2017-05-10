# For a single-input model with 2 classes (binary classification):
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Reshape, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
from keras.optimizers import SGD
from keras.utils import plot_model
import keras.backend as K

import pydot_ng
import numpy as np
import h5py
import keras
from helpers import preprocess
from helpers import models


# Generate dummy data
x_train, y_train, _ = preprocess.produce_training_set(traindir = 'D:/EtienneData/trainmdeiummedlab',
                                                   trainsize = 100,
                                                   numlabs = 18
                                                   )
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
#y_train = K.one_hot(y_train,num_classes = 18)
y_labs = keras.utils.to_categorical(y_train)
print(y_labs.shape)

model = models.models_dict['up'](input_shape = (256,512,3),
                                 num_classes = 18
                                 )
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
#model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 512, 3)))
#model.add(Conv2D(32, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

#model.add(Conv2D(64, (3, 3), activation='relu'))
#model.add(Conv2D(64, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

#model.add(Flatten())
#model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.summary()
model.fit(x_train, y_labs, epochs=20, verbose=2 , batch_size=5)
#model.save(filepath = 'model.hdf5')
plot_model(model, to_file='modeltest.png')