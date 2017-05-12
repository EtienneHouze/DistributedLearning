from keras.models import Model, Sequential
from keras.layers import Input,Conv2D,Lambda
from keras.initializers import random_uniform, zeros
from keras.layers.advanced_activations import PReLU
import keras.backend as K


"""
    This script contains all the builder functions used to build the keras models for 
"""



def simple_model(input_shape):
    ins = Input(shape = input_shape)
    a = Conv2D(filters = 32, 
               kernel_size = (3,3),
               padding = 'same',
               use_bias = True,
               kernel_initializer = random_uniform(minval = -0.1, maxval = 0.1),
               bias_initialzer = zeros()
               )(ins)
    a = Conv2D(filters = 32, 
               kernel_size = (3,3),
               padding = 'same',
               use_bias = True,
               kernel_initializer = random_uniform(minval = -0.1, maxval = 0.1),
               bias_initialzer = zeros()
               )(a)
    a = Conv2D(filters = 32, 
               kernel_size = (3,3),
               padding = 'same',
               use_bias = True,
               kernel_initializer = random_uniform(minval = -0.1, maxval = 0.1),
               bias_initialzer = zeros()
               )(a)
    return mod

def upscaled(input_shape, num_classes):
    
    mod = Sequential()

    mod.add(Conv2D(filters = 16,
                   kernel_size = (3,3),
                   padding = 'same',
                   dilation_rate = 1,
                   activation = 'linear',
                   use_bias = True,
                   kernel_initializer = random_uniform(),
                   bias_initializer = random_uniform(),
                   input_shape = input_shape
                   )
            )
    mod.add(PReLU())
    mod.add(Conv2D(filters = 32,
                   kernel_size = (3,3),
                   padding = 'same',
                   dilation_rate = 1,
                   activation = 'linear',
                   use_bias = True,
                   kernel_initializer = random_uniform(),
                   bias_initializer = random_uniform(),
                   input_shape = input_shape
                   )
            )
    mod.add(PReLU())
    mod.add(Conv2D(filters = 64,
                   kernel_size = (3,3),
                   padding = 'same',
                   dilation_rate = 2,
                   activation = 'linear',
                   use_bias = True,
                   kernel_initializer = random_uniform(),
                   bias_initializer = random_uniform(),
                   input_shape = input_shape
                   )
            )
    mod.add(PReLU())
    mod.add(Conv2D(filters = 64,
                   kernel_size = (3,3),
                   padding = 'same',
                   dilation_rate = 2,
                   activation = 'linear',
                   use_bias = True,
                   kernel_initializer = random_uniform(),
                   bias_initializer = random_uniform(),
                   input_shape = input_shape
                   )
            )
    mod.add(PReLU())
    mod.add(Conv2D(filters = 128,
                   kernel_size = (3,3),
                   padding = 'same',
                   dilation_rate = 4,
                   activation = 'relu',
                   use_bias = True,
                   kernel_initializer = random_uniform(),
                   bias_initializer = random_uniform(),
                   input_shape = input_shape
                   )
            )
    mod.add(Conv2D(filters = 128,
                   kernel_size = (3,3),
                   padding = 'same',
                   dilation_rate = 4,
                   activation = 'relu',
                   use_bias = True,
                   kernel_initializer = random_uniform(),
                   bias_initializer = random_uniform(),
                   input_shape = input_shape
                   )
            )
    mod.add(Conv2D(filters = 128,
                   kernel_size = (3,3),
                   padding = 'same',
                   dilation_rate = 8,
                   activation = 'relu',
                   use_bias = True,
                   kernel_initializer = random_uniform(),
                   bias_initializer = random_uniform(),
                   input_shape = input_shape
                   )
            )
    mod.add(Conv2D(filters = 128,
                   kernel_size = (3,3),
                   padding = 'same',
                   dilation_rate = 8,
                   activation = 'relu',
                   use_bias = True,
                   kernel_initializer = random_uniform(),
                   bias_initializer = random_uniform(),
                   input_shape = input_shape
                   )
            )
    mod.add(Conv2D(filters = 256,
                   kernel_size = (3,3),
                   padding = 'same',
                   dilation_rate = 16,
                   activation = 'relu',
                   use_bias = True,
                   kernel_initializer = random_uniform(),
                   bias_initializer = random_uniform(),
                   input_shape = input_shape
                   )
            )
    mod.add(Conv2D(filters = num_classes,
                   kernel_size = (1,1),
                   padding = 'same',
                   dilation_rate = 4,
                   activation = 'softmax',
                   use_bias = True,
                   kernel_initializer = random_uniform(),
                   bias_initializer = random_uniform(),
                   input_shape = input_shape
                   )
            )

    return mod

def upscaled_truncated(input_shape, num_classes):
    
    mod = Sequential()

    mod.add(Conv2D(filters = 16,
                   kernel_size = (3,3),
                   padding = 'same',
                   dilation_rate = 1,
                   activation = 'linear',
                   use_bias = True,
                   kernel_initializer = random_uniform(),
                   bias_initializer = random_uniform(),
                   input_shape = input_shape
                   )
            )
    mod.add(PReLU())
    mod.add(Conv2D(filters = 32,
                   kernel_size = (3,3),
                   padding = 'same',
                   dilation_rate = 1,
                   activation = 'linear',
                   use_bias = True,
                   kernel_initializer = random_uniform(),
                   bias_initializer = random_uniform()
                   )
            )
    mod.add(PReLU())
    mod.add(Conv2D(filters = 64,
                   kernel_size = (3,3),
                   padding = 'same',
                   dilation_rate = 2,
                   activation = 'linear',
                   use_bias = True,
                   kernel_initializer = random_uniform(),
                   bias_initializer = random_uniform()
                   )
            )
    mod.add(PReLU())
    mod.add(Conv2D(filters = 64,
                   kernel_size = (3,3),
                   padding = 'same',
                   dilation_rate = 2,
                   activation = 'linear',
                   use_bias = True,
                   kernel_initializer = random_uniform(),
                   bias_initializer = random_uniform()
                   )
            )
    mod.add(PReLU())
    
    mod.add(Conv2D(filters = num_classes,
                   kernel_size = (1,1),
                   padding = 'same',
                   dilation_rate = 4,
                   activation = 'softmax',
                   use_bias = True,
                   kernel_initializer = random_uniform(),
                   bias_initializer = random_uniform()
                   )
            )

    return mod

models_dict = {
    'simple_model' : simple_model,
    'up' : upscaled,
    'up_mini' : upscaled_truncated

    }