from keras.models import Model
from keras.layers import Input,Conv2D
from keras.initializers import random_uniform, zeros


models_dict = {
    'simple_model' : simple_model


    }

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
    outs = 
    mod = Model(
    return mod