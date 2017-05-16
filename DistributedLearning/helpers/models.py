from keras.initializers import random_uniform, zeros
from keras.layers import Input, Conv2D
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential, Model

"""
    This script contains all the builder functions used to build the keras models for 
"""


def simple_model(input_shape):
    """
    Dummy function, just to test. Builds a reallys simple model, very fast but useless.
    :param 
        input_shape: a tuple of 3 ints, the shape of the input. 
    :return
        mod : a keras model of the network
    """
    ins = Input(shape=input_shape)
    a = Conv2D(filters=32,
               kernel_size=(3, 3),
               padding='same',
               use_bias=True,
               kernel_initializer=random_uniform(minval=-0.1, maxval=0.1),
               bias_initialzer=zeros(),
               __name__='Test'
               )(ins)
    a = Conv2D(filters=32,
               kernel_size=(3, 3),
               padding='same',
               use_bias=True,
               kernel_initializer=random_uniform(minval=-0.1, maxval=0.1),
               bias_initialzer=zeros()
               )(a)
    a = Conv2D(filters=32,
               kernel_size=(3, 3),
               padding='same',
               use_bias=True,
               kernel_initializer=random_uniform(minval=-0.1, maxval=0.1),
               bias_initialzer=zeros()
               )(a)
    mod = Model(inputs=ins,
                outputs=a,
                name='Network')
    return mod


def upscaled(input_shape, num_classes):
    """
    Builds a simple network using upscaled 2D convolutions.
    :param 
        input_shape: tuple of 3 integers, the shape of the input of the network
        num_classes: integer, number of classes in the output.
    :return
        mod: a keras model of the network.
            
    """
    mod = Sequential()

    mod.add(Conv2D(filters=16,
                   kernel_size=(3, 3),
                   padding='same',
                   dilation_rate=1,
                   activation='linear',
                   use_bias=True,
                   kernel_initializer=random_uniform(),
                   bias_initializer=random_uniform(),
                   input_shape=input_shape
                   )
            )
    mod.add(PReLU())
    mod.add(Conv2D(filters=32,
                   kernel_size=(3, 3),
                   padding='same',
                   dilation_rate=1,
                   activation='linear',
                   use_bias=True,
                   kernel_initializer=random_uniform(),
                   bias_initializer=random_uniform(),
                   input_shape=input_shape
                   )
            )
    mod.add(PReLU())
    mod.add(Conv2D(filters=64,
                   kernel_size=(3, 3),
                   padding='same',
                   dilation_rate=2,
                   activation='linear',
                   use_bias=True,
                   kernel_initializer=random_uniform(),
                   bias_initializer=random_uniform(),
                   input_shape=input_shape
                   )
            )
    mod.add(PReLU())
    mod.add(Conv2D(filters=64,
                   kernel_size=(3, 3),
                   padding='same',
                   dilation_rate=2,
                   activation='linear',
                   use_bias=True,
                   kernel_initializer=random_uniform(),
                   bias_initializer=random_uniform(),
                   input_shape=input_shape
                   )
            )
    mod.add(PReLU())
    mod.add(Conv2D(filters=128,
                   kernel_size=(3, 3),
                   padding='same',
                   dilation_rate=4,
                   activation='relu',
                   use_bias=True,
                   kernel_initializer=random_uniform(),
                   bias_initializer=random_uniform(),
                   input_shape=input_shape
                   )
            )
    mod.add(Conv2D(filters=128,
                   kernel_size=(3, 3),
                   padding='same',
                   dilation_rate=4,
                   activation='relu',
                   use_bias=True,
                   kernel_initializer=random_uniform(),
                   bias_initializer=random_uniform(),
                   input_shape=input_shape
                   )
            )
    mod.add(Conv2D(filters=128,
                   kernel_size=(3, 3),
                   padding='same',
                   dilation_rate=8,
                   activation='relu',
                   use_bias=True,
                   kernel_initializer=random_uniform(),
                   bias_initializer=random_uniform(),
                   input_shape=input_shape
                   )
            )
    mod.add(Conv2D(filters=128,
                   kernel_size=(3, 3),
                   padding='same',
                   dilation_rate=8,
                   activation='relu',
                   use_bias=True,
                   kernel_initializer=random_uniform(),
                   bias_initializer=random_uniform(),
                   input_shape=input_shape
                   )
            )
    mod.add(Conv2D(filters=256,
                   kernel_size=(3, 3),
                   padding='same',
                   dilation_rate=16,
                   activation='relu',
                   use_bias=True,
                   kernel_initializer=random_uniform(),
                   bias_initializer=random_uniform(),
                   input_shape=input_shape
                   )
            )
    mod.add(Conv2D(filters=num_classes,
                   kernel_size=(1, 1),
                   padding='same',
                   dilation_rate=4,
                   activation='softmax',
                   use_bias=True,
                   kernel_initializer=random_uniform(),
                   bias_initializer=random_uniform(),
                   input_shape=input_shape
                   )
            )

    return mod


def upscaled_truncated(input_shape, num_classes):
    """
    Builds a smaller network of upscaled convolutions, to fit on a gtx980ti, for testing purpose only.
    :param 
        input_shape: same as above. 
        num_classes: same as above
    :return
        mod: a keras model of the network.
    """
    mod = Sequential()

    mod.add(Conv2D(filters=16,
                   kernel_size=(3, 3),
                   padding='same',
                   dilation_rate=1,
                   activation='linear',
                   use_bias=True,
                   kernel_initializer=random_uniform(),
                   bias_initializer=random_uniform(),
                   input_shape=input_shape,
                   name='test'
                   )
            )
    mod.add(PReLU())
    mod.add(Conv2D(filters=32,
                   kernel_size=(3, 3),
                   padding='same',
                   dilation_rate=1,
                   activation='linear',
                   use_bias=True,
                   kernel_initializer=random_uniform(),
                   bias_initializer=random_uniform()
                   )
            )
    mod.add(PReLU())
    mod.add(Conv2D(filters=64,
                   kernel_size=(3, 3),
                   padding='same',
                   dilation_rate=2,
                   activation='linear',
                   use_bias=True,
                   kernel_initializer=random_uniform(),
                   bias_initializer=random_uniform()
                   )
            )
    mod.add(PReLU())
    mod.add(Conv2D(filters=64,
                   kernel_size=(3, 3),
                   padding='same',
                   dilation_rate=2,
                   activation='linear',
                   use_bias=True,
                   kernel_initializer=random_uniform(),
                   bias_initializer=random_uniform()
                   )
            )
    mod.add(PReLU())

    mod.add(Conv2D(filters=num_classes,
                   kernel_size=(1, 1),
                   padding='same',
                   dilation_rate=4,
                   activation='softmax',
                   use_bias=True,
                   kernel_initializer=random_uniform(),
                   bias_initializer=random_uniform()
                   )
            )

    return mod


# A dictionnary linking model builder names to the actual functions.
models_dict = {
    'simple_model': simple_model,
    'up': upscaled,
    'up_mini': upscaled_truncated

}
