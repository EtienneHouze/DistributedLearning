from __future__ import absolute_import,print_function,division

from keras.initializers import random_uniform, zeros
from keras.activations import relu
from keras.layers import Input, Conv2D
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential, Model

from src.Layers import Inception

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

def upscaled_without_aggreg(input_shape, num_classes):

    ins = Input(shape=input_shape,
                name='net_inputs')
    a = Conv2D(
            filters=16,
            kernel_size=(3,3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=1,
            activation='relu',
            name='net_conv0'
            )(ins)
    a = Conv2D(
            filters=32,
            kernel_size=(3,3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=1,
            activation='relu',
            name = 'net_conv1'
            )(a)
    a = Conv2D(
            filters=64,
            kernel_size=(3,3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=2,
            activation='relu',
            name = 'net_conv2'
            )(a)
    a = Conv2D(
            filters=64,
            kernel_size=(3,3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=2,
            activation='relu',
            name = 'net_conv3'
            )(a)
    a = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=4,
            activation='relu',
            name='net_conv4'
    )(a)
    a = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=4,
            activation='relu',
            name='net_conv5'
    )(a)
    a = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=8,
            activation='relu',
            name='net_conv6'
    )(a)
    a = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=8,
            activation='relu',
            name='net_conv7'
    )(a)
    a = Conv2D(
            filters=256,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=16,
            activation='relu',
            name='net_conv8'
    )(a)
    a = Conv2D(
            filters=num_classes,
            kernel_size=1,
            kernel_initializer=random_uniform(),
            use_bias=False,
            activation='softmax',
            padding='same',
            name = 'net_out'
    )(a)

    mod = Model(
            inputs = ins,
            outputs = a,
        )

    return mod

def upscaled_with_aggreg(input_shape, num_classes):
    ins = Input(shape=input_shape,
                name='net_inputs')
    a = Conv2D(
            filters=16,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=1,
            activation='relu',
            name='net_conv0'
    )(ins)
    a = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=1,
            activation='relu',
            name='net_conv1'
    )(a)
    a = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=2,
            activation='relu',
            name='net_conv2'
    )(a)
    a = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=2,
            activation='relu',
            name='net_conv3'
    )(a)
    a = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=4,
            activation='relu',
            name='net_conv4'
    )(a)
    a = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=4,
            activation='relu',
            name='net_conv5'
    )(a)
    a = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=8,
            activation='relu',
            name='net_conv6'
    )(a)
    a = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=8,
            activation='relu',
            name='net_conv7'
    )(a)
    a = Conv2D(
            filters=256,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=16,
            activation='relu',
            name='net_conv8'
    )(a)
    a = Conv2D(
            filters=num_classes,
            kernel_size=1,
            kernel_initializer=random_uniform(),
            use_bias=False,
            activation='softmax',
            padding='same',
            name='net_out'
    )(a)

    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_0'
    )(a)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_1'
    )(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            dilation_rate=2,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_2'
    )(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            dilation_rate=4,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_3'
    )(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            dilation_rate=8,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_4'
    )(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            dilation_rate=16,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_5'
    )(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            dilation_rate=1,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_6'
    )(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=1,
            dilation_rate=1,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='softmax',
            padding='same',
            name='aggreg_7'
    )(b)

    mod = Model(
        inputs=ins,
        outputs=b
    )
    return mod

def upscaled_with_deeper_aggreg(input_shape, num_classes):
    ins = Input(shape=input_shape,
                name='net_inputs')
    a = Conv2D(
            filters=16,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=1,
            activation='relu',
            name='net_conv0'
    )(ins)
    a = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=1,
            activation='relu',
            name='net_conv1'
    )(a)
    a = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=2,
            activation='relu',
            name='net_conv2'
    )(a)
    a = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=2,
            activation='relu',
            name='net_conv3'
    )(a)
    a = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=4,
            activation='relu',
            name='net_conv4'
    )(a)
    a = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=4,
            activation='relu',
            name='net_conv5'
    )(a)
    a = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=8,
            activation='relu',
            name='net_conv6'
    )(a)
    a = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=8,
            activation='relu',
            name='net_conv7'
    )(a)
    a = Conv2D(
            filters=256,
            kernel_size=(3, 3),
            kernel_initializer=random_uniform(),
            use_bias=True,
            bias_initializer=zeros(),
            padding='same',
            dilation_rate=16,
            activation='relu',
            name='net_conv8'
    )(a)
    a = Conv2D(
            filters=num_classes,
            kernel_size=1,
            kernel_initializer=random_uniform(),
            use_bias=False,
            activation='softmax',
            padding='same',
            name='net_out'
    )(a)

    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_0'
    )(a)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_1'
    )(b)
    b = Conv2D(
            filters=2*num_classes,
            kernel_size=3,
            dilation_rate=2,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_2'
    )(b)
    b = Conv2D(
            filters=4*num_classes,
            kernel_size=3,
            dilation_rate=4,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_3'
    )(b)
    b = Conv2D(
            filters=8*num_classes,
            kernel_size=3,
            dilation_rate=8,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_4'
    )(b)
    b = Conv2D(
            filters=8*num_classes,
            kernel_size=3,
            dilation_rate=16,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_5'
    )(b)
    b = Conv2D(
            filters=8*num_classes,
            kernel_size=3,
            dilation_rate=1,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_6'
    )(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=1,
            dilation_rate=1,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='softmax',
            padding='same',
            name='aggreg_7'
    )(b)

    mod = Model(
        inputs=ins,
        outputs=b
    )
    return mod

def test_inception(input_shape,num_classes):
    mod = Sequential()

    mod.add(Inception(input_shape=input_shape,
                      output_depth=num_classes)
            )
    return mod

def inception_with_deeper_aggreg(input_shape, num_classes):
    inputs = Input(input_shape,
                   name='net_ins')
    a = Inception(input_shape=input_shape,
                  output_depth=16,
                  dilation_rate=(1, 1),
                  name='net_incept_1'
                  )(inputs)
    current_shape = input_shape[:-1] + (16,)
    a = Inception(input_shape=current_shape,
                  output_depth=32,
                  dilation_rate=(2,2),
                  name='net_incept_2'
                  )(a)
    current_shape=current_shape[:-1]+(32,)
    a = Inception(input_shape=current_shape,
                  output_depth=64,
                  dilation_rate=(4, 4),
                  name='net_incept_3'
                  )(a)
    current_shape=current_shape[:-1]+(64,)
    a = Inception(input_shape=current_shape,
                  output_depth=128,
                  dilation_rate=(8,8),
                  name='net_incept_4'
                  )(a)
    current_shape=current_shape[:-1]+(64,)
    a = Inception(input_shape=current_shape,
                  output_depth=128,
                  dilation_rate=(16,16),
                  name='net_incept_5'
                  )(a)
    current_shape=current_shape[:-1]+(128,)
    a = Inception(input_shape=current_shape,
                  output_depth=256,
                  dilation_rate=(32, 32),
                  name='net_incept_6'
                  )(a)
    current_shape=current_shape[:-1]+(256,)
    a = Inception(input_shape=current_shape,
                  output_depth=num_classes,
                  dilation_rate=(32, 32),
                  name='net_incept_6'
                  )(a)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_0'
    )(a)
    b = Conv2D(
            filters=num_classes,
            kernel_size=3,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_1'
    )(b)
    b = Conv2D(
            filters=2 * num_classes,
            kernel_size=3,
            dilation_rate=2,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_2'
    )(b)
    b = Conv2D(
            filters=4 * num_classes,
            kernel_size=3,
            dilation_rate=4,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_3'
    )(b)
    b = Conv2D(
            filters=8 * num_classes,
            kernel_size=3,
            dilation_rate=8,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_4'
    )(b)
    b = Conv2D(
            filters=8 * num_classes,
            kernel_size=3,
            dilation_rate=16,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_5'
    )(b)
    b = Conv2D(
            filters=8 * num_classes,
            kernel_size=3,
            dilation_rate=1,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='relu',
            padding='same',
            name='aggreg_6'
    )(b)
    b = Conv2D(
            filters=num_classes,
            kernel_size=1,
            dilation_rate=1,
            use_bias=False,
            kernel_initializer=random_uniform(),
            activation='softmax',
            padding='same',
            name='aggreg_7'
    )(b)
    mod = Model(
            inputs=inputs,
            outputs=b
    )
    return mod

# A dictionnary linking model builder names to the actual functions.
models_dict = {
    'simple_model': simple_model,
    'up': upscaled,
    'up_mini': upscaled_truncated,
    'up_without': upscaled_without_aggreg,
    'up_with': upscaled_with_aggreg,
    'up_with_deeper_aggreg': upscaled_with_deeper_aggreg,
    'test_inception': test_inception,
    'inception_with': inception_with_deeper_aggreg
}
