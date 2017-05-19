from __future__ import absolute_import, print_function, division

import keras.backend as K
from keras.layers import Layer

# TODO : voir implémentation du bias pour le layer...
class Inception(Layer):
    """
    Defines a inception layer. Weights are not optimized due to the hight memory cost of the architecture.
    :properties : 
    """
    def __init__(self, output_depth, **kwargs):
        self.output_depth = output_depth
        self.use_bias = False
        self.dilation_rate = (1,1)
        if 'use_bias' in kwargs.keys():
            self.use_bias=kwargs.pop('use_bias')
        if 'dilation_rate' in kwargs.keys():
            self.dilation_rate = kwargs.pop('dilation_rate')
        super(Inception,self).__init__(**kwargs)

    def build(self, input_shape):
        self.K1 = self.add_weight(
                shape=[3, 3, input_shape[-1], self.output_depth],
                initializer='uniform',
                trainable=True,
                name = self.name + '_K1'
        )
        self.K2 = self.add_weight(
                shape=[3, 3, self.output_depth, self.output_depth],
                initializer='uniform',
                trainable=True,
                name=self.name+'_K2'
        )
        self.K3=self.add_weight(
                shape=[3,3,input_shape[-1], self.output_depth],
                initializer='uniform',
                trainable=True,
                name=self.name+'_K3'
        )
        self.K4=self.add_weight(
                shape=[1, 1, input_shape[-1], self.output_depth],
                trainable=True,
                initializer='uniform',
                name=self.name+'_K4'
        )
        super(Inception,self).build(input_shape)

    def call(self, inputs, **kwargs):
        tower1 = K.conv2d(x=inputs,
                          kernel=self.K1,
                          strides=(1, 1),
                          padding='same',
                          dilation_rate=self.dilation_rate
                          )
        tower1 = K.conv2d(x=tower1,
                          kernel=self.K2,
                          strides=(1, 1),
                          padding='same',
                          dilation_rate=self.dilation_rate
                          )
        tower2 = K.conv2d(x=inputs,
                          kernel=self.K3,
                          strides=(1, 1),
                          padding='same',
                          dilation_rate=self.dilation_rate
                          )
        tower3 = K.pool2d(x=inputs,
                          pool_size=(3, 3),
                          strides=(1, 1),
                          padding='same',
                          pool_mode='max'
                          )
        tower3 = K.conv2d(x=tower3,
                          kernel=self.K4,
                          strides=(1,1),
                          padding='same',
                          dilation_rate=(1, 1)
                          )
        output = tower3+tower2+tower1
        return K.relu(output)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.output_depth,)

class InceptionConcat(Layer):
    """
    Defines a inception layer. Weights are not optimized due to the hight memory cost of the architecture.
    :properties : 
    """
    def __init__(self, output_depth, **kwargs):
        self.output_depth = output_depth
        self.mid_depth = self.output_depth // 4
        self.use_bias = False
        self.dilation_rate = (1,1)
        if 'use_bias' in kwargs.keys():
            self.use_bias=kwargs.pop('use_bias')
        if 'dilation_rate' in kwargs.keys():
            self.dilation_rate = kwargs.pop('dilation_rate')
        if 'mid_depth' in kwargs.keys():
            self.mid_depth = kwargs.pop('mid_depth')
        super(Inception,self).__init__(**kwargs)

    def build(self, input_shape):
        self.K1 = self.add_weight(
                shape=[3, 3, input_shape[-1], self.mid_depth],
                initializer='uniform',
                trainable=True,
                name = self.name + '_K1'
        )
        self.K2 = self.add_weight(
                shape=[3, 3, self.mid_depth, self.mid_depth],
                initializer='uniform',
                trainable=True,
                name=self.name+'_K2'
        )
        self.K3=self.add_weight(
                shape=[3,3,input_shape[-1], self.mid_depth],
                initializer='uniform',
                trainable=True,
                name=self.name+'_K3'
        )
        self.K4=self.add_weight(
                shape=[1, 1, input_shape[-1], self.mid_depth],
                trainable=True,
                initializer='uniform',
                name=self.name+'_K4'
        )
        self.K5 = self.add_weight(
                shape=[1,1,self.mid_depth*3,self.output_depth],
                trainable=True,
                initializer='uniform',
                name=self.name+'_K5'
        )
        super(Inception,self).build(input_shape)

    def call(self, inputs, **kwargs):
        tower1 = K.conv2d(x=inputs,
                          kernel=self.K1,
                          strides=(1, 1),
                          padding='same',
                          dilation_rate=self.dilation_rate
                          )
        tower1 = K.conv2d(x=tower1,
                          kernel=self.K2,
                          strides=(1, 1),
                          padding='same',
                          dilation_rate=self.dilation_rate
                          )
        tower2 = K.conv2d(x=inputs,
                          kernel=self.K3,
                          strides=(1, 1),
                          padding='same',
                          dilation_rate=self.dilation_rate
                          )
        tower3 = K.pool2d(x=inputs,
                          pool_size=(3, 3),
                          strides=(1, 1),
                          padding='same',
                          pool_mode='max'
                          )
        tower3 = K.conv2d(x=tower3,
                          kernel=self.K4,
                          strides=(1,1),
                          padding='same',
                          dilation_rate=(1, 1)
                          )
        output = K.concatenate(tower1,tower2,tower3)
        output = K.conv2d(x=output,
                          kernel=self.K5,
                          strides=(1,1),
                          padding='same',
                          dilation_rate=(1,1)
                          )
        return K.relu(output)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.output_depth,)
