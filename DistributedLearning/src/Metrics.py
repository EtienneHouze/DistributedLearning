from __future__ import absolute_import, print_function, division

import numpy as np
import tensorflow as tf

# TODO : il y a visiblement un pb avec cette ****** de fonction
def iou(y_true, y_pred):
    """
    Defines an Intersection over Union metrics
    :param y_true: a tensor of ground truth labels (4D)
    :param y_pred: a tensor of probabilites obtained after softmax. Must be of same shape as y_true
    :return: a scalar
    """
    numlabs = y_pred.get_shape()[-1].value
    y_pred = tf.argmax(input=y_pred,
                       axis=-1
                       )
    y_pred = tf.one_hot(indices=y_pred,
                        axis=-1,
                        depth=numlabs)
    intersection = tf.where(
            condition=tf.equal(y_true, y_pred),
            x=y_true,
            y=tf.zeros_like(y_true)
    )
    union = y_pred + y_true - intersection
    nd = intersection.get_shape().ndims
    TP = tf.reduce_sum(intersection,
                       axis=np.arange(start=0, stop=nd - 1, step=1)
                       )
    Neg = tf.minimum(tf.reduce_sum(union,
                                   axis=np.arange(start=0, stop=nd - 1, step=1)
                                   ),
                     1)
    return tf.reduce_mean(TP / Neg)


# Dictionary renferencing metrics

met_dict = {
    'iou': iou
}
