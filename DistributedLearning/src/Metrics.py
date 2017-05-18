from __future__ import absolute_import, print_function, division

import numpy as np
import tensorflow as tf


def iou(y_true, y_pred):
    numlabs = y_pred.get_shape()[-1].value
    y_pred = tf.argmax(input=y_pred,
                       axis=-1
                       )
    y_pred = tf.one_hot(indices=y_pred,
                        axis=-1,
                        depth=numlabs)
    intersection = tf.where(
            condition=tf.equal(y_true, y_pred),
            x = y_true,
            y = tf.zeros_like(y_true)
    )
    union = y_pred+y_true-intersection
    nd = intersection.get_shape().ndims
    TP = tf.reduce_sum(intersection,
                       axis=np.arange(start=0, stop=nd - 1, step=1)
                       )
    Neg = tf.reduce_sum(union,
                        axis=np.arange(start=0, stop=nd - 1, step=1)
                        )
    return tf.reduce_mean(TP / (Neg-TP))


# Dictionary renferencing metrics

met_dict = {
    'iou': iou
}