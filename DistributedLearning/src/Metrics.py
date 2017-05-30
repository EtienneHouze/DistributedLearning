from __future__ import absolute_import, print_function, division

import numpy as np
import tensorflow as tf
import keras.backend as K
# from sklearn.metrics import jaccard_similarity_score
# TODO : Trouver un moyen de passer la valeur du nombre de labels en évitant d'appeler le tensuer
def iou(y_true, y_pred):
    """
    
    Args:
        y_true (4D-tensor): ground truth label 
        y_pred (4D-tensor): output of the network (after softmax)

    Returns:
        The value of the mean IOU loss
    """
    # numlabs = y_pred.get_shape()[-1].value
    numlabs = 19
    y_pred = tf.argmax(input=y_pred,
                       axis=-1
                       )
    y_pred = tf.one_hot(indices=y_pred,
                        axis=-1,
                        depth=numlabs)
    equality = tf.cast(tf.equal(y_true, y_pred),
                       dtype=tf.float32)
    intersection = tf.multiply(y_true,equality)
    union = y_pred + y_true - intersection
    nd = intersection.get_shape().ndims
    TP = tf.reduce_sum(intersection,
                       axis=np.arange(start=0, stop=nd - 1, step=1)
                       )
    Neg = tf.maximum(tf.reduce_sum(union,
                                   axis=np.arange(start=0, stop=nd - 1, step=1)
                                   ),
                     1)

    return tf.reduce_mean(TP/Neg,axis=-1)

class Metrics():
    def __init__(self, citymodel):
        self.citymodel = citymodel

    def iou(self, y_true, y_pred):
        """

        Args:
            y_true (4D-tensor): ground truth label
            y_pred (4D-tensor): output of the network (after softmax)

        Returns:
            The value of the mean IOU loss
        """
        # numlabs = y_pred.get_shape()[-1].value
        numlabs = self.citymodel.prop_dict['num_labs']+1
        y_pred = tf.argmax(input=y_pred,
                           axis=-1
                           )
        y_pred = tf.one_hot(indices=y_pred,
                            axis=-1,
                            depth=numlabs)
        equality = tf.cast(tf.equal(y_true, y_pred),
                           dtype=tf.float32)
        intersection = tf.multiply(y_true, equality)
        union = y_pred + y_true - intersection
        nd = intersection.get_shape().ndims
        TP = tf.reduce_sum(intersection,
                           axis=np.arange(start=0, stop=nd - 1, step=1)
                           )
        Neg = tf.maximum(tf.reduce_sum(union,
                                       axis=np.arange(start=0, stop=nd - 1, step=1)
                                       ),
                         1)

        return tf.reduce_mean(TP / Neg, axis=-1)
# Dictionary renferencing metrics

met_dict = {
    'iou': iou
}

def create_metrics(metricname, citymodel):
    met = Metrics(citymodel)
    fun = None
    if metricname == 'iou':
        fun = met.iou
    return fun
