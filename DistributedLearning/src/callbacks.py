from keras.callbacks import Callback
# TODO : Create callback functions to do stuff...

class ViewOutput(Callback):
    def __init__(self, citymodel):
        super(ViewOutput, self).__init__()
        self.citymodel = citymodel

    def on_train_begin(self, logs={}):

    def on_batch_end(self, batch, logs={}):

    def on_epoch_begin(self, logs={}):
