from os.path import join

from PIL import Image
from helpers.BatchGenerator import BatchGenerator
from keras.callbacks import Callback
import numpy as np


# TODO : Create callback functions to do stuff...

class ViewOutput(Callback):
    """
    Allows user to save an output of the network every specified step as a png file.
    """

    def __init__(self, citymodel, options):
        """
        :param 
            citymodel: the cityscape model 
            options: dictionnary of the options. Can contain :
                * 'batch_interval' : an int, how ofter we want to save an image. Default is 0.
                * 'num_ins' : number of inputs to compute. For now, only 1 is relevant.
                * 'on_epoch' : a bool, whether to print on every epoch or not. Default is True.
        """
        super(ViewOutput, self).__init__()
        self.citymodel = citymodel
        self.batch_interval = 0
        self.i = 0
        self.epoch_counter = 0
        self.num_ins = 1
        self.on_epoch = True
        self.gen = None
        self.x = None
        if ('batch_interval' in options.keys()):
            self.batch_interval = options['batch_interval']
        if ('num_ins' in options.keys()):
            self.num_ins = options['num_ins']
        if ('on_epoch' in options.keys()):
            self.on_epoch = options['on_epoch']

    def on_train_begin(self, logs={}):
        self.gen = BatchGenerator(traindir=self.citymodel.prop_dict['trainset'][1],
                                  city_model=self.citymodel,
                                  trainsetsize=self.citymodel.prop_dict['trainset'][2],
                                  batchsize=self.num_ins)
        self.x, _ = next(self.gen.generate_batch(option=self.citymodel.prop_dict['trainset'][0]))
        bob = np.squeeze(self.x,axis=0)
        bob = bob[:,:,0:3].astype(np.uint8)
        Input = Image.fromarray(bob)
        Input.save(
            join(self.citymodel.prop_dict['directory'], 'watch', self.citymodel.prop_dict['name'] + '_input.png'))

    def on_batch_end(self, batch, logs={}):
        if self.batch_interval != 0:
            self.i = (self.i + 1) % self.batch_interval
            if self.i == 0:
                y = self.citymodel.compute_output(np.squeeze(self.x,axis=0))
                Output = Image.fromarray(np.squeeze(y[:, :, :],axis=0).astype(np.uint8))
                Output.save(join(self.citymodel.prop_dict['directory'], 'watch',
                                 self.citymodel.prop_dict['name'] + '_output_' + str(self.epoch_counter) + '_' + str( self.i) + '_.png'))

    def on_epoch_end(self, epoch, logs={}):
        if (self.on_epoch):
            y = self.citymodel.compute_output(self.x)
            Output = Image.fromarray(y[0, :, :, 0])
            Output.save(join(self.citymodel.prop_dict['directory'], 'watch',
                             self.citymodel.prop_dict['name'] + '_output_epoch_' + str(self.epoch_counter) + '_.png'))
        self.epoch_counter += 1


callbacks_dict = {
    'view_output': ViewOutput
}