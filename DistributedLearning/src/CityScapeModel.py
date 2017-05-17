from __future__ import print_function, absolute_import, division

import json
import os
import keras
import numpy as np

from helpers import models, preprocess
from helpers.BatchGenerator import BatchGenerator
from src import callbacks

# TODO : si j'ai le temps, revoir les noms...
class CityScapeModel:
    """
        A high-level, user friendly interface to create dense labelling model based on the Keras API.
        Properties (stored in prop_dict):
            - name : a string, name of the model.
            - net_builder : the function to build model of the neural network
            - input_shape : a tuple of the size of the input
            - num_labs = number of classes.
            - directory = path to the directory where the model is located
            - opt : a keras.Optimizer function to use for training
            - loss : a keras.losses function to use for training
            - met : a dictionnary of metrics, used in training, see keras doc
            - w_mode : sample weight mode, see documentation for keras training
            - trainset : a list [set_builder, folder, trainsize] with
                * set_builder : a batch generator.
                * folder : the path to the folder containing all training images.
                * trainsize : the number of image to put in the training set.
            - callbacks : a list of tuple (function, args) with:
                * function : name of a callback function as defined in the helper file
                * args : a dict of the arguments to pass to the function
        """

    # ==============================================================================

    def __init__(self, dir='./default_model'):
        """
            Initialisation method, creates an object reading its properties stored in the 'properties.json' file in the directory.
            If no directory exists, a new one is created.
        """

        self.model = None
        if (not os.path.isdir(dir)):
            print("Model directory did not exist, creating from scratch")
            # Creating directories
            os.mkdir(dir)
            os.mkdir(os.path.join(dir, "saves"))
            os.mkdir(os.path.join(dir, "logs"))
            os.mkdir(os.path.join(dir, 'watch'))
            # Initializing the dictionnary
            self.prop_dict = {'name': 'default',
                              'net_builder': None,
                              'directory': dir,
                              'loss': 'MAE',
                              'opt': 'Adam',
                              'input_shape': None,
                              'num_labs': None,
                              'met': None,
                              'w_mode': None,
                              'trainset': [],
                              'callbacks' : []
                              }
            # Saving default in a file
            with open(os.path.join(dir, 'properties.json'), 'w') as outfile:
                json.dump(self.prop_dict, outfile)
        else:
            self.prop_dict = json.load(open(os.path.join(dir, 'properties.json')))
            print("Model loaded from .json file")
            if (os.path.isfile(os.path.join(dir,'saves','net_.h5'))):
                print ( "Saving file found, restoring network...")
                self.model = keras.models.load_model(os.path.join(dir,'saves','net_.h5'))
            else:
                print ("No save found, building model from function")
                self.build_net()

            # ==============================================================================

    # ==============================================================================
    # Setters
    # ==============================================================================

    def define_network(self, building_function=None, in_shape=None, out_shape=None):
        """
            Define the neural network building function from the string passed as argument.
            The string must designate a valid network builder function, see 'Network.py' for more info
        """
        if (not (building_function in models.models_dict.keys())):
            print("Please specify a valid building function")
            # self.prop_dict['net_builder'] = None
        else:
            print("Defining building function")
            self.prop_dict['net_builder'] = building_function

    def define_loss(self,loss_name):
        """
        :param 
            loss_name: name of the loss to use. Must be a valid keras loss. 
        """
        self.prop_dict['loss'] = loss_name

    def define_name(self, name='default'):
        """
            Changes the name of the model as defined
        """
        self.prop_dict['name'] = name

    def define_input(self, shape):
        """
            Sets the input shape for the model. Argument 'shape' must be a tuble or list of 3 integers.
        """
        self.prop_dict['input_shape'] = shape

    def define_numlabs(self, numlabs):
        """
            Sets the number of labels for the model.
            WARNING : since there is an unlabelled class, there is actually one more class than labels !
        """
        self.prop_dict['num_labs'] = numlabs

    def define_training_set(self, trainset, trainsetbuilder, trainsize):
        """
            Sets the 'trainset' field of the properties dict to the specified arguments.
        """
        self.prop_dict['trainset'] = [trainsetbuilder, trainset, trainsize]

    def add_callback(self, function_name, **kwargs):
        """
        Adds a callback function
        """
        self.prop_dict['callbacks'].append([function_name, kwargs])

    # =============================================================================
    # Other functions
    # =============================================================================

    def build_net(self):
        """
            Builds the network using the defined function
        """
        if (self.prop_dict['net_builder']):
            print(' Building network from function : ' + self.prop_dict['net_builder'])
            self.model = models.models_dict[self.prop_dict['net_builder']](self.prop_dict['input_shape'],
                                                                           self.prop_dict['num_labs']+1)
        else:
            print('Error : no building function defined')


    # DEPRECATED
    """
        def add_network_from_builder(self, building_function,in_shape=None,out_shape = None):
                Adds a network from the builder function name provided, with corresponding input and output shapes.
                The list (building_function,in_shape,out_shape) is the appended to the 'net_builder' key of the dictionnary.
            if (building_function in models.models_dict.keys()):
                if (not in_shape):
                    in_shape = self.prop_dict['input_shape']
                if (not out_shape) :
                    out_shape = self.prop_dict['output_shape']
                self.models.append(models.models_dict[building_function](input_shape = in_shape, num_classes = out_shape[-1]))
                self.prop_dict['net_builder'].append([building_function,in_shape,out_shape])
            else:
                print("Please enter valid building function")
    """

    def print_model(self):
        """
            Prints the properties dictionnary in the console.
        """
        print(self.prop_dict)

    def print_net(self):
        """
            Print the network in the console.
        """
        self.model.summary()

    def save_tojson(self):
        """
            Saves the property dictionnary into a 'properties.json' file
        """
        with open(os.path.join(self.prop_dict['directory'], 'properties.json'), 'w') as outfile:
            json.dump(self.prop_dict, outfile)

    def save_net(self, weights_only=False):
        """
            Save the network and its weights
        """
        if (not weights_only):
            self.model.save(os.path.join(self.prop_dict['directory'], 'saves', 'net_.h5'))
        else:
            self.model.save_weights(os.path.join(self.prop_dict['directory'], 'saves', 'weights_.h5'))

    def freeze_layers_with_name(self, name):
        """
            Freeze the layers whose names contain the 'name' string for learning.
        """
        for layer in self.model.layers:
            if (name in layer.name):
                layer.trainable = False
                print("Layer "+ layer + " is frozen for training.")
        self.compile()

    def unfreeze_all(self):
        """
            Unfreeze every layer in the model, allowing their weights to be learnt in the next training process.
        """
        for layer in self.model.layers:
            layer.trainable = True
        print("All layers unfrozen.")
        self.compile()

    def compile(self):
        """
            Compiles the embedded Keras model using the optimizer, loss, metrics and weight options defined in the properties dictionnary of the object.
        """
        self.model.compile(optimizer=self.prop_dict['opt'],
                           loss=self.prop_dict['loss'],
                           metrics=self.prop_dict['met'],
                           sample_weight_mode=self.prop_dict['w_mode'])

    def load_weights(self, filepath):
        self.model.load_weights(filepath,by_name=True)

    def train(self, epochs, batch_size, save=True):
        """
            Trains the neural network according to the values passed as arguments.
            @ Args:
                - epochs : an int, number of epochs to train on.
                - batch_size : an int, size of the batch to use.
                - save : a bool, whether to save the model at the end of training or not.
        """

        # First, we compile the model
        print('compiling')
        self.compile()
        # We the build the callback functions, distinguishes cases between built-in callbacks and custom callbacks.
        print("Building Callback functions...")
        call_list = []
        for call_def in self.prop_dict['callbacks']:
            if call_def[0] == 'tensorboard':
                call = keras.callbacks.TensorBoard(log_dir=os.path.join(self.prop_dict['directory'],'logs',self.prop_dict['name']),
                                                   histogram_freq=1,
                                                   write_graph=True
                                                   )
            elif (call_def[0] == 'csv'):
                call = keras.callbacks.CSVLogger(filename=os.path.join(self.prop_dict['directory'],'logs',self.prop_dict['name']+'.csv'),
                                                 separator=',',
                                                 append=True
                                                 )
            elif call_def[0] == 'ckpt':
                call = keras.callbacks.ModelCheckpoint(filepath=os.path.join(self.prop_dict['directory'],'saves'))
            else:
                call = callbacks.callbacks_dict[call_def[0]](self,
                                                             options=call_def[1]
                                                             )
            call_list.append(call)
        batch_gen = BatchGenerator(traindir=self.prop_dict['trainset'][1],
                                   city_model = self,
                                   trainsetsize = self.prop_dict['trainset'][2],
                                   batchsize = batch_size)
        self.model.fit_generator(generator=batch_gen.generate_batch(option=self.prop_dict['trainset'][0]),
                                 steps_per_epoch=batch_gen.epoch_size,
                                 epochs=epochs,
                                 verbose=2,
                                 callbacks=call_list
                                 )
        if (save):
            print('Saving model')
            self.save_tojson()
            self.save_net()
            self.save_net(weights_only=True)
        print('done')

    def compute_output(self, x):
        """
        Computes the output of the net on a single input x.
        :param 
            x: a 3D np array containing the input 
        :return
            y: a 3D np array containing the predictions of the net.
        """
        y = self.model.predict_on_batch(np.expand_dims(x, axis = 0))
        y = np.argmax(y, axis = -1)
        return  y