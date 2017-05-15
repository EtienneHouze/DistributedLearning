from __future__ import print_function, absolute_import, division
import json
import os, sys
from helpers import models
from helpers.BatchGenerator import BatchGenerator
import keras
from keras.models import Model, load_model, Sequential

class CityScapeModel:
    """
        A high-level, user friendly interface to create dense labelling model based on the Keras API.
        Properties :
            - name : a string, name of the model.
            - net_builder : the function to build model of the neural network
            - input_shape : a tuple of the size of the input
            - numlabs = number of classes.
            - dir = path to the directory where the model is located
        """
#==============================================================================
    def __init__(self,dir = './default_model'):
        """
            Initialisation method, creates an object reading its properties stored in the 'properties.json' file in the directory.
            If no directory exists, a new one is created.
        """

        self.model = None
        if ( not os.path.isdir(dir)):
            print("Model directory did not exist, creating from scratch")
            #Creating directories
            os.mkdir(dir)
            os.mkdir(os.path.join(dir,"saves"))
            os.mkdir(os.path.join(dir,"logs"))
            #Initializing the dictionnary
            self.prop_dict = { 'name' : 'default',
                              'net_builder' : None,
                              'directory' : dir,
                              'loss' : keras.losses.MAE,
                              'opt' : keras.optimizers.Adam(),
                              'input_shape' : None,
                              'num_labs' : None,
                              'met' : None,
                              'w_mode' : None
                              }
            #Saving default in a file 
            with open(os.path.join(dir,'properties.json'), 'w') as outfile:
                json.dump(self.prop_dict,outfile)
        else:
            self.prop_dict = json.load(open(os.path.join(dir,'properties.json')))
            self.build_net()
            print("Model loaded from .json file")
            
#==============================================================================

    def define_network(self,building_function=None,in_shape=None,out_shape=None):
        """
            Define the neural network building function from the string passed as argument.
            The string must designate a valid network builder function, see 'Network.py' for more info
        """
        if (not (building_function in models.models_dict.keys())):
            print("Please specify a valid building function")
            #self.prop_dict['net_builder'] = None
        else:
            print("Defining building function")
            self.prop_dict['net_builder']= building_function


    def save_model(self):
        """
            Saves the current model properties into a json file
        """
        with (open(os.path.join(self.prop_dict['directory'], 'properties.json'), 'w')) as outfile:
            json.dump(self.prop_dict,outfile)


    def build_net(self):
        """
            Builds the network using the defined function
        """
        if (self.prop_dict['net_builder']):
            print (' Building network from function : ' + self.prop_dict['net_builder'])
            self.model = models.models_dict[self.prop_dict['net_builder']](self.prop_dict['input_shape'], self.prop_dict['numlabs'])
        else:
            print ('Error : no building function defined')

    def define_name(self, name = 'default'):
        """
            Changes the name of the model as defined
        """
        self.prop_dict['name'] = name

    def define_input(self,shape):
        self.prop_dict['input_shape'] = shape

    def define_numlabs(self, numlabs):
        self.prop_dict['numlabs'] = numlabs

    #DEPRECATED
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


    def print_net(self):
        """
            Print the network in the console.
        """
        self.model.summary()

    def save_tojson(self):
        """
            Saves the property dictionnary into a 'properties.json' file
        """
        with open(os.path.join(self.prop_dict['directory'],'properties.json'), 'w') as outfile:
                json.dump(self.prop_dict,outfile)

    def save_net(self,weights_only=False):
        """
            Save the network and its weights
        """
        if (not weights_only):
            self.model.save(os.path.join(self.prop_dict['directory'],'saves','net_'))
        else:
            self.model.save_weights(os.path.join(self.prop_dict['directory'],'saves','weights_'))
     
    def freeze_layers_with_name(self, name):
        """
            Freeze the layers whose names contain the 'name' string for learning.
        """
        for layer in self.model.layers:
            if (name in layer.name):
                layer.trainable = False

    def unfreeze_all(self):
        for layer in self.model.layers:
            layer.trainable = True
     
    def compile(self):
        self.model.compile(optimizer = self.prop_dict['opt'],
                           loss = self.prop_dict['loss'],
                           metrics = self.prop_dict['met'],
                           sample_weight_mode = self.prop_dict['w_mode'])

    def train(self, dataset, epochs, batch_size, layer, loss = 'categorical_crossentropy', opt = keras.optimizers.Adam(), save = True):
        print('compiling')
        self.compile(None,opt,loss)
        gen = BatchGenerator(dataset[0],dataset[1],self,batch_size)
        self.models.layers[layer].fit_generator(gen.generate_batch(layer),steps_per_epoch = dataset[0][:,0,0,0].size//batch_size, epochs = epochs, verbose = 2)
        if (save):
            self.save_model()
            self.save_net()
        print ('done')

