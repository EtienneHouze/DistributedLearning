from __future__ import print_function, absolute_import, division
import json
import os, sys
from helpers import models
import keras
from keras.models import Model, load_model, Sequential

class CityScapeModel:
    """
        A high-level, user friendly interface to create dense labelling model based on the Keras API.
        Properties :
            - name : a string, name of the model.
            - models : a list of Keras models, which the data will flow trhought sequentially.
        """
#==============================================================================
    def __init__(self,dir = './default_model'):
        """
            Initialisation method, creates an object reading its properties stored in the 'properties.json' file in the directory.
            If no directory exists, a new one is created.
        """

        self.prop_dict = {'directory':dir}
        self.models = []
        if ( not os.path.isdir(dir)):
            print("Model directory did not exist, creating from scratch")
            os.mkdir(dir)
            os.mkdir(os.path.join(dir,"saves"))
            os.mkdir(os.path.join(dir,"logs"))
            self.prop_dict['name'] = 'default'
            self.prop_dict['net_builders'] = []
            with open(os.path.join(dir,'properties.json'), 'w') as outfile:
                json.dump(self.prop_dict,outfile)
        else:
            self.prop_dict = json.load(open(os.path.join(dir,'properties.json')))
            print("Model loaded from .json file")
            for builder in self.prop_dict['net_builders']:
                print ("building network from " + builder[0] +"...")
                self.models.append(models.models_dict[builder[0]](input_shape = builder[1], num_classes = builder[2][-1]))
#==============================================================================
    def define_network(self,building_function=None,in_shape=None,out_shape=None):
        """
            Define the neural network building function from the string passed as argument.
            The string must designate a valid network builder function, see 'Network.py' for more info
        """
        if (not (building_function in models.models_dict.keys())):
            print("Please specify a valid building function")
            self.prop_dict['net_builders'] = []
        else:
            print("Defining building function")
            self.prop_dict['net_builders'].append([building_function,in_shape,out_shape])

    def define_name(self, name = 'default'):
        """
            Changes the name of the model as defined
        """
        self.prop_dict['name'] = name

    def define_input(self,shape):
        self.prop_dict['input_shape'] = shape

    def define_output(self, output):
        self.prop_dict['output_shape'] = output

    def add_network_from_builder(self, building_function,in_shape=None,out_shape = None):
        """
            Adds a network from the builder function name provided, with corresponding input and output shapes.
            The list (building_function,in_shape,out_shape) is the appended to the 'net_builders' key of the dictionnary.
        """
        if (building_function in models.models_dict.keys()):
            if (not in_shape):
                in_shape = self.prop_dict['input_shape']
            if (not out_shape) :
                out_shape = self.prop_dict['output_shape']
            self.models.append(models.models_dict[building_function](input_shape = in_shape, num_classes = out_shape[-1]))
            self.prop_dict['net_builders'].append([building_function,in_shape,out_shape])
        else:
            print("Please enter valid building function")

    def print_net(self, net_index=None):
        """
            Prints the specified net (or all if no index is passed) in the console.
        """
        if (not net_index):
            for mod in self.models:
                mod.summary()
        else:
            self.models[net_index].summary()  

    def save_tojson(self):
        """
            Saves the property dictionnary into a 'properties.json' file
        """
        with open(os.path.join(self.prop_dict['directory'],'properties.json'), 'w') as outfile:
                json.dump(self.prop_dict,outfile)

    def save_net(self,index=None,weights_only=False):
        if (not index):
            i = 0
            for mod in self.models:
                if (not weights_only):
                    mod.save(os.path.join(self.prop_dict['directory'],'saves','net_'+str(i)+'_'))
                else:
                    mod.save_weights(os.path.join(self.prop_dict['directory'],'saves','weights_'+str(i)+'_'))
                i += 1
        else:
            if (not weights_only):
                self.models[index].save(os.path.join(self.prop_dict['directory'],'saves','net_'+str(index)+'_'))
            else : 
                self.models[index].save_weights(os.path.join(self.prop_dict['directory'],'saves','weights_'+str(index)+'_'))

    def compile(self,index=None,opt=keras.optimizers.SGD(),loss=keras.losses.MAE):
        if (not index):
            for mod in self.models:
                mod.compile(optimizer=opt,loss=loss)
        else:
            self.models[index].compile(optimizer=opt,loss=loss)

    #def train(self, dataset, epochs, batch_size, layer):
        

