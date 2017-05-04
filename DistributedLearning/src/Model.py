from __future__ import print_function, absolute_import, division
import json
import os, sys

class Model:
    """
        A class describing models, with different properties stored in a dictionnary object.
        Properties :
            - name : a string, name of the model.
            - 
        """
#==============================================================================
    def __init__(self,dir = './default_model'):
        """
            Initialisation method, creates an object reading its properties stored in the 'properties.json' file in the directory.
            If no directory exists, a new one is created.
        """

        self.prop_dict = {}

        if ( not os.path.isdir(dir)):
            print("Model directory did not exist, creating from scratch")
            os.mkdir(dir)
            os.mkdir(os.path.join(dir,"saves"))
            os.mkdir(os.path.join(dir,"logs"))
            self.define_name()
            self.define_network()
        else:
            prop_dict = json.load(open(os.path.join(dir,properties)))
            print("Model loaded from .json file")
#==============================================================================
    def define_network(self,building_function=None):
        """
            Define the neural network building function from the string passed as argument.
            The string must designate a valid network builder function, see 'Network.py' for more info
        """
        if (not is_valid(building_function)):
            print("Please specify a valid building function")
            self.prop_dict['net_builder'] = None
        else:
            print("Defining building function")
            self.prop_dict['net_builder'] = building_function

    def define_name(self, name = 'default'):
        """
            Changes the name of the model as defined
        """
        self.prop_dict['name'] = name

