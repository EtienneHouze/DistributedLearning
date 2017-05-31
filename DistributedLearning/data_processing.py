# Data processing script

from __future__ import absolute_import, division, print_function

from helpers import visualization
from os.path import join


DATAFOLDER = 'C:/Users/Etienne.Houze/Documents/EtienneData'
model_name = 'test'

visualization.visualize_csv(join(DATAFOLDER,model_name))
visualization.convert_labelled_output(join(DATAFOLDER,model_name),num_labs=18)
