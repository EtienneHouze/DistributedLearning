# Learning Script

from __future__ import absolute_import, division, print_function

from os.path import join
from src.CityScapeModel import CityScapeModel

DATAFOLDER = 'C:/Users/Etienne.Houze/Documents/EtienneData'
model_name = 'test_newscript'

model = CityScapeModel(join(DATAFOLDER,model_name))


# Callbacks ==================================================
model.add_callback('view_output',
                  batch_interval=10,
                  on_epoch=False,
                  num_ins=5)

model.add_callback(
        'history_loss',
        write_on_epoch=True
        )
model.add_callback('console_display')
# =======================================================
model.define_input((256, 512, 4))
model.define_numlabs(18)
model.define_network('inception_pooling')
model.define_training_set('D:/EtienneData/train_with_disp','with_disp',300)
model.define_validation_set('D:/EtienneData/train_with_disp','with_disp',50)
model.define_loss('categorical_crossentropy')
model.define_learning_rate(0.001)
model.build_net()
model.print_net()
model.define_metrics('iou','acc')
# # test.print_model()
# test.load_weights()
model.save_tojson()
model.print_png()
model.train(epochs=30,
           batch_size=15,
           save=True
           )
# test.compute_on_dir('D:/EtienneData/train_with_disp', outdir=join(DATAFOLDER,'output'))
# # test.evaluate(outputfile='./results_val.csv')