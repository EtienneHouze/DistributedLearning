import numpy as np


class BatchGenerator:
    def __init__(self, imset, labset, city_model, batchsize=5):
        self.i = 0
        self.imset = imset
        self.labset = labset
        self.batchsize = batchsize
        self.city_model = city_model
        self.epoch_size = self.imset[:, 0, 0, 0].size // self.batchsize

    def generate_batch(self, layer_index=0):
        while self.i > -1:
            i = self.i % self.epoch_size
            if (i == 0):
                np.random.shuffle(self.imset)
                np.random.shuffle(self.labset)
            ins_list = self.imset[self.batchsize * i:(i * self.batchsize) + self.batchsize, :, :, :]
            labs_list = self.labset[self.batchsize * i:(i * self.batchsize) + self.batchsize, :, :, :]
            for index in range(layer_index):
                ins_list = self.city_model.models[index].predict_on_batch(ins_list)
            self.i += 1
            yield (ins_list, labs_list)
