import numpy as np
from PIL import Image
from os.path import join, basename, isfile, normpath
from os import listdir, walk
import random

import helpers.labels

#======================================================================================================
    #Preprocess functions that can be used on images before feeding them into the net.
    #Please refer to the description of every function
#======================================================================================================


def produce_training_dir(imdir, labeldir, outdir, training_set_size, imW=640, imH=360, crop=True, alllabels = True):
    """
        Creates a folder containing cropped images.
        @ args :
            - imdir : directory of the training images
            - labeldir : directory of the label images
            - outdir : path to the folder where we want to write the new cropped images
            - training_set_size : the number of images to write
            - imW, imH : width and height of the cropping to perform
            - crop : whether to crop or not the images
        @ returns :
            - nothing, simply writes images
    """
    filelist = []
    imdir = normpath(imdir)
    labeldir = normpath(labeldir)
    for path, subdirs, files in walk(imdir):
        for name in files:
            splt_name = str(basename(name)).split(sep="_")
            img_name = join(path, name)
            city = splt_name[0]
            label_name = join(normpath(labeldir), city,
                              city + '_' + splt_name[1] + '_' + splt_name[2] + '_gtFine_labelIds.png')
            if (isfile(label_name)):
                filelist.append([img_name, label_name])
    out = []
    random_indices = np.random.randint(low=0, high=len(filelist), size=training_set_size)
    step = 0
    for i in random_indices:
        if (alllabels):
            Im = Image.open(filelist[i][0])
            Label = Image.open(filelist[i][1])
            if (crop):
                x = np.random.randint(2048 - imW)
                y = np.random.randint(1024 - imH)
                Im = Im.crop((x, y, x + imW, y + imH))
                Label = Label.crop((x, y, x + imW, y + imH))
            else:
                Im.thumbnail((imW, imH))
                Label.thumbnail((imW, imW))
            Im.save(join(outdir, '_' + str(step) + '_im_.png'))
            Label.save(join(outdir, '_' + str(step) + '_lab_.png'))
        else:
            Im = Image.open(filelist[i][0])
            Label = Image.open(filelist[i][1])
            if (crop):
                x = np.random.randint(2048 - imW)
                y = np.random.randint(1024 - imH)
                Im = Im.crop((x, y, x + imW, y + imH))
                Label = Label.crop((x, y, x + imW, y + imH))
            else:
                Im.thumbnail((imW, imH))
                Label.thumbnail((imW, imW))
            Label = Image.eval(Label,labels.convert2trainId)
            Im.save(join(outdir, '_' + str(step) + '_im_.png'))
            Label.save(join(outdir, '_' + str(step) + '_lab_.png'))
        print(step)
        step += 1
    return

def produce_training_set(traindir, trainsize,numlabs=35):
    """
        Produces a list of training images and labels.
        @ args :
            - traindir : path to the directory containing training images.
            - trainsize : an integer, the size of the training set we want to use. Must be lower than the number of images in the folde
        @ returns :
            - out : a list of pairs [im,lab], with 
                im : a 3D numpy array of the image
                lab : a 2D numpy array of the dense labels
    """

    num_labels = numlabs
    indices = list(range(trainsize))
    random.shuffle(indices)
    ins = []
    labs = []
    hist = np.zeros((num_labels))
    for i in indices:
        Im = Image.open(normpath(join(traindir, '_' + str(i) + '_im_.png')))
        im = np.asarray(Im, dtype=np.float32)
        Label = Image.open(join(traindir, '_' + str(i) + '_lab_.png'))
        lab = np.asarray(Label.convert(mode="L"), dtype=np.int)
        maxlabs = num_labels * np.ones_like(lab)
        lab = np.minimum(lab,maxlabs)
        lab = np.eye(num_labels+1)[lab]
        new_hist, _ = np.histogram(lab, bins=num_labels)
        hist += new_hist
        ins.append(im)
        labs.append(lab)
    return ins,labs, hist

def produce_testing_set(testdir, testsize = 100, imH = 128, imW = 256):
    out = []
    for i in range(testsize):
        Im = Image.open(normpath(join(testdir, '_' + str(i) + '_im_.png')))
        Im.thumbnail([imW,imH])
        im = np.asarray(Im, dtype=np.float32)
        Label = Image.open(join(testdir, '_' + str(i) + '_lab_.png'))
        Label.thumbnail([imW,imH])
        lab = np.asarray(Label.convert(mode="L"), dtype=np.float32)
        out.append([im,lab])
    return out

def one_hot(a,num_classes):
    b = np.zeros(shape=(a.shape).append(num_classes))

 