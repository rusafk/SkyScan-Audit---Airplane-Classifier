# SkyScan Audit - Airplane Classifier
Detect passenger jets from SkyScan images

import torch
import numpy as np
import pandas as pd
import sklearn
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import os
import torchvision
from torchvision.io import read_image
import torchvision.transforms as T

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from mlxtend.plotting import heatmap

## coefficient of determination (R**2)
from sklearn.metrics import r2_score

from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report


#######################################################

from fastai.vision.all import *


#######################################################

N_EPOCHS = 10000          ## 4000
batch_size = 10   ## 5    ## 32
learning_rate =  0.1    ## 0.01   ## 1e-5 


#######################################################

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

#######################################################

## Download SkyScanCorpus with a folder inside for each class (e.g. boeing, airbus, etc). 
## Put images in each corresponding folder

path = Path('CorpusSkyScan')
fns = get_image_files(path)
fns

## This is a data loader

## parent_label -> simply gets the name of the folder a file is in

planes = DataBlock(
     blocks = (ImageBlock, CategoryBlock),
     get_items = get_image_files,
     splitter = RandomSplitter(valid_pct=0.2, seed=42),
     get_y = parent_label,
     item_tfms = Resize(128)  ## by default it crops
)
dls = planes.dataloaders(path)

## by defaullt it will give the model batches of 64 for training and testing

## to view

dls.valid.show_batch(max_n=4, nrows=1)

## instead of cropping

## we can pad the images

planes = planes.new(item_tfms=Resize(128, ResizeMethod.Pad, pad_mode='zeros'))
dls = planes.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)

## or we can squish them

planes = planes.new(item_tfms=Resize(128, ResizeMethod.Squish))
dls = planes.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)

## random cropping of an image is considered better

planes = planes.new(item_tfms=RandomResizedCrop(128, min_scale=0.3))
dls = planes.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)

planes = planes.new(item_tfms=RandomResizedCrop(128, min_scale=0.5))
dls = planes.dataloaders(path)
dls.train.show_batch(max_n=4, nrows=1, unique=True)

## data augmentation: rotation, flipping, perspective warping, contrast and brightness changes
## via GPU intensive batch_transforms

planes = planes.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))
dls = planes.dataloaders(path)
dls.train.show_batch(max_n=16, nrows=4, unique=True)

## now fine tune cnn_learner with our data

planes = planes.new(
     item_tfms=RandomResizedCrop(224, min_scale=0.5),
     batch_tfms=aug_transforms()
)

dls = planes.dataloaders(  path  )

learn = cnn_learner(dls, resnet18, metrics=error_rate)    ## metrics=batch_accuracy, metrics=accuracy

learn.fine_tune(4)

interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()

# ML Performance Metrics
preds, y, losses = learn.get_preds(with_loss=True)
preds.shape


y_pred = torch.argmax(preds, dim=1)
y_pred.shape
torch.Size([2218])
y.shape
torch.Size([2218])
y_np       = y.numpy()
y_pred_np = y_pred.numpy()

accuracy = accuracy_score(y_np, y_pred_np)

f1 = f1_score(y_np, y_pred_np, average='weighted')

print(accuracy)
print(f1)


learn.dls.vocab

## target_names = ['class 0', 'class 1', 'class 2']
## print(classification_report(y_true, y_pred, target_names=target_names))

print(classification_report(y_np, y_pred_np))

## plot_top_losses shows us the images with the highest loss in our dataset


interp.plot_top_losses(4, nrows=4)


## will save a export file called export.pkl to save the model


learn.export("SkyScanModel16Labels.pkl")

## load_model from file


learn_inf = load_learner('SkyScanModel16Labels.pkl')

img = 'CorpusSkyScan/AIRBUS/AA1219_2021-03-29-22-19-39.jpg'


is_plane, _, probs = learn.predict(img)
print(is_plane)
print(probs)


is_plane, _, probs = learn_inf.predict(img)
print(is_plane)
print(probs)

learn_inf.dls.vocab

im = Image.open('CorpusSkyScan/AIRBUS/AA1219_2021-03-29-22-19-39.jpg')
im.thumbnail((256,256))
im




learn.show_results()


learn.show_results(max_n=10, figsize=(7, 8))