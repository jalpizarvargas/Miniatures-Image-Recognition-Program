#%%
import fastai
import torch
import pandas as pd
import numpy as np
import pywt
from fastai.vision.all import *
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
import torch.utils as utils

#Read Images and Tags, using supervised learning
imgs = get_image_files('./DataSetImages')
ids = pd.read_excel('./DataSetImagesIds.xlsx')

#Using the approach learned from the lab to label the images
def img_labels(img_name):
    return ids.loc[ids['Image ID'] == int(img_name.name.partition('.')[0])].get('Unit Name').values[0]

#Get labels
labels = []

for i, p in enumerate(imgs):
    label = img_labels(p)
    labels.append(label)

labels = np.array(labels)

#Generate wavelets of the images and assign labels

uniqueLabels = list(np.unique(labels))

def label_func(name):
    return uniqueLabels[name]

imgsWavelets = []
labelsWavelets = []

resize = Resize(256,method='squish')

for i, l in enumerate(labels):
    image = PILImage.create(imgs[i])
    image = resize(image)
    image = ImageOps.grayscale(image)

    c = pywt.dwt2(image,'db3',mode='periodization')
    cA,(cH,cV,cD) = c

    imgsWavelets.append(cA)
    labelsWavelets.append(uniqueLabels.index(l))

    imgsWavelets.append(cH)
    labelsWavelets.append(uniqueLabels.index(l))

    imgsWavelets.append(cV)
    labelsWavelets.append(uniqueLabels.index(l))

    imgsWavelets.append(cD)
    labelsWavelets.append(uniqueLabels.index(l))

imgsWavelets = np.array(imgsWavelets)
labelsWavelets = np.array(labelsWavelets)

#Split testing and training data

imgsWavelets_train, imgsWavelets_test, labelsWavelets_train, labelsWavelets_test = train_test_split(imgsWavelets,labelsWavelets,test_size=0.2,shuffle=True,random_state=0)

#Train set

x_train = Tensor(imgsWavelets_train)
y_train = Tensor(labelsWavelets_train)

dataset_train = utils.data.TensorDataset(x_train,y_train)

trn_data = utils.data.DataLoader(dataset_train,batch_size=300,shuffle=True)

#Test set

x_test = Tensor(imgsWavelets_test)
y_test = Tensor(labelsWavelets_test)

dataset_test = utils.data.TensorDataset(x_test,y_test)

#Cnn

tst_data = utils.data.DataLoader(dataset_test,shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cnn_layers = nn.Sequential(
    nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
).to(device)

fc_layers = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features=256*14*14,out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128,out_features=26),
    nn.LogSoftmax()
).to(device)

model = nn.Sequential(*cnn_layers,*fc_layers)

nll_loss = BaseLoss(nn.NLLLoss)

def decode_nllloss(x):
    return x.argmax(axis=1)
nll_loss.decodes = decode_nllloss

for name, layer in model.named_children():
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()
    
learn = Learner(trn_data, model, opt_func=Adam, loss_func=nll_loss, metrics=accuracy)
learn.fit(5, lr=.01)

tst_logprobs, tst_targets = learn.get_preds(dl=tst_data)
tst_loss, tst_acc = learn.validate(dl=tst_data)
tst_preds = tst_logprobs.argmax(axis=1)

print("Test loss: {:.5f} accuracy: {:.5f}".format(tst_loss, tst_acc))
# %%