#%%
import fastai
import torch
import pandas as pd
import numpy as np
import pywt
from fastai.vision.all import *
from PIL import ImageOps
from sklearn.model_selection import train_test_split
import torch.utils as utils
from fastai.data.core import DataLoaders

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

rotateDegrees = [-45,-90,-135,0]

for a in rotateDegrees:
    for i, l in enumerate(labels):
        image = PILImage.create(imgs[i])
        image = resize(image)
        image = ImageOps.grayscale(image)
        image = image.rotate(a)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Split the valid dataset from the training data
imgsWavelets_train, imgsWavelets_valid, labelsWavelets_train, labelsWavelets_valid = train_test_split(imgsWavelets_train,labelsWavelets_train,test_size=0.2,shuffle=True,random_state=0)

x_train = torch.tensor(imgsWavelets_train, dtype=torch.float, device=device)
y_train = torch.tensor(labelsWavelets_train, dtype=torch.long, device=device)

dataset_train = utils.data.TensorDataset(x_train,y_train)

trn_data =utils.data.DataLoader(dataset_train,batch_size=500,shuffle=True)

#Valid set

x_valid = torch.tensor(imgsWavelets_valid, dtype=torch.float, device=device)
y_valid = torch.tensor(labelsWavelets_valid, dtype=torch.long, device=device)

dataset_valid = utils.data.TensorDataset(x_valid,y_valid)

valid_data = utils.data.DataLoader(dataset_valid,shuffle=False)

dls = DataLoaders(trn_data,valid_data)

#CNN

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        ).to(device)

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        ).to(device)

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        ).to(device)

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        ).to(device)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=256*6*6,out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,out_features=26),
            nn.LogSoftmax()
        ).to(device)

    def forward(self, x):
        x = x.float()
        x = x.unsqueeze(1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(-1,256*6*6)
        x = self.fc(x)
        return x

model = Model()

#Train the model with fastai learner

nll_loss = BaseLoss(nn.NLLLoss)

def decode_nllloss(x):
    return x.argmax(axis=1)

nll_loss.decodes = decode_nllloss

for name, layer in model.named_children():
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()
    
learn = Learner(dls, model, opt_func=Adam, loss_func=nll_loss, metrics=accuracy)
learn.fit(5, lr=.01)

#Test set

x_test = torch.tensor(imgsWavelets_test, dtype=torch.float, device=device)
y_test = torch.tensor(labelsWavelets_test, dtype=torch.long, device=device)

dataset_test = utils.data.TensorDataset(x_test,y_test)

tst_data = utils.data.DataLoader(dataset_test,shuffle=False)

dls_test = DataLoaders(tst_data)

#Test the model

#The train tensor of dls_test is the testing tensor since it only contains the tst_data dataloader
tst_loss, tst_acc = learn.validate(dl=dls_test.train)

print("Test loss: {:.5f} accuracy: {:.5f}".format(tst_loss, tst_acc))
# %%