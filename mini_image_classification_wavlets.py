#%%
import fastai
import torch
import pandas as pd
import numpy as np
import pywt
from fastai.vision.all import *
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

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

image = PILImage.create(imgs[0])
image = image.resize((256,256))
image = ImageOps.grayscale(image)

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
    labelsWavelets.append(l)

    imgsWavelets.append(cH)
    labelsWavelets.append(l)

    imgsWavelets.append(cV)
    labelsWavelets.append(l)

    imgsWavelets.append(cD)
    labelsWavelets.append(l)

imgsWavelets = np.array(imgsWavelets)
labelsWavelets = np.array(labelsWavelets)

print('Total')
print(len(imgsWavelets))
print(len(labelsWavelets))

imgsWavelets_train, imgsWavelets_test, labelsWavelets_train, labelsWavelets_test = train_test_split(imgsWavelets,labelsWavelets,test_size=0.2,shuffle=True,random_state=0)

print('Test Set')
print(np.unique(labelsWavelets_test).size)
print(len(imgsWavelets_test))
print(len(labelsWavelets_test))

print('Train Set')
print(len(imgsWavelets_train))
print(len(labelsWavelets_train))

valid_split = .2
seed = 0
data_block = ImageBlock(cls=PILImageBW)
target_block = CategoryBlock()

img_db = DataBlock(blocks=(data_block,target_block),splitter=RandomSplitter(valid_pct=valid_split,seed=seed))
# %%