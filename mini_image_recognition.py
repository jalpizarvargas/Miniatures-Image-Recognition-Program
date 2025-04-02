import fastai
from fastai.vision.all import *
import pandas as pd
import numpy as np

#Read Images and Tags, using supervised learning
imgs = get_image_files('./DataSetImages')
ids = pd.read_excel('./DataSetImagesIds.xlsx')

#Using the approach learned from the lab to label the images
def img_labels(img_name):
    return ids.loc[ids['Image ID'] == int(img_name.name.partition('.')[0])].get('Unit Name').values[0]

print(imgs[1].name.partition('.')[0])

pl1 = PILImage.create(imgs[1])

print(pl1.shape)

r = Resize(256,method='squish')

print(r(pl1).shape)

print(img_labels(imgs[1]))

labels = []

for i, p in enumerate(imgs):
    label = img_labels(p)
    labels.append(label)

labels = np.array(labels)

print(labels)

#Ensure all images are at the same resolution

#Run the images through wavelet transform to disect the image for specific characteristics

#Run this images through a classification algorithm 

#Determine performance metrics and analyze results