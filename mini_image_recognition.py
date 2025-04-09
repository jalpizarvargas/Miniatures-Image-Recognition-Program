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

#Get labels
labels = []

for i, p in enumerate(imgs):
    label = img_labels(p)
    labels.append(label)

labels = np.array(labels)

#Split the images into testing and training
tts = TrainTestSplitter(test_size=.20, shuffle=True, stratify=labels, random_state=0)

trn_set, tst_set = tts(imgs)

trn_paths = imgs[trn_set]
tst_paths  = imgs[tst_set]

#Ensure all images are at the same resolution and initiate DataBlock class
valid_split = .2
seed = 0
data_block = ImageBlock(cls=PILImageBW)
target_block = CategoryBlock()

img_db = DataBlock(blocks=(data_block,target_block),splitter=RandomSplitter(valid_pct=valid_split,seed=seed),
                   get_y=img_labels,item_tfms=Resize(256,method='squish'))

#Train dataset
trn_vld_batch = img_db.dataloaders(trn_paths,batch_size=256)

#Test dataset
testing_set = trn_vld_batch.test_dl(tst_paths, with_labels=True)

print(testing_set.dataset)

#Run the images through wavelet transform to disect the image for specific characteristics

#Run this images through a classification algorithm 

#Determine performance metrics and analyze results