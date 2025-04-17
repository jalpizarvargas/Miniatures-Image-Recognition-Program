import fastai
import torch
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
trn_vld_dls = img_db.dataloaders(trn_paths,batch_size=100,num_workers=0,shuffle=True)

#Test dataset
testing_set = trn_vld_dls.test_dl(tst_paths, with_labels=True)

#Run dataset through a convolutional neural net to compare with a wavelet scattering network and wavelet preprocessing

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
    
learn = Learner(trn_vld_dls, model, opt_func=Adam, loss_func=nll_loss, metrics=accuracy)
learn.fit(5, lr=.01)

tst_logprobs, tst_targets = learn.get_preds(dl=testing_set)
tst_loss, tst_acc = learn.validate(dl=testing_set)
tst_preds = tst_logprobs.argmax(axis=1)

print("Test loss: {:.5f} accuracy: {:.5f}".format(tst_loss, tst_acc))