import torch
torch.manual_seed(1907)
import torch.nn as nn

from torchvision.transforms import ToTensor,Compose,Resize,ToPILImage,PILToTensor,RandomRotation
from torch.utils.data import DataLoader
import torch.optim as optim

from sklearn.model_selection import train_test_split


import numpy as np
from CamusEDImageDataset import CamusEDImageDataset

from transformers import SegformerForSemanticSegmentation, SegformerConfig
from datasets import load_metric

from sklearn.metrics import accuracy_score
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import glob

#Load dataset
NB_EPOCHS = 50
NBSAMPLES = len(glob.glob("./data/training/**/*_ED.mhd"))
VALID_SIZE = 2
TF = False


train_data =CamusEDImageDataset(
    transform=Compose([ToPILImage(),Resize((256,256)),RandomRotation(10),ToTensor()]),
    target_transform=Compose([ToPILImage(),Resize((256,256)),RandomRotation(10),PILToTensor()]),
)

valid_data =CamusEDImageDataset(
    transform=Compose([ToPILImage(),Resize((256,256)),ToTensor()]),
    target_transform=Compose([ToPILImage(),Resize((256,256)),PILToTensor()]),
)

train_indices, val_indices = train_test_split(np.arange(0,NBSAMPLES,1),test_size=VALID_SIZE,random_state=1907)

train_data = torch.utils.data.Subset(train_data,train_indices)
valid_data =torch.utils.data.Subset(valid_data,val_indices)

train_dataloader = DataLoader(train_data, batch_size=5)
valid_dataloader = DataLoader(valid_data, batch_size=5)
id2label = {i: i for i in range(4)}
label2id = {i: i for i in range(4)}

if TF:
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",
                                                             num_labels=4, 
                                                             id2label=id2label, 
                                                             label2id=label2id,
    )
else:
    configSegformerB0 = SegformerConfig(input_channel=1,depths=[2,2,2,2],hidden_sizes=[32,64,160,256],decoder_hidden_size=256,num_labels=4,id2label=id2label,       label2id=label2id,)
    model = SegformerForSemanticSegmentation(configSegformerB0)


total_params = sum(
	param.numel() for param in model.parameters()
)

print(total_params)

metric = load_metric("mean_iou")
optimizer = optim.Adam(model.parameters(),lr=1e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.train()
imgs = []

lossEvolve = []
valEvolve = []
valIOUEvolve = []
valAccEvolve = []
model = model.to(device)
for epoch in tqdm(range(50)):  # loop over the dataset multiple times
    print("Epoch:", epoch+1)
    for idx, batch in enumerate(train_dataloader):
        # get the inputs;
        pixel_values = batch[0].to(device)
        labels = batch[1].squeeze(1).type(torch.LongTensor).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss, logits = outputs.loss, outputs.logits
        loss.backward()
        optimizer.step()

        # evaluate
        with torch.no_grad():
          upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
          predicted = upsampled_logits.argmax(dim=1)
          metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

    metrics = metric.compute(
        num_labels=len(id2label), 
        ignore_index=0,
        reduce_labels=False,
    )
    print("TRAIN METRIC")
    print("Loss:", loss.item())
    print("Mean_iou:", metrics["mean_iou"])
    print("Mean accuracy:", metrics["mean_accuracy"])
    
    lossEvolve.append(loss.item())
    
    #VALIDATION
    for idx, batch in enumerate(valid_dataloader):
        
        # get the inputs;
        pixel_values = batch[0].to(device)
        labels = batch[1].squeeze(1).type(torch.LongTensor).to(device)


        outputs = model(pixel_values=pixel_values, labels=labels)
        loss, logits = outputs.loss, outputs.logits
        
        #For Animation:
        if idx == 0:
            if epoch == 0:
                baseImage = pixel_values[0].permute(1,2,0)
                baseImage = baseImage.detach().cpu().numpy()

            upsampled_logits = nn.functional.interpolate(logits,
                size=(256,256), # (height, width)
                mode='bilinear',
                align_corners=False)
            seg = upsampled_logits.argmax(dim=1)[0]
            imgs.append(seg)

        # evaluate
        with torch.no_grad():
          upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
          predicted = upsampled_logits.argmax(dim=1)
          
          metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

    metrics = metric.compute(
        num_labels=len(id2label), 
        ignore_index=0,
        reduce_labels=False,
    )
    print("VALID METRIC")
    print("Loss:", loss.item())
    print("Mean_iou:", metrics["mean_iou"])
    print("Mean accuracy:", metrics["mean_accuracy"])

    valEvolve.append(loss.item())
    valIOUEvolve.append(metrics["mean_iou"])
    valAccEvolve.append(metrics["mean_accuracy"])
    
    if loss.item() == min(valEvolve):
        if TF:
            model.save_pretrained("./FTSegFormer.pt")
        else:
            model.save_pretrained("./SFTSegFormer.pt")


plt.figure(figsize=(5,5))
plt.plot(lossEvolve,label="Train set loss")
plt.plot(valEvolve,label="Validation set loss")
plt.title("Evolution of loss for validation and train dataset")
plt.legend()
plt.show()

plt.figure(figsize=(5,5))
plt.plot(valIOUEvolve)
plt.title("Evolution of mean IOU metric on validation set")
plt.show()

plt.figure(figsize=(5,5))
plt.plot(valAccEvolve)
plt.title("Evolution of mean accuracy on validation set")
plt.show()


palette = np.array([[0,0,0],[255,0,0],[0,255,0],[0,0,255]])

def updateSeg(i,imgBase,segEvolve):
    plt.clf()
    seg = segEvolve[i].detach().cpu().numpy()
    plt.title(f"Evolution of segmentation with SegFormerB0: Epochs {i+1}")
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3

    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    color_seg = color_seg[..., ::-1]

    plt.imshow(imgBase)
    plt.imshow(color_seg,alpha=0.3)

#Animation
frames = [] # for storing the generated images
fig = plt.figure()
plt.title("Evolution of segmentation with SegFormer during 50 Epochs")
ani = animation.FuncAnimation(fig, updateSeg,frames=len(imgs), interval=1000,repeat_delay=300,fargs=(baseImage,imgs))

writergif = animation.PillowWriter(fps=10) 
ani.save('Segformer.gif', writer=writergif)
plt.show()
