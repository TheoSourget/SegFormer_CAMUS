import matplotlib.pyplot as plt
import torch
import torch.nn as nn
torch.manual_seed(1907)
from CamusEDImageDataset import CamusEDImageDataset
from torchvision.transforms import ToTensor,Compose,Resize,ToPILImage,PILToTensor
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torchmetrics.functional import dice

from transformers import SegformerForSemanticSegmentation
from datasets import load_metric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VISUAL = True
TF = False

id2label = {i: i for i in range(4)}
label2id = {i: i for i in range(4)}
if TF:
    model = SegformerForSemanticSegmentation.from_pretrained("./FTSegFormer.pt",
                                                    num_labels=4, 
                                                    id2label=id2label, 
                                                    label2id=label2id,
    )
else:     
    model = SegformerForSemanticSegmentation.from_pretrained("./SFTSegFormer.pt",
                                                    num_labels=4, 
                                                    id2label=id2label, 
                                                    label2id=label2id,
    )  
model.eval()

test_dataset =CamusEDImageDataset(
    transform=Compose([ToPILImage(),Resize((256,256)),ToTensor()]),
    target_transform=Compose([ToPILImage(),Resize((256,256)),PILToTensor()]),
    test=True
)

metric = load_metric("mean_iou")

test_dataloader = DataLoader(test_dataset, batch_size=1)
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

 
dices = {i:[] for i in range(4)}
hd = {i:[] for i in range(4)}
with torch.no_grad():
    for idx, batch in enumerate(tqdm(test_dataloader)):
        # get the inputs;
        pixel_values = batch[0].to(device)
        labels = batch[1].squeeze(1).type(torch.LongTensor).to(device)

        # zero the parameter gradients
        # forward + backward + optimize
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss, logits = outputs.loss, outputs.logits
        upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        predicted = upsampled_logits.argmax(dim=1)
         
        metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())
        
        dice_metric = dice(predicted.detach().cpu(),labels.detach().cpu(),average = None,num_classes=4,ignore_index=0)
        for i in range(len(dice_metric)):
            dices[i].append(dice_metric[i].item())
            
        if VISUAL:
            if idx == 0:
                baseImage = pixel_values[0].permute(1,2,0)
                baseImage = baseImage.detach().cpu().numpy()
                gt = labels[0]
                gt = gt.detach().cpu().numpy()
                upsampled_logits = nn.functional.interpolate(logits,
                    size=(256,256), # (height, width)
                    mode='bilinear',
                    align_corners=False)
                seg = upsampled_logits.argmax(dim=1)[0].detach().cpu().numpy()
                

                
    metrics = metric.compute(
        num_labels=len(id2label), 
        ignore_index=0,
        reduce_labels=False,
    )
    
    print("TEST METRIC")
    print("Mean_iou:", np.mean(metrics["per_category_iou"][1:])) #Remove first value because it's 0 and not nan 
    print("Mean accuracy:", metrics["mean_accuracy"])
    print("iou per category:", metrics["per_category_iou"])
    print("Accuracy per category:", metrics["per_category_accuracy"])
    
    print('DICES:')
    for i in range(1,4):
        print("Class",i)
        print(f"Mean: {np.mean(dices[i])}\t+/-{np.std(dices[i])}")
    
    if VISUAL:
        plt.figure(figsize=(20,20))
        plt.subplot(131)
        plt.title("Echography image")
        plt.axis("off")
        plt.imshow(baseImage,cmap="gray")
        
        plt.subplot(132)
        plt.title("Ground Truth")
        plt.imshow(gt)
        plt.axis("off")
        
        plt.subplot(133)
        plt.title("Model Segmentation")
        plt.imshow(seg,vmax=3,vmin=0)
        plt.axis("off")
        
        plt.show()