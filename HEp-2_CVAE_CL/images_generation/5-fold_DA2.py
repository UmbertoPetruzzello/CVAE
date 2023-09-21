from __future__ import print_function, division
import os
import cv2
import torch
from torchvision import transforms
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import imutils
from numpy import newaxis
import numpy as np
import csv
from tqdm import tqdm
from PIL import Image
from torchvision.utils import save_image
from utils import save

generated_dir = 'generated_augmented_images/5-fold/'
root_dir= 'generated_images/5-fold/'
frame = pd.read_csv('k-fold_csv/gen_5-fold.csv', names=["Image", "Mask", "Label", "Intensity"])
total_frame = len(frame)
generated_csv = 'k-fold_csv/gen_aug_5-fold.csv'

transf1 = transforms.Compose([transforms.RandomRotation((+180,+180)), transforms.ToTensor()])
transf2 = transforms.Compose([transforms.RandomRotation((+90,+90), expand = True), transforms.ToTensor()])
transf3 = transforms.Compose([transforms.RandomRotation((+270,+270), expand = True), transforms.ToTensor()])
transf4 = transforms.Compose([transforms.RandomHorizontalFlip(p=1), transforms.ToTensor()])
transf5 = transforms.Compose([transforms.RandomVerticalFlip(p=1), transforms.ToTensor()])
transf6 = transforms.Compose([transforms.RandomRotation((+180,+180)), transforms.RandomHorizontalFlip(p=1), transforms.ToTensor()])
transf7 = transforms.Compose([transforms.RandomRotation((+180,+180)), transforms.RandomVerticalFlip(p=1), transforms.ToTensor()])
transf8 = transforms.Compose([transforms.RandomRotation((+90,+90), expand = True), transforms.RandomHorizontalFlip(p=1), transforms.ToTensor()])
transf9 = transforms.Compose([transforms.RandomRotation((+90,+90), expand = True), transforms.RandomVerticalFlip(p=1), transforms.ToTensor()])
transf10 = transforms.Compose([transforms.RandomRotation((+270,+270), expand = True), transforms.RandomHorizontalFlip(p=1), transforms.ToTensor()])

#precisare il numero di campioni di partenza per ogni classe 
n_Hp = 1115
n_Hi = 1050
n_Sp = 1041
n_Si = 1057
n_Np = 1145
n_Ni = 999  
n_Cp = 1056
n_Ci = 1059
n_Gi = 1257
n_Gp = 1262
n_NMp = 1143
n_NMi = 1079

total_generated_img = 0
num_to_reach = 1500 #precisare il numero di campioni da voler raggiungere per ogni classe 

for i in tqdm(range(total_frame)):
    img_name = os.path.join(root_dir,frame.loc[i]["Image"])
    label = frame.loc[i]["Label"]
    intensity = frame.loc[i]["Intensity"]
    mask_name = os.path.join(root_dir,frame.loc[i]["Mask"])
    mask = Image.open(mask_name)
    image = Image.open(img_name)

    if label == 'Nucleolar' and intensity == 'intermediate' and n_Ni < num_to_reach: #majority class generate only one sample
        if n_Ni < num_to_reach: 
            new_img = transf1(image)
            new_mask = transf1(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv, total_generated_img,i)
            n_Ni += 1
            total_generated_img +=1


        if n_Ni < num_to_reach: 
            new_img = transf2(image)
            new_mask = transf2(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv, total_generated_img, i)
            n_Ni += 1
            total_generated_img +=1
    
    if label == 'Homogeneous' and intensity == 'positive' and n_Hp < num_to_reach:
        if n_Hp < num_to_reach: 
            new_img = transf1(image)
            new_mask = transf1(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv, total_generated_img,i)
            n_Hp += 1
            total_generated_img +=1


        if n_Hp < num_to_reach: 
            new_img = transf2(image)
            new_mask = transf2(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv, total_generated_img, i)
            n_Hp += 1
            total_generated_img +=1

        if n_Hp < num_to_reach:
            new_img = transf3(image)
            new_mask = transf3(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img, i)
            n_Hp += 1
            total_generated_img +=1


    if label == 'Homogeneous' and intensity == 'intermediate' and n_Hi < num_to_reach:
        if n_Hi < num_to_reach: 
            new_img = transf1(image)
            new_mask = transf1(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img,i)
            n_Hi += 1
            total_generated_img +=1

        if n_Hi < num_to_reach: 
            new_img = transf2(image)
            new_mask = transf2(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img,i)
            n_Hi += 1
            total_generated_img +=1
    
    if label == 'Speckled' and intensity == 'positive' and n_Sp < num_to_reach:
        if n_Sp < num_to_reach: 
            new_img = transf1(image)
            new_mask = transf1(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img,i)
            n_Sp += 1
            total_generated_img +=1

        if n_Sp < num_to_reach: 
            new_img = transf2(image)
            new_mask = transf2(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img,i)
            n_Sp += 1
            total_generated_img +=1
    
    if label == 'Speckled' and intensity == 'intermediate' and n_Si < num_to_reach:
        if n_Si < num_to_reach: 
            new_img = transf1(image)
            new_mask = transf1(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img,i)
            n_Si += 1
            total_generated_img +=1

        if n_Si < num_to_reach: 
            new_img = transf2(image)
            new_mask = transf2(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img,i)
            n_Si += 1
            total_generated_img +=1
    
    if label == 'Centromere' and intensity == 'positive' and n_Cp < num_to_reach:
        if n_Cp < num_to_reach: 
            new_img = transf1(image)
            new_mask = transf1(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img,i)
            n_Cp += 1
            total_generated_img +=1

        if n_Cp < num_to_reach: 
            new_img = transf2(image)
            new_mask = transf2(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img,i)
            n_Cp += 1
            total_generated_img +=1

    
    if label == 'Centromere' and intensity == 'intermediate' and n_Ci < num_to_reach:
        if n_Ci < num_to_reach: 
            new_img = transf1(image)
            new_mask = transf1(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img,i)
            n_Ci += 1
            total_generated_img +=1

        if n_Ci < num_to_reach: 
            new_img = transf2(image)
            new_mask = transf2(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img,i)
            n_Ci += 1
            total_generated_img +=1

    
    if label == 'Nucleolar' and intensity == 'positive' and n_Np < num_to_reach:
        if n_Np < num_to_reach: 
            new_img = transf1(image)
            new_mask = transf1(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img,i)
            n_Np += 1
            total_generated_img +=1

        if n_Np < num_to_reach: 
            new_img = transf2(image)
            new_mask = transf2(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img,i)
            n_Np += 1
            total_generated_img +=1

        if n_Np < num_to_reach: 
            new_img = transf2(image)
            new_mask = transf2(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img,i)
            n_Np += 1
            total_generated_img +=1

    
    if label == 'NuMem' and intensity == 'positive' and n_NMp < num_to_reach:
        if n_NMp < num_to_reach: 
            new_img = transf1(image)
            new_mask = transf1(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img,i)
            n_NMp += 1
            total_generated_img +=1

        if n_NMp < num_to_reach: 
            new_img = transf2(image)
            new_mask = transf2(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img,i)
            n_NMp += 1
            total_generated_img +=1

        if n_NMp < num_to_reach: 
            new_img = transf2(image)
            new_mask = transf2(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img,i)
            n_NMp += 1
            total_generated_img +=1

    
    if label == 'NuMem' and intensity == 'intermediate' and n_NMi < num_to_reach:
        if n_NMi < num_to_reach: 
            new_img = transf1(image)
            new_mask = transf1(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img,i)
            n_NMi += 1
            total_generated_img +=1

        if n_NMi < num_to_reach: 
            new_img = transf2(image)
            new_mask = transf2(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img,i)
            n_NMi += 1
            total_generated_img +=1

    
    if label == 'Golgi' and intensity == 'positive' and n_Gp < num_to_reach:

        if n_Gp < num_to_reach:
            new_img = transf1(image)
            new_mask = transf1(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img,i)
            n_Gp += 1
            total_generated_img +=1

        if n_Gp < num_to_reach:
            new_img = transf2(image)
            new_mask = transf2(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img,i)
            n_Gp += 1
            total_generated_img +=1

        if n_Gp < num_to_reach:
            new_img = transf3(image)
            new_mask = transf3(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img,i)
            n_Gp += 1
            total_generated_img +=1

        if n_Gp < num_to_reach:
            new_img = transf4(image)
            new_mask = transf4(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img,i)
            n_Gp += 1
            total_generated_img +=1

        if n_Gp < num_to_reach:
            new_img = transf5(image)
            new_mask = transf5(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img,i)
            n_Gp += 1
            total_generated_img +=1

        if n_Gp < num_to_reach:
            new_img = transf6(image)
            new_mask = transf6(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img,i)
            n_Gp += 1
            total_generated_img +=1

        if n_Gp < num_to_reach:
            new_img = transf7(image)
            new_mask = transf7(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img,i)
            n_Gp += 1
            total_generated_img +=1

        if n_Gp < num_to_reach:
            new_img = transf8(image)
            new_mask = transf8(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img,i)
            n_Gp += 1
            total_generated_img +=1

        if n_Gp < num_to_reach:
            new_img = transf9(image)
            new_mask = transf9(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img,i)
            n_Gp += 1
            total_generated_img +=1

        
    
    if label == 'Golgi' and intensity == 'intermediate' and n_Gi < num_to_reach:
        if n_Gi < num_to_reach:
            new_img = transf1(image)
            new_mask = transf1(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img,i)
            n_Gi += 1
            total_generated_img +=1

        if n_Gi < num_to_reach:
            new_img = transf2(image)
            new_mask = transf2(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img,i)
            n_Gi += 1
            total_generated_img +=1

        if n_Gi < num_to_reach:
            new_img = transf3(image)
            new_mask = transf3(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img,i)
            n_Gi += 1
            total_generated_img +=1

        if n_Gi < num_to_reach:
            new_img = transf4(image)
            new_mask = transf4(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img,i)
            n_Gi += 1
            total_generated_img +=1

        if n_Gi < num_to_reach:
            new_img = transf5(image)
            new_mask = transf5(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img,i)
            n_Gi += 1
            total_generated_img +=1

        if n_Gi < num_to_reach:
            new_img = transf6(image)
            new_mask = transf6(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img,i)
            n_Gi += 1
            total_generated_img +=1

        if n_Gi < num_to_reach:
            new_img = transf7(image)
            new_mask = transf7(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img,i)
            n_Gi += 1
            total_generated_img +=1

        if n_Gi < num_to_reach:
            new_img = transf8(image)
            new_mask = transf8(mask)
            save(new_img,new_mask,label,intensity,generated_dir,generated_csv,total_generated_img,i)
            n_Gi += 1
            total_generated_img +=1


print(f'Total Homogeneous positive:{n_Hp}') 
print(f'Total Homogeneous intermediate:{n_Hi}') 
print(f'Total Speckled  positive:{n_Sp}') 
print(f'Total Speckled internediate:{n_Si}') 
print(f'Total Nucleolar positive:{n_Np}') 
print(f'Total Nucleolar intermediate:{n_Ni}') 
print(f'Total Centromere positive:{n_Cp}') 
print(f'Total Centromere intermediate:{n_Ci}') 
print(f'Total Golgi positive:{n_Gp}') 
print(f'Total Golgi intermediate:{n_Gi}') 
print(f'Total NuMem positive:{n_NMp}') 
print(f'Total NuMem intermediate:{n_NMi}') 
