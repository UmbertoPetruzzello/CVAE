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
from PIL import Image
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import matplotlib.pyplot as plt
from utils import save, find_label
from CVAE import CVAE, CVAEMask


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using %s for computation" % device)

model_dir = './models_CVAE/'


#precisare il numero di campioni per ogni classe 
n_Hp = 217
n_Hi = 281
n_Sp = 291
n_Si = 275
n_Np = 187
n_Ni = 333  #majority class 
n_Cp = 276
n_Ci = 273
n_Gi = 75
n_Gp = 70
n_NMp = 188
n_NMi = 253

#precisare il numero di campioni da voler raggiungere
num_to_reach = n_Ni*2

total_generated_img = 0


transf = transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor()])


def generate(model, total_generated_img, num_model,n_Hp, n_Hi, n_Sp, n_Si, n_Np, n_Ni, n_Cp, n_Ci, n_Gp, n_Gi, n_NMp, n_NMi):
    global num_to_reach

    root_dir= '../train/'
    #per generare da fold diversi cambiare le directory
    generated_dir = 'generated_images/5-fold/'
    generated_csv = 'k-fold_csv/gen_5-fold.csv'
    frame = pd.read_csv('k-fold_csv/5-fold.csv', names=["Image", "Mask", "Label", "Intensity"])
    total_frame = len(frame)
    
    for i in tqdm(range(total_frame)):
        img_name = os.path.join(root_dir,frame.loc[i]["Image"])
        label = frame.loc[i]["Label"]
        intensity = frame.loc[i]["Intensity"]
        mask_name = os.path.join(root_dir,frame.loc[i]["Mask"])
        mask = Image.open(mask_name)
        image = Image.open(img_name)

        img = transf(image)
        mask = transf(mask)

        lab, inten = find_label(label,intensity)

        lab = torch.as_tensor(int(lab), dtype=torch.float32)
        inten = torch.as_tensor(int(inten), dtype=torch.int16)

        lab = [lab,inten]
        lab = torch.as_tensor(lab)

        img = img.view(1,1,128,128)
        mask_emb = mask.view(1,-1)
        img = img.to(device)
        lab = lab.to(device)
        lab = lab.view(1,-1)
        mask_emb = mask_emb.to(device)
        if label == 'Homogeneous' and intensity == 'positive' and n_Hp<num_to_reach:
            generated, _, _ = model(img, lab, mask_emb)
            save(generated,mask,label,intensity,generated_dir,generated_csv, total_generated_img,i)
            total_generated_img += 1
            n_Hp += 1
        if label == 'Homogeneous' and intensity == 'intermediate' and n_Hi<num_to_reach:
            generated, _, _ = model(img, lab, mask_emb)
            save(generated,mask,label,intensity,generated_dir,generated_csv, total_generated_img,i)
            total_generated_img += 1
            n_Hi += 1  
        if label == 'Speckled' and intensity == 'positive' and n_Sp<num_to_reach:
            generated, _, _ = model(img, lab, mask_emb)
            save(generated,mask,label,intensity,generated_dir,generated_csv, total_generated_img,i)
            total_generated_img += 1
            n_Sp += 1 
        if label == 'Speckled' and intensity == 'intermediate' and n_Si<num_to_reach:
            generated, _, _ = model(img, lab, mask_emb)
            save(generated,mask,label,intensity,generated_dir,generated_csv, total_generated_img,i)
            total_generated_img += 1
            n_Si += 1 
        if label == 'Nucleolar' and intensity == 'positive' and n_Np<num_to_reach:
            generated, _, _ = model(img, lab, mask_emb)
            save(generated,mask,label,intensity,generated_dir,generated_csv, total_generated_img,i)
            total_generated_img += 1
            n_Np += 1 
        if label == 'Nucleolar' and intensity == 'intermediate' and n_Ni<num_to_reach:
            generated, _, _ = model(img, lab, mask_emb)
            save(generated,mask,label,intensity,generated_dir,generated_csv, total_generated_img,i)
            total_generated_img += 1
            n_Ni += 1
        if label == 'Centromere' and intensity == 'positive' and n_Cp<num_to_reach:
            generated, _, _ = model(img, lab, mask_emb)
            save(generated,mask,label,intensity,generated_dir,generated_csv, total_generated_img,i)
            total_generated_img += 1
            n_Cp += 1
        if label == 'Centromere' and intensity == 'intermediate' and n_Ci<num_to_reach:
            generated, _, _ = model(img, lab, mask_emb)
            save(generated,mask,label,intensity,generated_dir,generated_csv, total_generated_img,i)
            total_generated_img += 1
            n_Ci += 1
        if label == 'Golgi' and intensity == 'positive' and n_Gp<num_to_reach:
            generated, _, _ = model(img, lab, mask_emb)
            save(generated,mask,label,intensity,generated_dir,generated_csv, total_generated_img,i)
            total_generated_img += 1
            n_Gp += 1
        if label == 'Golgi' and intensity == 'intermediate' and n_Gi<num_to_reach:
            generated, _, _ = model(img, lab, mask_emb)
            save(generated,mask,label,intensity,generated_dir,generated_csv, total_generated_img,i)
            total_generated_img += 1
            n_Gi += 1
        if label == 'NuMem' and intensity == 'positive' and n_NMp<num_to_reach:
            generated, _, _ = model(img, lab, mask_emb)
            save(generated,mask,label,intensity,generated_dir,generated_csv, total_generated_img,i)
            total_generated_img += 1
            n_NMp += 1
        if label == 'NuMem' and intensity == 'intermediate' and n_NMi<num_to_reach:
            generated, _, _ = model(img, lab, mask_emb)
            save(generated,mask,label,intensity,generated_dir,generated_csv, total_generated_img,i)
            total_generated_img += 1
            n_NMi += 1
    
    return total_generated_img, n_Hp, n_Hi, n_Sp, n_Si, n_Np, n_Ni, n_Cp, n_Ci, n_Gp, n_Gi, n_NMp, n_NMi


for num_model in range(9):
    print(f'Generation with model {num_model}...')
    model = CVAEMask().to(device)
    model.load_state_dict(torch.load(model_dir+"model0"+str(num_model)+".pth", map_location=torch.device(device)))
    total_generated_img,n_Hp, n_Hi, n_Sp, n_Si, n_Np, n_Ni,  n_Cp, n_Ci, n_Gp, n_Gi, n_NMp, n_NMi = generate(model, total_generated_img, num_model ,n_Hp, n_Hi, n_Sp, n_Si, n_Np, n_Ni,  n_Cp, n_Ci, n_Gp, n_Gi, n_NMp, n_NMi)

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


