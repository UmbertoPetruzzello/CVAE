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

from PIL import Image


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
plt.ion()  # interactive mode


class HEP2Dataset(Dataset):
    """HEp-2 dataset."""

    def __init__(self, csv_file, root_dir, transform=False, num_classes=12):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): whether apply or not DA.
        """
        old_frame = pd.read_csv(csv_file, names=["Image", "Mask", "Label", "Intensity"])
        self.total_old = len(old_frame)
        self.frame = pd.DataFrame(columns=["Image", "Mask", "Label", "Intensity","Aug"])
        self.num_classes = num_classes
        if transform:
            #for i in range(self.total_old):
            for i in range(10):
                row_0={'Image':old_frame.loc[i]["Image"],'Mask':old_frame.loc[i]["Mask"],'Label':old_frame.loc[i]["Label"],'Intensity': old_frame.loc[i]["Intensity"],'Aug': 0}
                row_1={'Image':old_frame.loc[i]["Image"],'Mask':old_frame.loc[i]["Mask"],'Label':old_frame.loc[i]["Label"],'Intensity': old_frame.loc[i]["Intensity"],'Aug': 1}
                row_2={'Image':old_frame.loc[i]["Image"],'Mask':old_frame.loc[i]["Mask"],'Label':old_frame.loc[i]["Label"],'Intensity': old_frame.loc[i]["Intensity"],'Aug': 2}
                row_3={'Image':old_frame.loc[i]["Image"],'Mask':old_frame.loc[i]["Mask"],'Label':old_frame.loc[i]["Label"],'Intensity': old_frame.loc[i]["Intensity"],'Aug': 3}
                self.frame = self.frame.append(row_0, ignore_index=True)
                self.frame = self.frame.append(row_1, ignore_index=True)
                self.frame = self.frame.append(row_2, ignore_index=True)
                self.frame = self.frame.append(row_3, ignore_index=True)
        else:
            for i in range(self.total_old):
                row_0={'Image':old_frame.loc[i]["Image"],'Mask':old_frame.loc[i]["Mask"],'Label':old_frame.loc[i]["Label"],'Intensity': old_frame.loc[i]["Intensity"],'Aug': 0}
                self.frame = self.frame.append(row_0, ignore_index=True)
        self.total = len(self.frame)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        aug = os.path.join(self.root_dir, str(self.frame.iloc[idx, 4]))
        if  aug == 0:
            img_name = os.path.join(self.root_dir, self.frame.iloc[idx, 0])
            image = io.imread(img_name)
            mask_name = os.path.join(self.root_dir, self.frame.iloc[idx, 1])
            mask = io.imread(mask_name)
            label = self.frame.iloc[idx, 2]
            
            if label == 'Homogeneous':
                lab = 0
            elif label == 'Speckled':
                lab = 1
            elif label == 'Nucleolar':
                lab = 2
            elif label == 'Centromere':
                lab = 3
            elif label == 'Golgi':
                lab = 4
            elif label == 'NuMem':
                lab = 5
            else: # mistp
                lab = 6
            
            intensity = self.frame.iloc[idx, 3]
            
            if intensity == 'positive':
                inten = 0
            else:
                inten = 1
            
        else:
            img_name = os.path.join(self.root_dir, self.frame.iloc[idx, 0])
            image = io.imread(img_name)
            mask_name = os.path.join(self.root_dir, self.frame.iloc[idx, 1])
            mask = io.imread(mask_name)
            if aug == 1:
                image = imutils.rotate(image, 90)
                mask = imutils.rotate(mask, 90)
            elif aug == 2:
                image = imutils.rotate(image, 180)
                mask = imutils.rotate(mask, 180)
            #aug == 3:
            else:
                image = imutils.rotate(image, 270)
                mask = imutils.rotate(mask, 270)
            label = self.frame.iloc[idx, 2]
            intensity = self.frame.iloc[idx, 3]

            if label == 'Homogeneous' and intensity == 'positive':
                lab = 0
            if label == 'Homogeneous' and intensity != 'positive':
                lab = 1
            elif label == 'Speckled' and intensity == 'positive':
                lab = 2
            elif label == 'Speckled' and intensity != 'positive':
                lab = 3
            elif label == 'Nucleolar'and intensity == 'positive':
                lab = 4
            elif label == 'Nucleolar'and intensity != 'positive':
                lab = 5
            elif label == 'Centromere' and intensity == 'positive':
                lab = 6
            elif label == 'Centromere' and intensity != 'positive':
                lab = 7
            elif label == 'Golgi' and intensity == 'positive':
                lab = 8
            elif label == 'Golgi' and intensity != 'positive':
                lab = 9
            elif label == 'NuMem' and intensity == 'positive':
                lab = 10
            elif label == 'NuMem' and intensity != 'positive':
                lab = 11
            
                        
            if intensity == 'positive':
                inten = 0
            else:
                inten = 1
        
        #FOR EXPERIMENT WITH CYTOPLASM
        image = Image.open(img_name)
        transf = transforms.Compose([transforms.Resize((128,128)),transforms.Grayscale(num_output_channels=1),transforms.ToTensor()])
        image = transf(image)
         
        mask = Image.open(mask_name)
        mask = transf(mask)
        
        lab = torch.as_tensor(int(lab), dtype=torch.int16)

        #FOR EXPERIMENT WITHOUT CYTOPLASM 
        '''
        mask = Image.open(mask_name)
        mask = transf(mask)
        
        image = image.detach().numpy()
        mask = mask.detach().numpy()
        prod= np.multiply(image,mask)

        image = torch.from_numpy(prod)
        #save_image(image, './product_image_mask/img'+str(idx)+'.png')

        lab = torch.as_tensor(int(lab), dtype=torch.int16)
        '''
        return image, lab