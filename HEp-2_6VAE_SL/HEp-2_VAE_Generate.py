import numpy as np
import gc
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
import csv
import argparse

from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import dataloader

import matplotlib.pyplot as plt

from model import VAE
from utils import imshow


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using %s for computation" % device)


project_dir = '6VAE_model/'
images_dir = project_dir + 'data/'
model_dir = project_dir + 'models/'
masks_dir = project_dir + 'masks/'
generated_dir = project_dir + 'generated/'

if not(os.path.exists(generated_dir)):
    os.mkdir(generated_dir)


def generate(args):


    label = args.pattern
    intensity = args.intensity
    num_samples = args.numsamples

    # initialize the NN
    model = VAE().to(device)
    model.load_state_dict(torch.load(
    model_dir+label+"/model_epoch200.pth", map_location=torch.device(device)))

    transf = transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor()])


    for i in range(num_samples):
        
        #generated_csv = pd.read_csv('generated_images.csv')
        #total_generated_img = len(generated_csv)
        
    
        z = torch.randn(512).mul(1.0)
       
        with torch.no_grad():
            z = z.to(device)
            z = z.view(1,-1)
            x = model.fc4(z)
            x = model.fc5(x)
            x = model.decoder(x)
            #save_image(x.cpu(), generated_dir+str(total_generated_img)+'.png')
            save_image(x.cpu(),  generated_dir+label+str(i)+'.png')
            
            '''
            with open('generated_images.csv', 'a') as csvfile: 
                # creating a csv writer object 
                csvwriter = csv.writer(csvfile) 
                
                row = [str(total_generated_img)+'.png', label]
                    
                # writing the data rows 
                csvwriter.writerow(row)
           
                csvfile.close()
            '''


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, default='Centromere')
    parser.add_argument("--intensity", type=str, default='positive')
    parser.add_argument("--numsamples", type=int, default=5)

    args = parser.parse_args()

    generate(args)