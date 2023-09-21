import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import os

import matplotlib.pyplot as plt

import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim

from dataloader_classifier import HEP2Dataset

from torch.utils.tensorboard import SummaryWriter

from ResNet import Bottleneck, ResNet, ResNet50

batch_size = 32
num_classes = 12

def eval_model_train(model, trainLoader, device, tra_acc_list):
    correct = 0
    total = 0
    MCA = 0.0
    MCA_array = np.zeros(num_classes)
    with torch.no_grad():
        for data in trainLoader:
            images, labels = data
            labels = labels.type(torch.LongTensor)
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for p in range(labels.size(0)):
                MCA_array = MCA_array + 1
                if predicted[p] != labels[p]:
                    MCA_array[predicted[p]] -= 1
                    MCA_array[labels[p]] -= 1
    
    SCA = (MCA_array/(len(trainLoader)*batch_size))
    MCA = sum(SCA)/num_classes
    print('Accuracy of trainloader: %d %%' % (100 * correct / total))
    print('MCA of trainloader: {}'.format(MCA))
    print('SCA of trainloader:')
    print(SCA)
    tra_acc_list.append(100 * correct / total)

    return MCA



def eval_model_test(model, testLoader, device, fold):
    correct = 0
    total = 0
    MCA = 0.0
    MCA_array = np.zeros(num_classes)
    length = 0

    if fold == 4:
        length = 2720
    else:
        length = 2719
    
    y_test_list = []
    y_pred_list = []

    with torch.no_grad():
        for data in testLoader:
            images, labels = data
            labels = labels.type(torch.LongTensor)
            images, labels = images.to('cuda'), labels.to('cuda')

            y_test_list.extend(labels.cpu().numpy())

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            y_pred_list.extend(predicted.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for p in range(labels.size(0)):
                MCA_array = MCA_array + 1
                if predicted[p] != labels[p]:
                    MCA_array[predicted[p]] -= 1
                    MCA_array[labels[p]] -= 1
    
    SCA = (MCA_array/length)
    MCA = sum(SCA)/num_classes
    print('Accuracy of test: %d %%' % (100 * correct / total))
    print('MCA of test: {}'.format(MCA))
    print('SCA of test:')
    print(SCA)


    # Display the confusion matrix as a heatmap
    arr = confusion_matrix(y_test_list, y_pred_list)
    print(arr)
    if num_classes == 6:
        class_names = [' Homogeneous', '  Speckled', ' Nucleolar', '  Centromere', ' Golgi', '  NuMem']
    elif num_classes == 12:
        class_names = [' Homogeneous pos',' Homogeneous inter', '  Speckled pos', '  Speckled inter', ' Nucleolar pos',' Nucleolar inter', '  Centromere pos', '  Centromere inter',  ' Golgi pos', ' Golgi inter',  '  NuMem pos', '  NuMem inter']
    df_cm = pd.DataFrame(arr, class_names, class_names)
    plt.figure(figsize = (9,6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
    plt.xlabel("prediction")
    plt.ylabel("label (ground truth)")
    plt.show()
    plt.savefig('confusion_matrix_fold'+str(fold)+'.png', bbox_inches = "tight", pad_inches = 0.0)
    return MCA




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using %s for computation" % device)

project_dir = ''
dataset_dir = project_dir + '../k-fold_csv/'
images_dir = project_dir + '../../train/'
augmented_dir = project_dir + '../augmented_images/'
generated_dir = project_dir + '../generated_images/'

experiment = 'Generated Augmented' #Undersampling, Traditional Augmentation, Generative Augmentation, None, Generated Augmented

if not(os.path.exists('./models')):
    os.mkdir('./models')

print("...Loading data")
fold1 = HEP2Dataset(dataset_dir+'1-fold.csv', images_dir)
fold2 = HEP2Dataset(dataset_dir+'2-fold.csv', images_dir)
fold3 = HEP2Dataset(dataset_dir+'3-fold.csv', images_dir)
fold4 = HEP2Dataset(dataset_dir+'4-fold.csv', images_dir)
fold5 = HEP2Dataset(dataset_dir+'5-fold.csv', images_dir)
print("Finished!")

if experiment == 'Undersampling':
    print("...Loading data undersampled")
    under_fold1 = HEP2Dataset(dataset_dir+'under_1-fold.csv', images_dir)
    under_fold2 = HEP2Dataset(dataset_dir+'under_2-fold.csv', images_dir)
    under_fold3 = HEP2Dataset(dataset_dir+'under_3-fold.csv', images_dir)
    under_fold4 = HEP2Dataset(dataset_dir+'under_4-fold.csv', images_dir)
    under_fold5 = HEP2Dataset(dataset_dir+'under_5-fold.csv', images_dir)
    print("Finished!")

    
if experiment == 'Traditional Augmentation' or experiment == 'Generated Augmented':
    print("Loading data augmented")
    aug_fold1 = HEP2Dataset(dataset_dir+'aug_1-fold.csv', augmented_dir + '1-fold')
    aug_fold2 = HEP2Dataset(dataset_dir+'aug_2-fold.csv', augmented_dir + '2-fold')
    aug_fold3 = HEP2Dataset(dataset_dir+'aug_3-fold.csv', augmented_dir + '3-fold')
    aug_fold4 = HEP2Dataset(dataset_dir+'aug_4-fold.csv', augmented_dir + '4-fold')
    aug_fold5 = HEP2Dataset(dataset_dir+'aug_5-fold.csv', augmented_dir + '5-fold')

if experiment == 'Generative Augmentation' or experiment == 'Generated Augmented':
    print("Loading data generated")
    gen_fold1 = HEP2Dataset(dataset_dir+'gen_1-fold.csv', generated_dir + '1-fold')
    gen_fold2 = HEP2Dataset(dataset_dir+'gen_2-fold.csv', generated_dir + '2-fold')
    gen_fold3 = HEP2Dataset(dataset_dir+'gen_3-fold.csv', generated_dir + '3-fold')
    gen_fold4 = HEP2Dataset(dataset_dir+'gen_4-fold.csv', generated_dir + '4-fold')
    gen_fold5 = HEP2Dataset(dataset_dir+'gen_5-fold.csv', generated_dir + '5-fold')

'''
if experiment == 'Generated Augmented':
    print("Loading data generated augmented")

    gen_aug_fold1 = HEP2Dataset(dataset_dir+'gen_aug_1-fold.csv', '../generated_augmented_images/1-fold')
    gen_aug_fold2 = HEP2Dataset(dataset_dir+'gen_aug_2-fold.csv', '../generated_augmented_images/2-fold')
    gen_aug_fold3 = HEP2Dataset(dataset_dir+'gen_aug_3-fold.csv', '../generated_augmented_images/3-fold')
    gen_aug_fold4 = HEP2Dataset(dataset_dir+'gen_aug_4-fold.csv', '../generated_augmented_images/4-fold')
    gen_aug_fold5 = HEP2Dataset(dataset_dir+'gen_aug_5-fold.csv', '../generated_augmented_images/5-fold')
'''
#itera per i 5 fold
for fold in range(5):

    model = ResNet50(num_classes = num_classes, channels = 1).to('cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5)


    #1-FOLD
    if fold == 0:
        print("Doing 1-FOLD cross validation...")
        writer = SummaryWriter()
        train_data = torch.utils.data.ConcatDataset([fold2,fold3,fold4,fold5])
        test_data = fold1

        if experiment == 'Undersampling':
            train_data = torch.utils.data.ConcatDataset([under_fold2,under_fold3,under_fold4,under_fold5])

        if experiment == 'Traditional Augmentation' or experiment == 'Generated Augmented':
            train_data = torch.utils.data.ConcatDataset([train_data,aug_fold2,aug_fold3,aug_fold4,aug_fold5])
        
        if experiment == 'Generative Augmentation' or experiment == 'Generated Augmented':
            train_data = torch.utils.data.ConcatDataset([train_data,gen_fold2,gen_fold3,gen_fold4,gen_fold5])
        
        #if experiment == 'Generated Augmented':
        #    train_data = torch.utils.data.ConcatDataset([train_data,gen_aug_fold2,gen_aug_fold3,gen_aug_fold4,gen_aug_fold5])

        
    #2-FOLD
    if fold == 1:
        print("Doing 2-FOLD cross validation...")
        writer = SummaryWriter()
        train_data = torch.utils.data.ConcatDataset([fold1,fold3,fold4,fold5])
        test_data = fold2

        if experiment == 'Undersampling':
            train_data = torch.utils.data.ConcatDataset([under_fold1,under_fold3,under_fold4,under_fold5])

        if experiment == 'Traditional Augmentation' or experiment == 'Generated Augmented':
            train_data = torch.utils.data.ConcatDataset([train_data,aug_fold1,aug_fold3,aug_fold4,aug_fold5])
        
        if experiment == 'Generative Augmentation' or experiment == 'Generated Augmented':
            train_data = torch.utils.data.ConcatDataset([train_data,gen_fold1,gen_fold3,gen_fold4,gen_fold5])
        
        #if experiment == 'Generated Augmented':
        #    train_data = torch.utils.data.ConcatDataset([train_data,gen_aug_fold1,gen_aug_fold3,gen_aug_fold4,gen_aug_fold5])


    #3-FOLD
    if fold == 2:
        print("Doing 3-FOLD cross validation...")
        writer = SummaryWriter()
        train_data = torch.utils.data.ConcatDataset([fold1,fold2,fold4,fold5])
        test_data = fold3

        if experiment == 'Undersampling':
            train_data = torch.utils.data.ConcatDataset([under_fold1,under_fold2,under_fold4,under_fold5])

        if experiment == 'Traditional Augmentation' or experiment == 'Generated Augmented':
            train_data = torch.utils.data.ConcatDataset([train_data,aug_fold1,aug_fold2,aug_fold4,aug_fold5])
        
        if experiment == 'Generative Augmentation' or experiment == 'Generated Augmented':
            train_data = torch.utils.data.ConcatDataset([train_data,gen_fold1,gen_fold2,gen_fold4,gen_fold5])
        
        #if experiment == 'Generated Augmented':
        #    train_data = torch.utils.data.ConcatDataset([train_data,gen_aug_fold1,gen_aug_fold3,gen_aug_fold4,gen_aug_fold5])

    #4-FOLD
    if fold == 3:
        print("Doing 4-FOLD cross validation...")
        writer = SummaryWriter()
        train_data = torch.utils.data.ConcatDataset([fold1,fold2,fold3,fold5])
        test_data = fold4

        if experiment == 'Undersampling':
            train_data = torch.utils.data.ConcatDataset([under_fold1,under_fold2,under_fold3,under_fold5])

        if experiment == 'Traditional Augmentation' or experiment == 'Generated Augmented':
            train_data = torch.utils.data.ConcatDataset([train_data,aug_fold1,aug_fold2,aug_fold3,aug_fold5])
        
        if experiment == 'Generative Augmentation' or experiment == 'Generated Augmented':
            train_data = torch.utils.data.ConcatDataset([train_data,gen_fold1,gen_fold2,gen_fold3,gen_fold5])
        
        #if experiment == 'Generated Augmented':
        #    train_data = torch.utils.data.ConcatDataset([train_data,gen_aug_fold1,gen_aug_fold3,gen_aug_fold4,gen_aug_fold5])
        

    #5-FOLD
    if fold == 4:
        print("Doing 5-FOLD cross validation...")
        writer = SummaryWriter()
        train_data = torch.utils.data.ConcatDataset([fold1,fold2,fold3,fold4])
        test_data = fold5

        if experiment == 'Undersampling':
            train_data = torch.utils.data.ConcatDataset([under_fold1,under_fold2,under_fold3,under_fold4])

        if experiment == 'Traditional Augmentation' or experiment == 'Generated Augmented':
            train_data = torch.utils.data.ConcatDataset([train_data,aug_fold1,aug_fold2,aug_fold3,aug_fold4])
        
        if experiment == 'Generative Augmentation' or experiment == 'Generated Augmented':
            train_data = torch.utils.data.ConcatDataset([train_data,gen_fold1,gen_fold2,gen_fold3,gen_fold4])
        
        #if experiment == 'Generated Augmented':
        #    train_data = torch.utils.data.ConcatDataset([train_data,gen_aug_fold1,gen_aug_fold3,gen_aug_fold4,gen_aug_fold5])

      
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    EPOCHS = 30
    iteration = 10
    tra_acc_list, loss_list = [], []
    for epoch in range(EPOCHS):
        losses = []
        running_loss = 0.0
        MCA = 0.0
        for i, inp in enumerate(trainloader):
            inputs, labels = inp
            labels = labels.type(torch.LongTensor)
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            if i % iteration == iteration - 1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / iteration))
                loss_list.append(running_loss / iteration)
            running_loss = 0.0

        avg_loss = sum(losses)/len(losses)
        scheduler.step(avg_loss)
        MCA = eval_model_train(model, trainloader, device, tra_acc_list)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("MCA/train", MCA, epoch)

                
    print('Training Done')
    writer.flush()
    writer.close()
    MCA_test = eval_model_test(model, testloader, device, fold)
    file = open("evaluate_"+ experiment+".txt","a")#append mode
    file.write(f"Fold {fold+1} of 5\n")
    file.write(f"Mean Class Accuracy: {MCA_test:.4f}\n")

    file.close()

    torch.save(model, './models/model_fold-'+str(fold)+'.pth')
