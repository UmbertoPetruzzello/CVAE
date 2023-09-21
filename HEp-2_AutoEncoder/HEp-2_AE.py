import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from model import ConvAutoencoder
from torch.utils.data import DataLoader
import dataloader


# helper function to un-normalize and display an image
def imshow(img, file_name = "reconstructed_image_conv2.png"):
    #img = img / 2 + 0.5  # unnormalize
    img = np.transpose(img, (1,2,0))
    img = img[:, :, 0]
    plt.imshow(img, cmap="gray")
    plt.savefig(file_name, bbox_inches = "tight", pad_inches = 0.0)
    print("done!")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using %s for computation" % device)


project_dir = ''
dataset_dir = project_dir + 'final_patches.csv'
images_dir = project_dir + 'data/'
model_dir = project_dir + 'models/'


batch_size = 1

print("...Loading data")
data = dataloader.HEP2Dataset(dataset_dir, images_dir)
print("Finished!")

train_size = int(0.1 * data.__len__())
test_size = data.__len__() - train_size
train_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])

val_size = int(0.3 * train_size)
train_size = train_size - val_size
train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, val_size])


trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
validloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# initialize the NN
model = ConvAutoencoder().to(device)
print(model)            


#criterion = nn.MSELoss()
criterion = nn.BCE()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


# number of epochs to train the model
n_epochs = 500

for epoch in range(1, n_epochs+1):
    print("Epoch : {}".format(epoch))
    # monitor training loss
    train_loss = 0.0
    ###################
    # train the model #
    ###################
    for i, data in enumerate(trainloader):
        # _ stands in for labels, here
        # no need to flatten images
        images, _ = data
        images= images.to(device)

        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(images)
        
        if epoch % 5 == 0 or epoch == 1 :
            if i == 0 :
                to_print = images.cpu().detach().numpy()
                imshow(to_print[0], file_name = 'outputs/original/original_image{}_Epoch:{}'.format(i,epoch))

                to_print = outputs.cpu().detach().numpy()
                imshow(to_print[0], file_name = 'outputs/reconstructed/reconstructed_image{}_Epoch:{}'.format(i,epoch))
                
                loss =criterion(outputs, images)
                print(loss.item())

        # calculate the loss
        loss =criterion(outputs, images)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()*images.size(0)
            
    # print avg training statistics 
    train_loss = train_loss/len(trainloader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, 
        train_loss
        ))
    torch.save(model.state_dict(), model_dir + 'model_AE_epoch{}'.format(epoch)+'.pth')

'''
model.load_state_dict(torch.load('models/model_conv_v2_epoch40.pth'))

# obtain one batch of test images
dataiter = iter(testloader)
images, labels = dataiter.next()

images = images.to(device)

# get sample outputs
output = model(images)


# output is resized into a batch of iages
#output = output.view(batch_size, 1, 384, 384)
# use detach when it's an output that requires_grad
output = output.cpu().detach().numpy()
for i in range(4):
    imshow(output[i], file_name ="reconstructed/test_convolutionalAE/test_image_model_conv_v2_{}.png".format(i))


# plot the first ten input images and then reconstructed images
#fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(24,4))
#for idx in np.arange(20):
#    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
#    #imshow(output[idx])
'''