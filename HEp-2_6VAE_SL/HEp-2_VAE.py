import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from torchvision.utils import save_image
from tqdm import tqdm

from torch.utils.data import DataLoader
import dataloader

from model import VAE
from utils import imshow



batch_size = 32
# number of epochs to train the model
epochs = 500
hidden_size = 4096        # hidden dimension
latent_size = 512          # latent vector dimension
image_channels = 1 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using %s for computation" % device)

pattern_class = 'NuMem'

project_dir = ''

dataset_dir = project_dir + 'csv_byclass/gt_training_'+pattern_class+'.csv'

images_dir = project_dir + 'train/'
model_dir = project_dir + 'models/' + pattern_class +'/'
output_dir = project_dir + 'outputs/' + pattern_class +'/'
original_dir = output_dir + 'original/'
reconstructed_dir = output_dir + 'reconstructed/' 

if not(os.path.exists(model_dir)):
    os.mkdir(model_dir)

if not(os.path.exists(output_dir)):
    os.mkdir(output_dir)

if not(os.path.exists(original_dir)):
    os.mkdir(original_dir)

if not(os.path.exists(reconstructed_dir)):
    os.mkdir(reconstructed_dir)



print("...Loading data")
data = dataloader.HEP2Dataset(dataset_dir, images_dir)
print("Finished!")

print(data.__len__())

train_size = int(data.__len__())
test_size = data.__len__() - train_size
train_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])


val_size = int(0.2 * train_size)
train_size = train_size - val_size
train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, val_size])

print('training set size: {}'.format(train_size))
print('validation set size: {}'.format(val_size))
print('test set size: {}'.format(test_size))



trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
validloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)


# initialize the NN
model = VAE().to(device)
print(model)            


optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.BCELoss()

writer = SummaryWriter()



def final_loss(bce_loss, mu, logvar, i):
    """
    This function will add the reconstruction loss (BCELoss) and the 
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    a = 0.01
    if (i % 3 == 0):
        a = 0.5
    BCE = 1000*bce_loss 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE +  a*KLD , a*KLD

min_train_loss = 1000000

def fit(model, dataloader, num_epoch):
    global min_train_loss

    model.train()
    running_loss = 0.0
    running_bce_loss = 0.0
    running_kld_loss = 0.0
    num_samples = len(dataloader.dataset)
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        images, _, _ = data
       
        images= images.to(device)
       

        optimizer.zero_grad()
        reconstruction, mu, logvar = model(images)
        
        if num_epoch % 3 == 0 or num_epoch == 0:
            if i == int(len(train_data)/dataloader.batch_size) -1 :
                to_print = images.cpu().detach().numpy()
                imshow(to_print[0], file_name = original_dir + 'original_image_Epoch:{}'.format(epoch))

                #print('massimo valore img {}'. format(np.max(to_print)))

                to_print = reconstruction.cpu().detach().numpy()
                imshow(to_print[0], file_name = reconstructed_dir+'reconstructed_Epoch:{}'.format(epoch))

                #print('massimo valore reco {}'. format(np.max(to_print)))

        
        #reconstruction = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        if torch.isnan(reconstruction).any():
            print('Ciclo saltato per vettore nan\n')

            num_samples -= 1
        else: 
            bce_loss = criterion(reconstruction, images)

            loss, kld_loss= final_loss(bce_loss, mu, logvar, i)

            loss.backward()
            #gradient clipping
            #torch.nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=1, norm_type=2.0)
            optimizer.step()


            running_loss += loss.item()
            running_bce_loss += bce_loss
            running_kld_loss += kld_loss

    train_loss = running_loss/num_samples
    bce_loss = running_bce_loss/num_samples
    kld_loss =  running_kld_loss/num_samples
    
    if epoch % 10 == 0:
        torch.save(model.state_dict(), model_dir + 'model_epoch{}.pth'.format(epoch))
    return train_loss, bce_loss, kld_loss

def validate(model, dataloader, num_epoch):
    model.eval()
    running_loss = 0.0
    running_bce_loss = 0.0
    running_kld_loss = 0.0
    num_samples = len(dataloader.dataset)
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(valid_data)/dataloader.batch_size)):
            images, _, _ = data
            images= images.to(device)

            reconstruction, mu, logvar = model(images)

            if torch.isnan(reconstruction).any():
                print('Ciclo saltato per vettore nan')
                num_samples -= 1
            
            else:
                bce_loss = criterion(reconstruction, images)
                loss, kld_loss = final_loss(bce_loss, mu, logvar, i)

                running_loss += loss.item()
                running_bce_loss += bce_loss
                running_kld_loss += kld_loss
        
            # save the last batch input and output of every epoch
            if i == int(len(valid_data)/dataloader.batch_size) - 1:
                num_rows = 8
                both = torch.cat((images[:8], 
                                  reconstruction[:8]))
                save_image(both.cpu(), output_dir+f"output{epoch}.png", nrow=num_rows)
    val_loss = running_loss/num_samples
    bce_loss = running_bce_loss/num_samples
    kld_loss =  running_kld_loss/num_samples
    
    
    return val_loss, bce_loss, kld_loss

train_loss = []
val_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch} of {epochs}")
    train_epoch_loss, t_bce_epoch_loss, t_kld_epoch_loss = fit(model, trainloader, epoch)
    val_epoch_loss, v_bce_epoch_loss, v_kld_epoch_loss  = validate(model, validloader, epoch)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {val_epoch_loss:.4f}")
    # Append-adds at last
    
    file = open("loss.txt","a")#append mode
    file.write(f"Epoch {epoch} of {epochs}\n")
    file.write(f"Train Loss: {train_epoch_loss:.4f}\n")
    file.write(f"BCE Train Loss: {t_bce_epoch_loss:.4f}\n")
    file.write(f"KLD Train Loss: {t_kld_epoch_loss:.4f}\n")
    file.write(f"Validation Loss: {val_epoch_loss:.4f}\n")
    file.write(f"BCE Validation Loss: {v_bce_epoch_loss:.4f}\n")
    file.write(f"KLD Validation Loss: {v_kld_epoch_loss:.4f}\n")

    file.close()
    

    writer.add_scalar("Loss/train", train_epoch_loss, epoch)
    writer.add_scalar("BCE/train", t_bce_epoch_loss, epoch)
    writer.add_scalar("KLD/train", t_kld_epoch_loss, epoch)


    writer.add_scalar("Loss/validation", val_epoch_loss, epoch)
    writer.add_scalar("BCE/validation", v_bce_epoch_loss, epoch)
    writer.add_scalar("KLD/validation", v_kld_epoch_loss, epoch)

writer.flush()
writer.close()


