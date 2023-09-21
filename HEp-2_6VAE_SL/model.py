import torch
import torch.nn as nn

hidden_size = 4096        # hidden dimension
hidden_size_complex = 1024
latent_size = 512          # latent vector dimension


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=4):
        return input.view(input.size(0), 256 , 6, 6)

class UnFlatten2(nn.Module):
    def forward(self, input, size=4):
        return input.view(input.size(0),1024, 1, 1)

####################MODEL VARIATIONAL FOR EACH CLASS####################
class VAE(nn.Module):
    def __init__(self, image_channels=1, hidden_size=hidden_size, latent_size=latent_size, num_classes=2):
        super(VAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2),
            nn.LeakyReLU(0.2),
            Flatten(),
        )
        self.encoder_mean = nn.Linear(hidden_size, latent_size)
        self.encoder_logvar = nn.Linear(hidden_size, latent_size)
        self.fc4 = nn.Linear(latent_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, 256*6*6)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc1 = nn.Linear(256*6*6, hidden_size)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(256, 128, 4, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, 6, 2),
            nn.Sigmoid()
        )

    def sample(self, log_var, mean):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(self, x):
        x = self.encoder(x) #9216
        x = self.fc1(x)     #latent_size 4096 
        x = self.fc2(x)
        log_var = self.encoder_logvar(x)  #512
        mean = self.encoder_mean(x)       #512
        z = self.sample(log_var, mean)    #512
        
        x = self.fc4(z)                   #4096
        x = self.fc5(x)                   #9216
        x = self.decoder(x)
        return x, mean, log_var


