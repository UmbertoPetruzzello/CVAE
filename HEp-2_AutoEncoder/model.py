import torch
import torch.nn as nn

# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4),
            nn.LeakyReLU(0.2),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4),
            nn.Sigmoid()
        )
        


    def forward(self, x):
        x =self.encoder(x)
        x= self.decoder(x)
        return x 

