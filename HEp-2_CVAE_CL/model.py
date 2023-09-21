import torch
import torch.nn as nn

hidden_size = 4096         # hidden dimension
latent_size = 512          # latent vector dimension


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=4):
        return input.view(input.size(0), 256 , 6, 6)


####################MODEL CONDITIONAL WITHOUT MASK####################
class CVAE(nn.Module):
    def __init__(self, image_channels=1, hidden_size=hidden_size, latent_size=latent_size, num_classes=2):
        super(CVAE, self).__init__()
        
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
        self.fc3 = nn.Linear(latent_size, latent_size - num_classes)
        self.fc4 = nn.Linear(latent_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, 256*6*6)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc1 = nn.Linear(256*6*6, hidden_size - num_classes)

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

    def forward(self, x, a):
        x = self.encoder(x) #9216
        x = self.fc1(x)     #latent_size 4096 - 2 - 128 = 3966


        x = torch.cat((x, a), 1)  #4096
        x = self.fc2(x)           #4096

        log_var = self.encoder_logvar(x)  #512
        mean = self.encoder_mean(x)       #512
        z = self.sample(log_var, mean)    #512
        
        z = self.fc3(z)                   #382
        z = torch.cat((z, a), 1)          #512
        x = self.fc4(z)                   #4096
        x = self.fc5(x)                   #9216
        x = self.decoder(x)
        return x, mean, log_var



####################MODEL CONDITIONAL WITH MASK####################
class CVAEMask(nn.Module):
    def __init__(self, image_channels=1, hidden_size=hidden_size, latent_size=latent_size, num_classes=2):
        super(CVAEMask, self).__init__()
        
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
        self.fc3 = nn.Linear(latent_size, latent_size - num_classes - 128)
        self.fc4 = nn.Linear(latent_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, 256*6*6)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc1 = nn.Linear(256*6*6, hidden_size - num_classes - 128)

        self.fc_mask = nn.Linear(16384, 128)

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

    def forward(self, x, a, mask):
        x = self.encoder(x) #9216
        x = self.fc1(x)     #latent_size 4096 - 2 - 128 = 3966

        mask = self.fc_mask(mask)

        x = torch.cat((x, a, mask), 1)  #4096
        x = self.fc2(x)           #4096

        log_var = self.encoder_logvar(x)  #512
        mean = self.encoder_mean(x)       #512
        z = self.sample(log_var, mean)    #512
        
        z = self.fc3(z)                   #382
        z = torch.cat((z, a, mask), 1)    #512
        x = self.fc4(z)                   #4096
        x = self.fc5(x)                   #9216
        x = self.decoder(x)
        return x, mean, log_var
        
 
 