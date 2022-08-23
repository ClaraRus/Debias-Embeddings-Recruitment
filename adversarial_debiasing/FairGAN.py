import torch 
from torch import optim
import numpy as np
import pandas as pd
import tqdm
from itertools import repeat

class Encoder(torch.nn.Module):
    def __init__(self, hidden_size, input_dim, output_dim, hidden_dim):
        super(Encoder, self).__init__()    
        self.input_layer = self.init_layer(input_dim, hidden_dim)
        
        self.hidden_layer1 = self.init_layer(hidden_dim, hidden_dim)
        self.hidden_layer2 = self.init_layer(hidden_dim, hidden_dim)
        self.hidden_layer3 = self.init_layer(hidden_dim, hidden_dim)
                    
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)

    def init_layer(self, input_dim, out_dim):
        return torch.nn.Sequential(
          torch.nn.Linear(input_dim, out_dim),
          torch.nn.LeakyReLU(),
          torch.nn.BatchNorm1d(out_dim)
        )
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.output_layer(x)
        return x
    
    
class Decoder(torch.nn.Module):
    def __init__(self, hidden_size, input_dim, output_dim, hidden_dim):
        super(Decoder, self).__init__()    
        self.input_layer = self.init_layer(input_dim, hidden_dim)
        
        self.hidden_layer1 = self.init_layer(hidden_dim, hidden_dim)
        self.hidden_layer2 = self.init_layer(hidden_dim, hidden_dim)
        self.hidden_layer3 = self.init_layer(hidden_dim, hidden_dim)
                    
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)

    def init_layer(self, input_dim, out_dim):
        return torch.nn.Sequential(
          torch.nn.Linear(input_dim, out_dim),
          torch.nn.LeakyReLU(),
          torch.nn.BatchNorm1d(out_dim)
        )
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.output_layer(x)
        return x
    
    
    
class Classifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, logits=False):
        super(Classifier, self).__init__()    
        self.input_layer = self.init_layer(input_dim, hidden_dim)
        
        self.hidden_layer1 = self.init_layer(hidden_dim, hidden_dim)
   
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)
        self.logits = logits
        if not self.logits:
            self.sigmoid = torch.nn.Sigmoid()

    def init_layer(self, input_dim, out_dim):
        return torch.nn.Sequential(
          torch.nn.Linear(input_dim, out_dim),
          torch.nn.LeakyReLU(),
        )
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer1(x)
        x = self.output_layer(x)
        if not self.logits:
            x = self.sigmoid(x)
        return x
    
class Gender_Discriminator(torch.nn.Module):
    # Discriminator for Gender
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(Gender_Discriminator, self).__init__()    
        self.input_layer = self.init_layer(input_dim, hidden_dim)
        
        
        self.hidden_layer1 = self.init_layer(hidden_dim, hidden_dim)
                
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def init_layer(self, input_dim, out_dim):
        return torch.nn.Sequential(
          torch.nn.Linear(input_dim, out_dim),
          torch.nn.LeakyReLU(),
        )
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer1(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x
    
    
class FairGAN(torch.nn.Module):
    def __init__(self, chans, n_output, n_nuisance, logits=False):
        super(FairGAN, self).__init__()    
        # Input, data set and model training scheme parameters
        self.chans = chans # channels
        self.n_output = n_output # size of the output of the classification model
        self.n_nuisance = n_nuisance # size of the output of the adversary
        self.latent_dim = self.chans
        self.logits = logits
        # Build the network blocks
        self.enc = Encoder(hidden_size=3, input_dim=self.chans, output_dim=self.latent_dim, hidden_dim=128)
        self.cla = Classifier(input_dim=self.latent_dim, output_dim=self.n_output, hidden_dim=128, logits=logits)
        self.adv = Gender_Discriminator(input_dim=self.latent_dim, output_dim=self.n_nuisance, hidden_dim=128)
        self.dec = Decoder(hidden_size=3, input_dim=self.latent_dim, output_dim=self.chans, hidden_dim=128)
        
    def forward(self, x):
        x = self.enc(x)
        x1 = self.cla(x)
        x2 = self.adv(x)
        x3 = self.dec(x)
        return x1, x2, x3