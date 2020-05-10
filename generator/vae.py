from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Lambda(lambda x: x.view(-1))])

dataset_train = datasets.MNIST(
    '~/mnist', 
    train=True, 
    download=True, 
    transform=transform)
dataset_valid = datasets.MNIST(
    '~/mnist', 
    train=False, 
    download=True, 
    transform=transform)

dataloader_train = utils.data.DataLoader(dataset_train,
                                          batch_size=1000,
                                          shuffle=True,
                                          num_workers=4)
dataloader_valid = utils.data.DataLoader(dataset_valid,
                                          batch_size=1000,
                                          shuffle=True,
                                          num_workers=4)



device = 'cuda'

class VAE(nn.Module):
    def __init__(self, z_dim):
      super(VAE, self).__init__()
      self.dense_enc1 = nn.Linear(28*28, 200)
      self.dense_enc2 = nn.Linear(200, 200)
      self.dense_encmean = nn.Linear(200, z_dim)
      self.dense_encvar = nn.Linear(200, z_dim)
      self.dense_dec1 = nn.Linear(z_dim, 200)
      self.dense_dec2 = nn.Linear(200, 200)
      self.dense_dec3 = nn.Linear(200, 28*28)
    
    def _encoder(self, x):
      x = F.relu(self.dense_enc1(x))
      x = F.relu(self.dense_enc2(x))
      mean = self.dense_encmean(x)
      var = F.softplus(self.dense_encvar(x))
      return mean, var
    
    def _sample_z(self, mean, var):
      epsilon = torch.randn(mean.shape).to(device)
      return mean + torch.sqrt(var) * epsilon
 
    def _decoder(self, z):
      x = F.relu(self.dense_dec1(z))
      x = F.relu(self.dense_dec2(x))
      x = F.sigmoid(self.dense_dec3(x))
      return x

    def forward(self, x):
      mean, var = self._encoder(x)
      z = self._sample_z(mean, var)
      x = self._decoder(z)
      return x, z
    
    def loss(self, x):
      mean, var = self._encoder(x)
      KL = -0.5 * torch.mean(torch.sum(1 + torch.log(var) - mean**2 - var))
      z = self._sample_z(mean, var)
      y = self._decoder(z)
      reconstruction = torch.mean(torch.sum(x * torch.log(y) + (1 - x) * torch.log(1 - y)))
      lower_bound = [-KL, reconstruction]                                      
      return -sum(lower_bound)

def main():
    model = VAE(10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for i in range(20):
      losses = []
      for x, t in dataloader_train:
          x = x.to(device)
          model.zero_grad()
          y = model(x)
          loss = model.loss(x)
          loss.backward()
          optimizer.step()
          losses.append(loss.cpu().detach().numpy())
      print("EPOCH: {} loss: {}".format(i, np.average(losses)))
    
    
    model.eval()
    zs = []
    for x, t in dataloader_valid:
        # original
        for i, im in enumerate(x.view(-1, 28, 28).detach().numpy()[:10]):
          ax = fig.add_subplot(3, 10, i+1, xticks=[], yticks=[])
          ax.imshow(im, 'gray')
        x = x.to(device)
        # generate from x
        y, z = model(x)
        zs.append(z)
        y = y.view(-1, 28, 28)
        for i, im in enumerate(y.cpu().detach().numpy()[:10]):
          ax = fig.add_subplot(3, 10, i+11, xticks=[], yticks=[])
          ax.imshow(im, 'gray')
        # generate from z
        # 適当なzを以下で生成.
        z1to0 = torch.cat([z[1] * (i * 0.1) + z[0] * ((9 - i) * 0.1) for i in range(10)])
        y2 = model._decoder(z1to0).view(-1, 28, 28)
        for i, im in enumerate(y2.cpu().detach().numpy()):
          ax = fig.add_subplot(3, 10, i+21, xticks=[], yticks=[])
          ax.imshow(im, 'gray')
        break

if __name__ == "__main__":
    main()