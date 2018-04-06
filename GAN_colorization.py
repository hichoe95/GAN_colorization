
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from __future__ import print_function
import argparse
import csv
import os.path
import torch.nn.parallel
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import Image
import PIL

gpu = 2
batch_size = 32
imsz = 64


# In[2]:


#tempdir = '../data/parkIMG/train'
traindir = os.path.join('../data/data_set/catdog', 'training')
testdir = os.path.join('../data/data_set/catdog', 'testing')


class TrainImageFolder(datasets.ImageFolder):

    def __getitem__(self, index):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        image = super(TrainImageFolder, self).__getitem__(index)[0].numpy().transpose((1,2,0))
        image = std * image + mean
        input_gray = image
        input_gray = np.dot(input_gray[...,:3], [0.299, 0.587, 0.114])
        
        return torch.FloatTensor(input_gray.transpose((0,1))), torch.FloatTensor(image.transpose((2,0,1)))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_loader = data.DataLoader(
        TrainImageFolder(traindir,
                         transforms.Compose([
                             transforms.RandomResizedCrop(imsz),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             normalize,
                         ])),
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True)

test_loader = data.DataLoader(
        TrainImageFolder(testdir,
                        transforms.Compose([
                            transforms.RandomResizedCrop(imsz),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize,
                        ])),
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=False)


# In[3]:


def imshow(inp):
    inp = inp.numpy().transpose((1,2,0))
    print(inp.shape)
    inp = np.clip(inp, 0, 1) 
    plt.imshow(inp)

def gray_imshow(inp):
    inp = inp.numpy()#.transpose((1,2,0))
    print(inp.shape)
    plt.imshow(inp,cmap = plt.get_cmap('gray'))


# In[4]:


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        """noise image + gray image"""
        # batch_size * 2 * 64 * 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )
        
        # batch_size * 64 * 64 * 64
        self.maxpool = nn.MaxPool2d(2,2)
        
        # batch_size * 64 * 32 * 32
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64 * 2 , 3, 1, 1),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.1)
        )
        
        # batch_size * (64 * 2) * 32 * 32
        # maxpool(2,2)
        
        # batch_size * (64 * 2) * 16 * 16
        self.conv3 = nn.Sequential(
            nn.Conv2d(64*2, 64 * 4, 3, 1, 1),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.1)
        )
        
        # batch_size * (64 * 4) * 16 * 16
        # maxpool(2,2)
        
        # batch_size * (64 * 4) * 8 * 8
        self.conv4 = nn.Sequential(
            nn.Conv2d(64 * 4, 64 * 8, 8, 1, 0),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.1)
        )
        
        # batch_size * (64 * 8) * 1 * 1
        self.fc = nn.ConvTranspose2d(64 * 8, 64 * 16, 4, 1, 0)
        
        # batch_size * (64 * 16) * 4 * 4
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(64 * 16, 64 * 8, 4, 2, 1),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.1)
        )
        
        # batch_size * (64 * 8) * 8 * 8
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.1)
        )
        
        # batch_size * (64 * 4) * 16 * 16
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1),
            nn.BatchNorm2d(64 *2),
            nn.LeakyReLU(0.1)
        )
        
        # batch_size * (64 * 2) * 32 * 32        
        self.upsample4 = nn.Sequential(
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        
        # batch_size * 64 * 64 * 64
        self.conv1by1 = nn.Sequential(
            nn.Conv2d(64,64,1,1,0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )
        
        
        # batch_size * 64 * 64 * 64
        self.conv = nn.Sequential(
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )
        # batch_size * 3 * 64 * 64

    def forward(self, input):
        output1 = self.conv1(input)
        pool1 = self.maxpool(output1)
        output2 = self.conv2(pool1)
        pool2 = self.maxpool(output2)
        output3 = self.conv3(pool2)
        pool3 = self.maxpool(output3)
        output4 = self.conv4(pool3)
        output5 = self.fc(output4)
        output6 = self.upsample1(output5)
        output7 = self.upsample2(output6) + output3
        output8 = self.upsample3(output7) + output2
        output9 = self.upsample4(output8) + output1
        output10 = self.conv1by1(output9)
        out = self.conv(output10)
        
        return out


# In[5]:


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            #""" color imgage (fake or real image)"""
            # batch_size * 3 * 64 * 64
            nn.Conv2d(3,64,kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            
            # batch_size * 64 * 32 * 32
            nn.Conv2d(64,128,kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            
            # batch _size * 64 * 32 * 32
            nn.Conv2d(128, 128, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            
            # batch_size * 128 * 16 * 16
            nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            
            # batch_size * 256 * 8 * 8
            nn.Conv2d(256, 512, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
            
            # batch_size * 512 * 4 * 4
            nn.Conv2d(512, 1024, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace = True),
            
            # batch_size * 1024 * 2 * 2
            )
        
        
        # batch_size * 1024 * 2 * 2
        self.fc = nn.Sequential(
            nn.Linear(1024 * 2 * 2 , 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, batch_size),
            nn.Sigmoid()
        )
        
    def forward(self, input, b_size):
        output = self.main(input)
        output = self.fc(output.view(b_size,-1))
        return output
        


# In[6]:


def to_variable(x):
    if torch.cuda.is_available:
        x = x.cuda(gpu)
    return Variable(x)


# In[7]:


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:         # Conv weight init
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:  # BatchNorm weight init
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# In[8]:


Discri = Discriminator().cuda(gpu) if torch.cuda.is_available() else Discriminator()
Discri.apply(weights_init)
Gener = Generator().cuda(gpu) if torch.cuda.is_available() else Generator()
Gener.apply(weights_init)

real_label = 1
fake_label = 0

Loss= nn.BCELoss().cuda(gpu) if torch.cuda.is_available() else nn.BCELOSS()
optimizerD = torch.optim.Adam(Discri.parameters(), lr = 0.0002,betas = (0.5, 0.999))
optimizerG = torch.optim.Adam(Gener.parameters(), lr = 0.0002, betas = (0.5, 0.999))


# In[9]:


for epoch in range(50):
    loss_D = 0.0
    for i, data in enumerate(train_loader):
        gray, color = data
        #print(len(data[0]))
        b_size = len(data[0])

        
        # gray >> grays (batch_size * 1 * 64 * 64)
        grays = torch.from_numpy(np.resize(gray.numpy(), (b_size, 1, 64, 64)))
        
        ######## Train Discriminator ########
        
        color = to_variable(color)
        # Make noise
        noise = torch.randn(b_size, 1, 64, 64).uniform_(0,1)
 
        # gray image + noise image
        gray_noise = to_variable(torch.cat([grays,noise],dim=1))
        #print(gray_noise.shape)
        
        ####### Train d to recognize color image as real
        
        output = Discri(color,b_size)
        real_loss = torch.mean((output-1)**2)
        
        ###### Train d to recognize fake image as fake
        
        fake_img = Gener(gray_noise)
        #print("fake_img : ", fake_img.shape)
        output = Discri(fake_img,b_size)
        fake_loss = torch.mean(output**2)
        
        ###### Backpro & Optim D
        d_loss = real_loss + fake_loss
        Discri.zero_grad()
        Gener.zero_grad()
        d_loss.backward()
        optimizerD.step()
        
        
        ######## Train Generator ########
        noise = torch.randn(b_size, 1, 64, 64).uniform_(0,1)
        gray_noise2 = to_variable(torch.cat([grays,noise],dim=1))
        fake_img = Gener(gray_noise2)
        output = Discri(fake_img,b_size)
        g_loss = torch.mean((output-1)**2)
        
        ###### Backpro & Optim G
        Discri.zero_grad()
        Gener.zero_grad()
        g_loss.backward()
        optimizerG.step()
        
        
        #print(fake.shape)
        fake_img = torchvision.utils.make_grid(fake_img.data)

        if i%100 == 99:
            print('[%d, %5d] real loss: %.4f, fake_loss : %.4f, g_loss : %.4f' % (epoch + 1, i+1, real_loss.data[0],fake_loss.data[0], g_loss.data[0]))
            imshow(fake_img.cpu())
            plt.show()


# In[18]:


Discri.eval()
Gener.eval()

fixed_noise = torch.randn(batch_size, 1, 64, 64).uniform_(0,1)

for i, data in enumerate(test_loader,0) :
    images, label = data
    
    if len(data[0]) != batch_size:
        continue
        
    grays = torch.from_numpy(np.resize(images.numpy(), (batch_size, 1, 64, 64)))
    
    gray = to_variable(torch.cat([grays,fixed_noise],dim = 1))
    
    output = Gener(gray)
    inputs = torchvision.utils.make_grid(grays)
    labels = torchvision.utils.make_grid(label)
    out = torchvision.utils.make_grid(output.data)
    print('======inputs======')
    imshow(inputs.cpu())
    plt.show()
    print('======origin======')
    imshow(labels.cpu())
    plt.show()
    print('======output======')
    imshow(out.cpu())
    plt.show()


# In[ ]:




