
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
batch_size = 4


# In[2]:


tempdir = '../data/parkIMG/train'

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
        TrainImageFolder(tempdir,
                         transforms.Compose([
                             transforms.RandomResizedCrop(256),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             normalize,
                         ])),
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True)

test_loader = data.DataLoader(
        TrainImageFolder(tempdir,
                        transforms.Compose([
                            transforms.RandomResizedCrop(256),
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
        self.encode = nn.Sequential(
            # 노이즈와 합쳐진 흑백 input
            # batch_size * (1+1) * 256 * 256
            nn.Conv2d(2, 16, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace = False),
            
            # batch_size * 16 * 256 * 256
            nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = False),
            nn.MaxPool2d(2),
            
            # batch_size * 32 * 128 * 128
            nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = False),
            nn.MaxPool2d(2),
            
            # batch_size * 64 * 64 * 64
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = False),
            
            # batch_size * 128 * 64 * 64
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = False),
            
            # batch_size * 256 * 64 * 64
            nn.UpsamplingNearest2d(scale_factor=2),
            
            # batch_size * 256 * 128 * 128
            nn.Conv2d(256, 128, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = False),
            
            # batch_size * 128 * 128 * 128
            nn.UpsamplingNearest2d(scale_factor = 2),
            
            # batch_size * 128 * 256 * 256
            nn.Conv2d(128, 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = False)
            # batch_size * 64 * 256 * 256
            )
        
        self.decode = nn.Sequential(
            #흑백이미지와 노이즈를 다시 합쳐줌
            # batch_size * (64 + 2) * 256 * 256
            nn.Conv2d(66, 32, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            
            # batch_size * 32 * 256 * 256
            nn.Conv2d(32, 3, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.Sigmoid()
            # batch_size * 3 * 256 * 256
            )

    def forward(self, input):
        encoding = self.encode(input)
        decoding = self.decode(torch.cat([input,encoding], dim = 1))
        return decoding


# In[5]:


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            #흑백이미지와 컬러이미지를 합쳐서 진행
            # 4 * 256 * 256
            nn.Conv2d(4,8,kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            
            # 8 * 128 * 128
            nn.Conv2d(8,16,kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace = True),
            
            # 16 * 64 * 64
            nn.Conv2d(16, 32, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace = True),
            
            # 32 * 32 * 32
            nn.Conv2d(32, 64, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace = True),
            
            # 64 * 16 * 16
            nn.Conv2d(64,32,kernel_size = 4, stride = 2, padding = 1 , bias = False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace = True),
            
            # 32 * 8 * 8
            nn.Conv2d(32,16,kernel_size = 4, stride = 2, padding = 1 , bias = False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace = True),
            
            # 16 * 4 * 4
            nn.Conv2d(16, 1, kernel_size = 4, stride = 1, padding = 0 , bias = False),
            nn.Sigmoid()
            )
        
    def forward(self, input):
        output = self.main(input)
        """sigmoid로 가짜인지 진짜인지의 확률값으로 반환."""
        return output.view(-1,1).squeeze(1)
        


# In[6]:


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:         # Conv weight init
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:  # BatchNorm weight init
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# In[7]:


D = Discriminator().cuda(gpu)
D.apply(weights_init)
G = Generator().cuda(gpu)
G.apply(weights_init)
noise = torch.FloatTensor(batch_size, 1, 256, 256)
label = torch.FloatTensor(batch_size)

real_label = 1
fake_label = 0

Loss= nn.BCELoss().cuda(gpu)
optimizerD = torch.optim.Adam(D.parameters(), lr = 0.0001,betas = (0.5, 0.999))
optimizerG = torch.optim.Adam(G.parameters(), lr = 0.0001, betas = (0.5, 0.999))


# In[ ]:


for epoch in range(50):
    loss_D = 0.0
    for i, data in enumerate(train_loader):
        gray, color = data
        gray = np.resize(gray.numpy(),(batch_size,1,256,256))
        #gray : batch_size * 1 * 256 * 256, color : batch_size * 3 * 256 * 256
        
        #discriminator
        
        #real
        D.zero_grad()
        
        #진짜 이미지를 학습해야 하므로 real_label(=1) 로 학습하도록한다.
        label.fill_(real_label)
        
        #inputs을 color 이미지와 gray 이미지를 합쳐서 생성.
        inputs = Variable(torch.from_numpy(np.concatenate((color.numpy(), gray),axis = 1))).cuda(gpu)
        #inputs = Variable(inputs).cuda(gpu)
        labelv = Variable(label).cuda(gpu)
        
        output = D(inputs)
        loss_D_real = Loss(output, labelv)
        loss_D_real.backward()
        D_x = output.data.mean()
        
        #real end
        
        #fake
        
        # batch_size * 1 * 256 * 256
        noise.uniform_(0,1)
        gray_noise = np.concatenate((gray, noise.numpy()), axis = 1)
        gray_noise = Variable(torch.from_numpy(gray_noise)).cuda(gpu)
        
        fake = G(gray_noise)
        fake_img = fake
        fake = torch.from_numpy(np.concatenate((fake.data.cpu().numpy(), gray),axis = 1))
        fake = Variable(fake).cuda(gpu)
        
        labelv = Variable(label.fill_(fake_label)).cuda(gpu)
        output = D(fake)
        loss_D_fake = Loss(output, labelv)
        loss_D_fake.backward()
        
        D_G_1 = output.data.mean()
        
        loss_D += loss_D_real + loss_D_fake
        
        optimizerD.step()
        
        # generator
        
        G.zero_grad()
        
        labelv = Variable(label.fill_(real_label)).cuda(gpu)
        output = D(fake)
        
        loss_G = Loss(output, labelv)
        loss_G.backward()
        
        D_G_2 = output.data.mean()
        optimizerG.step()
        
        
        #print(fake.shape)
        fake_img = torchvision.utils.make_grid(fake_img.data)
#         out = torchvision.utils.make_grid(outputs.data)
#         inp = torchvision.utils.make_grid(input.data)
#         label = torchvision.utils.make_grid(label.data)
        if i%10 == 9:
            print('[%d, %5d]loss' % (epoch + 1, i+1))
            print(D_x)
            print(D_G_1)
            running_loss = 0.0
        if i%10 == 9:
            #print(outputs.data[0].cpu().numpy().shape)
            imshow(fake_img.cpu())
            plt.show()
#             imshow(inp.cpu())
#             plt.show()
#             imshow(out.cpu())
#             plt.show()
#             imshow(label.cpu())
#             plt.show()


# In[ ]:


model.eval()

for i, data in enumerate(test_loader,0) :
    images, label = data
    images = torch.FloatTensor(np.resize(images.numpy(),(4,1,256,256)))

    images, label = Variable(images.cuda(gpu)), Variable(label.cuda(gpu))
                
    output = model(images)
        
    out = torchvision.utils.make_grid(output.data)
    images = torchvision.utils.make_grid(images.data)
    imshow(images.cpu())
    plt.show()
    imshow(out.cpu())
    plt.show()


# In[ ]:




