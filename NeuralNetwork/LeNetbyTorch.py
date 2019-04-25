#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Treamy


import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils import data as utdata

import torchvision as tv
from torchvision import transforms as tsfm
from torchvision import datasets as tvds

import os

max_epoch = 10
epoch = 0
batchsz = 64
LR = 0.001


if not os.path.exists('./model/'):
    os.mkdir('model')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)


# load dataset, if you first use this script, you need unannotate the code below 'download=True' part.
tr_ds = tvds.MNIST(root='./data', train=True, transform=tsfm.ToTensor(),)# download=True)
ts_ds = tvds.MNIST(root='./data', train=False, transform=tsfm.ToTensor(),)# download=True)

class Flatten(nn.Module):
  def __init__(self):
    super(Flatten, self).__init__()

  def forward(self, x):
    return x.view(x.size(0), -1)


class LeNet(nn.Module):
  def __init__(self, ):
    super(LeNet, self).__init__()
    self.conv1 = nn.Sequential(  # 1,28,28
          nn.Conv2d(1, 6, 5, 1, 2),   # 6, 28, 28
          nn.ReLU(),
          nn.MaxPool2d(2, stride=2)  # 6, 14, 14
    )
    self.conv2 = nn.Sequential(
        nn.Conv2d(6, 16, 5), # 6, 10, 10
        nn.ReLU(),
        nn.MaxPool2d(2) # 6, 5, 5
    )
    self.flatten = Flatten()
    self.fc1 = nn.Sequential(
        nn.Linear(16*5*5, 120),
        nn.ReLU()
    )
    self.fc2 = nn.Sequential(
        nn.Linear(120, 84),
        nn.ReLU()
    )
    self.fc3 = nn.Linear(84, 10)
    
    
  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.flatten(x)
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)
    return x
  

tr_dl = utdata.DataLoader(tr_ds, batch_size=batchsz, shuffle=True)
ts_dl = utdata.DataLoader(ts_ds, batch_size=batchsz, shuffle=False)


# initial the model and optimizer
model = LeNet()
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)    
print(model)

# have a look on a batch
# bx, by = next(iter(tr_dl))
# bx = bx.to(device)
# print(bx.shape)    


# training
while epoch < max_epoch:
  model.train()
  s_loss = 0.0
  for i, batch in enumerate(tr_dl, 1):
    bx, by = batch
    bx, by = bx.to(device), by.to(device)
    
    optimizer.zero_grad()
    logits = model.forward(bx)
    loss = F.cross_entropy(logits, by)
    loss.backward()
    optimizer.step()
    
    s_loss += loss.item()
    if i%100==0:
      print('epoch: {} on train, batch: {}, loss:{}'.format(epoch, i, loss.item()))
      
  aver_loss = s_loss / i
  print('epoch: {} on train, average loss:{}'.format(epoch, aver_loss))
  
  
  model.eval()
  right = 0
  total = 0
  for i, batch in enumerate(ts_dl, 1):
    bx, by = batch
    bx, by = bx.to(device), by.to(device)
    logits = model.forward(bx)
    by_hat = torch.argmax(logits, 1)
    right += (by_hat==by).sum()
    total += bx.size(0)

  acc = 100 * right / total
  print('epoch:%d on test, accuracy is %.3f%%' % (epoch, acc))
  torch.save(model.state_dict(), './model/model%03d.pth' % epoch)

  epoch += 1
