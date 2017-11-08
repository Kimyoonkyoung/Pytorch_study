# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transform
from torch.utils.data import DataLoader
from torch.autograd import Variable

# hyper parameter
batch_size = 1
learning_rate = 0.0002
epoch = 100

# data loader
img_dir = "../data/sampleimage"
img_data = dset.ImageFolder(img_dir, transform.Compose([
            transform.Scale(256),
            transform.RandomCrop(224),
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            ]))

img_batch = data.DataLoader(img_data, batch_size=batch_size, shuffle=True, num_workers=2)


# basic block 정의
# basic block 이라고 함은.. layer 하나하나를 의미하는것이 아니라 CNN 의 기본구조 (conv-pool) 요런거 몇개 합치는걸 의미함
def conv_2_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model

def conv_3_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model


# VGG 모델 정의
class VGG(nn.Module):
    def __init__(self, base_dim, num_classes=2):
        super(VGG, self).__init__()

        # 얘 이름이 feature 인 이유는 feature detect 한다~ 고 의미해서 feature 라고 지은듯
        self.feature = nn.Sequential(
            conv_2_block(3, base_dim), # 최초의 3은 이미지 채널수 , 이후의 dimension 은 논문에서 제시한거
            conv_2_block(base_dim, 2*base_dim),
            conv_3_block(2*base_dim, 4*base_dim),
            conv_3_block(4*base_dim, 8*base_dim),
            conv_3_block(8*base_dim, 8*base_dim),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(8*base_dim*7*7, 100), # 7*7 은 마지막 conv layer 지나온 이미지의 크기가 7*7 이기 때문임
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100,20),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(20, num_classes),
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1) #TODO 뭐하는 함수인고?
        x = self.fc_layer(x)

        return x


# 모델 정의 끝!!
if torch.cuda.is_available():
    model = VGG(base_dim=64).cuda()
else:
    model = VGG(base_dim=64)

for i in model.named_children():
    print i


# optimizer & loss 정의
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 드디어 training
for i in range(epoch):
    for img, label in img_batch:

        if torch.cuda.is_available():
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)

        optimizer.zero_grad()
        output = model(img)
        loss = loss_func(output, label)
        loss.backward()
        optimizer.step()

    if i % 10 == 0:
        print loss










































