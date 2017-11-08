#-*-coding: utf-8-*-

# data loading tutorial
#https://github.com/pytorch/tutorials/blob/master/beginner_source/data_loading_tutorial.py

import torch
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms


'''
# image transform 할꺼면 여기서 쭈욱
# options
    .Scale (resize) / .RandomSizeCrop / .RandomHorizontalFlip  / .Normalize
    / .ToTensor (numpy image to torch image)
'''
transform = transforms.Compose(
[transforms.ToTensor(),
 transforms.Scale((32,32)),
 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

image_dir = '/image/dir'


trainset = dset.ImageFolder(root=image_dir, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = dset.ImageFolder(root='tests',transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=True, num_workers=2)

classes=('shirt','pants','sock')

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
#imshow(torchvision.utils.make_grid(images))

# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))