#-*-coding: utf-8-*-

import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable


# 1. Basic autograd example
# create tensor
# requires_grad 는 gradient 에 따라서 update 를 하느냐 마느냐의 파라미터
# False 로 설정하면 고정, True 는 가변
x = Variable(torch.Tensor([1]), requires_grad=True)
w = Variable(torch.Tensor([2]), requires_grad=True)
b = Variable(torch.Tensor([3]), requires_grad=True)

# build a computational graph
y = x * w + b # linear regression

# compute gradient
y.backward()

print '1] ', x.grad
print '1] ', w.grad
print '1] ', b.grad


#===================== 2. 기본 함수 터득하기 =====================#
# create tensor
x = Variable(torch.randn(5,3))
y = Variable(torch.randn(5,2))

# build a linear layer
linear = nn.Linear(3,2) # param : infeature, outfeature, bias
print ('w : ', linear.weight)
print ('b : ', linear.bias)

# build a loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# forward propagation
pred = linear(x)

# compute loss
loss = criterion(pred, y)
print ('loss : ', loss.data[0])

# back propagation
loss.backward()

# gradient
print ('dL/dW : ', linear.weight.grad)
print ('dL/db : ', linear.bias.grad)

# 1-step optimization (gradient descent)
optimizer.step()

pred = linear(x)
loss = criterion(pred, y)
print ('loss after 1 step optimizer: ', loss.data[0])


# 3. loading data from numpy
a = np.array([[1,2], [3,4]])
b = torch.from_numpy(a) # convert numpy to torch
print '3] torch type : ', type(b)
c = b.numpy() # convert torch to numpy
print '3] numpy type : ', type(c)


#===================== 4. public dataset 불러오기 =====================#
# download and construct dataset
train_dataset = dsets.CIFAR10(
    root='../data/', train=True, transform=transforms.ToTensor(), download=False)

# select one data pair (read data from disk)
image, label = train_dataset[0]
print ('CIFAR10 first image size & type : ', image.size(), type(image))
print ('CIFAR10 first label & type : ', label, type(label))

# data loader (queue 에 저장되며, 쓰레드를 사용한다)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=100, shuffle=True, num_workers=2)

# iteration 하면 disk 에서 data 를 load 한다
data_iter = iter(train_loader)

# mini-batch image and labels
images, labels = data_iter.next()
print ('type : ', type(images), type(labels))

# batch 만큼 불러오면서 수행하는 코드
for images, labels in train_loader:
    # 여기에서 trainging 코드 작성하면 됨
    pass


#===================== 5. custom dataset 불러오기 =====================#
# 이미지는 torch.FloatTensor 의 tensor (이미지 받아서 float tensor에 넣으면됨)
# 라벨은 type 'int' 의 torch.LognTensor (int numpy 만들고 to tensor 로 형변환하면됨)
class CustomDataset(data.Dataset):
    def __init__(self):
        # TODO
        # 1. initialize file path or list of file names
        pass

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (e.g. torchvision.Transform)
        # 3. Return a data pair (e.g. image and label) # return image, label 요렇게 하면될듯?
        pass

    def __len__(self):
        # You should change 0 to the total size of your dataset. # 무슨말인지 모르겟다..그냥 데이터 사이즈 리턴?
        return 0


# 위 CustomDataset 만 채워 넣어놓으면, 아래 구현으로 배치대로 들고 올 수 있다 오 짱좋아
custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(
    dataset=custom_dataset, batch_size=100, shuffle=True, num_workers=2)


#========================== 6. Using pretrained model ==========================#
# download and load pretrained resnet
resnet = torchvision.models.resnet18(pretrained=True)

# fine tune 안하고 고정시키기
for param in resnet.parameters():
    param.requires_grad = False

# replace top layer (class num)
resnet.fc = nn.Linear(resnet.fc.in_features, 100) # resnet 의 fc 를 100으로 변경

# for test
image = Variable(torch.randn(10, 3, 256, 256))
outputs = resnet(image)
print (outputs.size())


#============================ 7. Save and load the model ============================#
# save and load entire model
torch.save(resnet, 'resnet_save_entire.pkl')
model = torch.load('resnet_save_entire.pkl')

# save and load only model parameters (이것이 추천하는 방법이라고 함)
torch.save(resnet.state_dict(), 'resnet_save_param.pkl')
resnet.load_state_dict(torch.load('resnet_save_param.pkl'))