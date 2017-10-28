#-*-coding: utf-8-*-

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# Hyper parameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset load
train_dataset = dsets.MNIST(root='../data/',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='../data/',
                           train=False,
                           transform=transforms.ToTensor())


# data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# nn models (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

net = Net(input_size, hidden_size, num_classes) # model initialize

# load and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# train model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(test_loader):
        # convert torch tensor to variable
        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)

        # forward + bacward + optimize
        optimizer.zero_grad() # zero the gradient buffer
        outputs = net(images) # net.forward(images) 이렇게 안써도 되는듯
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                   % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))




# test model
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1,28*28))
    outputs = net(images)

    # 두번째파라미터는 dimension (리턴 label 의 갯수 인듯)
    # output : 1번째는 max 값 , 2번째는 argmax
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

torch.save(net.state_dict(), 'mnist_model.pkl')



