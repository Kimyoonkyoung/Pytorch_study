#-*-coding: utf-8-*-
import torch

# 1. torch.rand 는 [0,1) 사이의 랜덤 수 내줌
x = torch.rand(2,3)
print '1.1] ', x

x = torch.randperm(5)
print '1.2] ', x

# 2. zeros, ones, arange
x = torch.zeros(2,3)
print '2.1] ', x

x = torch.ones(2,3)
print '2.2] ', x

x = torch.arange(0, 5, 1.5)
print '2.3] ', x

# 3. data type
x = torch.FloatTensor(2,3) # size 지정해서 만들어준것
print '3.1] ', x

x = torch.FloatTensor([3, 5, 7.3]) # 3, 5, 7.3 라는 값을 넣어준것
print '3.2] ', x

x = x.type_as(torch.IntTensor()) # 형 변환
print '3.3] ', x


import numpy as np

# 4.1 torch.from_numpy(ndarray) -> tensor
x1 = np.ndarray(shape = (2,3), dtype=int, buffer=np.array([1,2,3,4,5,6])) # np.ndarray
x2 = torch.from_numpy(x1) # tensor
print '4.1] ', x1, x2

# 4.2 tensor.numpy() -> ndarray
x3 = x2.numpy()
print '4.2] ', x3

# 5. Tensor on CPU, GPU
x = torch.FloatTensor([[1,2,3],[4,5,6]])
#x_gpu = x.cuda() # cuda 용 빌드 해야 가능 output : [torch.cuda.FloatTensor of size 2x3 (GPU 0)]
x_cpu = x.cpu()
print '5.1] ', x_cpu


# 6. tensor size
x = torch.FloatTensor(10, 12, 3, 4)
print '6.1] ', x.size()[:] # tensor.size() -> indexing also possible


# 7. indexing
x = torch.rand(4,3)
out = torch.index_select(x, 0, torch.LongTensor([0,3])) # torch.index_select(input, dim, index) # 두번째 파라미터가 뭔지모르겠다
print '7.1] ', x, out
print '7.2] ', x[:,0], x[0,:], x[0:2, 0:2] # pythonic indexing also works

x = torch.randn(2, 3)
mask = torch.ByteTensor([[0,0,1], [0,1,0]])
out = torch.masked_select(x, mask) # torch.masked_select(input, mask)
print '7.3] ', x, mask, out


# 8. Joining
x = torch.FloatTensor([[1,2,3],[4,5,6]])
y = torch.FloatTensor([[-1, -2, -3], [-4, -5, -6]])
z1 = torch.cat([x,y], dim=0) # torch.cat(seq, dim=0) -> concatenate tensor along dim # concatenate
z2 = torch.cat([x,y], dim=1)
print '8.1] ', x, y, z1, z2

x = torch.FloatTensor([[1,2,3],[4,5,6]])
x_stack = torch.stack([x,x,x], dim=0) # torch.stack(sequence,dim=0) -> stack along new dim # 똑같은 배열을 여러개 쌓는다
print '8.2] ', x_stack


# 9. Slicing
x_1, x_2 = torch.chunk(z1, 2, dim=0) # torch.chunk(tensor, chunks, dim=0) -> tensor into num chunks # 덩어리로 나눠서 주는것 (딱 떨어지지않으면 에러)
y_1, y_2, y_3 = torch.chunk(z1, 3, dim=1)
print '9.1] ', z1, x_1, x_2, y_1, y_2, y_3


# 10. Split
x1,x2 = torch.split(z1,2,dim=0) # torch.split(tensor,split_size,dim=0) -> split into specific size
y1 = torch.split(z1,2,dim=1)
print '10.1] ', z1, x1, x2, y1


# 11. Squeeze
x1 = torch.FloatTensor(10,1,3,1,4)
x2 = torch.squeeze(x1) # torch.squeeze(input,dim=None) -> reduce dim by 1
print '11.1] ', x1.size(), x2.size()

x1 = torch.FloatTensor(10,3,4)
x2 = torch.unsqueeze(x1,dim=0)
print '11.2] ', x1.size(),x2.size() # torch.unsqueeze(input,dim=None) -> add dim by 1


import torch.nn.init as init

# 12. Initialization
x1 = init.uniform(torch.FloatTensor(3,4),a=0,b=15) # a ~ b 까지의 랜덤수로 init
x2 = init.normal(torch.FloatTensor(3,4),std=0.2)
x3 = init.constant(torch.FloatTensor(3,4),3.1415)
print '12.1 ]', x1, x2, x3

