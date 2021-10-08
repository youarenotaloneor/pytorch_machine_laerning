#coding=utf-8

import torch
import matplotlib.pyplot as plt
from torch import nn,optim   
 

cluster = torch.ones(500, 2) #ones函数生成500x2的数据
data0 = torch.normal(4*cluster, 2) #构造一个均值为4，标准差为2的数据簇
data1 = torch.normal(-4*cluster, 2) #构造一个均值为4，标准差为2的数据簇
label0 = torch.zeros(500) #500个标签0
label1 = torch.ones(500) #500个标签1

x = torch.cat((data0, data1), ).type(torch.FloatTensor) 
y = torch.cat((label0, label1), ).type(torch.LongTensor)

plt.scatter(x.numpy()[:,0], x.numpy()[:, 1], c=y.numpy(), s=10, lw=0, cmap='RdYlGn')
plt.show()

class Net(nn.Module):     # 继承 torch 的 Module
    def __init__(self):
        super(Net, self).__init__()     # 继承 __init__ 功能
        self.linear = nn.Linear(2,2)

    def forward(self, x):
        x = self.linear(x)
        print("x为：",x)
        x = torch.sigmoid(x)
        return x


CUDA = torch.cuda.is_available()

if CUDA:
    net = Net().cuda()
    inputs = x.cuda()
    target = y.cuda()
else:
    net = Net()
    inputs = x
    target = y

optimizer = optim.SGD(net.parameters(), lr=0.02)
criterion = nn.CrossEntropyLoss()

def draw(output):
    if CUDA:
        output=output.cpu()
    plt.cla()
    output = torch.max((output), 1)[1] 
    pred_y = output.data.numpy().squeeze()
    target_y = y.numpy()
    plt.scatter(x.numpy()[:, 0], x.numpy()[:, 1], c=pred_y, s=10, lw=0, cmap='RdYlGn')
    accuracy = sum(pred_y == target_y)/1000.0  
    plt.text(1.5, -4, 'Accuracy=%s' % (accuracy), fontdict={'size': 20, 'color':  'red'})
    plt.pause(0.1)


def train(model,criterion,optimizer,epochs):
    for epoch in range(epochs):
        #forward
        output = model(inputs)
        loss = criterion(output,target)

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 40 == 0:
            draw(output)


train(net,criterion,optimizer,1000)


        


