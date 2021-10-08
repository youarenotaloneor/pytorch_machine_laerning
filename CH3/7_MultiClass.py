#coding=utf-8

import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn,optim
 
# 生成数据
cluster = torch.ones(500, 2) 
data0 = torch.normal(4*cluster, 2)      
data1 = torch.normal(-4*cluster, 1)    
data2 = torch.normal(-8*cluster, 1)     
label0 = torch.zeros(500)
label1 = torch.ones(500)                
label2 = label1*2  #500个标签2
 
x = torch.cat((data0, data1, data2), ).type(torch.FloatTensor)  
y = torch.cat((label0, label1, label2), ).type(torch.LongTensor)    
 
plt.scatter(x.numpy()[:, 0], x.numpy()[:, 1], c=y.numpy(), s=10, lw=0, cmap='RdYlGn')
plt.show()

class Net(nn.Module):    
    def __init__(self, input_feature, num_hidden,outputs):
        super(Net, self).__init__()     
        self.hidden = nn.Linear(input_feature, num_hidden)   # 线性隐含层
        self.out = nn.Linear(num_hidden, outputs)       # 输出层

    def forward(self, x):
        x = F.relu(self.hidden(x))      # 激励函数ReLU处理隐含层的输出
        x = self.out(x)
        x = F.softmax(x)     #使用softmax将输出层的数据转换成概率值           
        return x

CUDA = torch.cuda.is_available()

if CUDA:
    net = Net(input_feature=2, num_hidden=20,outputs=3).cuda()
    inputs = x.cuda()
    target = y.cuda()
else:
    net = Net(input_feature=2, num_hidden=20,outputs=3)
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
    accuracy = sum(pred_y == target_y)/1500.0  
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


train(net,criterion,optimizer,10000)

