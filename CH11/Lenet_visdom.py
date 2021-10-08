#coding=utf-8
import os
import torch
from torch import nn,optim
import torch.nn.functional as F

from torchvision import datasets, transforms

transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])

trainset = datasets.MNIST('data', train=True, download=True, transform=transform)
testset = datasets.MNIST('data', train=False, download=True, transform=transform)


#初始化visdom
import visdom
vis = visdom.Visdom()

class LeNet(nn.Module):
    # 定义Net的初始化函数，本函数定义了神经网络的基本结构
    def __init__(self):
        # 继承父类的初始化方法，即先运行nn.Module的初始化函数
        super(LeNet,self).__init__()
        # C1卷积层：输入1张灰度图片，输出6张特征图，卷积核5x5
        self.c1 = nn.Conv2d(1,6,(5,5))
        # C3卷积层：输入6张特征图，输出16张特征图，卷积核5x5
        self.c3 = nn.Conv2d(6,16,5)
        # 全连接层S4->C5：从S4到C5是全连接，S4层中16*4*4个节点全连接到C5层的120个节点上
        self.fc1 = nn.Linear(16*4*4,120)
        # 全连接层C5->F6：C5层的120个节点全连接到F6的84个节点上
        self.fc2 = nn.Linear(120,84)
        # 全连接层F6->OUTPUT：F6层的84个节点全连接到OUTPUT层的10个节点上，10个节点的输出代表着0到9的不同分值。
        self.fc3 = nn.Linear(84,10)

    # 定义向前传播函数
    def forward(self,x):
        # 输入的灰度图片x经过c1的卷积之后得到6张特征图，然后使用relu函数，增强网络的非线性拟合能力，接着使用2x2窗口的最大池化，然后更新到x
        x = F.max_pool2d(F.relu(self.c1(x)),2)
        # 输入x经过c3的卷积之后由原来的6张特征图变成16张特征图，经过relu函数，并使用最大池化后将结果更新到x
        x = F.max_pool2d(F.relu(self.c3(x)),2)
        # 使用view函数将张量x（S4）变形成一维向量形式，总特征数不变，为全连接层做准备
        x = x.view(-1,self.num_flat_features(x))
        # 输入S4经过全连接层fc1，再经过relu，更新到x
        x = F.relu(self.fc1(x))
        # 输入C5经过全连接层fc2，再经过relu，更新到x
        x = F.relu(self.fc2(x))
        # 输入F6经过全连接层fc3，更新到x
        x = self.fc3(x)
        return x

    # 计算张量x的总特征量
    def num_flat_features(self,x):
        # 由于默认批量输入，第零维度的batch剔除
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

CUDA = torch.cuda.is_available()
if CUDA:
    lenet = LeNet().cuda()
else:
    lenet = LeNet()

criterion=nn.CrossEntropyLoss()
optimizer = optim.SGD(lenet.parameters(),lr=0.001,momentum=0.9)

trainloader = torch.utils.data.DataLoader(trainset,batch_size=4, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset,batch_size=4, shuffle=False, num_workers=2)

def train(model,criterion,optimizer,epochs=1):
    counter=1
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader,0):

            inputs,labels = data
            if CUDA:
                inputs,labels = inputs.cuda(),labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item() 
            
            if i%1000==999:
                print('[Epoch:%d, Batch:%5d] Loss: %.3f' % (epoch+1, i+1, running_loss / 1000))

                #每1000步更新一次折线图
                vis.line(X=torch.FloatTensor([counter*1000]), Y=torch.FloatTensor([running_loss / 1000]), win='loss', update='append')
                counter+=1
                running_loss = 0.0

    print('Finished Training')

def test(testloader,model):
    correct = 0
    total = 0
    for data in testloader:

        images, labels = data
        if CUDA:
            images = images.cuda()
            labels = labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1) 
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Accuracy on the test set: %d %%' % (100 * correct / total))

def load_param(model,path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))

def save_param(model,path):
    torch.save(model.state_dict(),path)

load_param(lenet,'model.pkl')

train(lenet,criterion,optimizer,epochs=2)
save_param(lenet,'model.pkl')

test(testloader,lenet)

