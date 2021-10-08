#coding=utf-8

import torch
import matplotlib.pyplot as plt
from torch import nn,optim
import torch.nn.functional as F

x = torch.unsqueeze(torch.linspace(-3,3,10000),dim=1)
y = x.pow(3)+1.3*torch.rand(x.size())

#plt.scatter(x.numpy(), y.numpy(),s=0.01)
#plt.show()

class Net(nn.Module):  # 继承 torch.nn 的 Module
    def __init__(self, input_feature, num_hidden, outputs):
        super(Net, self).__init__()     # 继承 __init__ 
        # 定义每层神经元的结构与数目
        self.hidden = nn.Linear(input_feature, num_hidden)   # 线性隐含层
        self.out = nn.Linear(num_hidden, outputs)   # 输出层
 
    def forward(self, x):  
        # 前向传播输入值
        x = F.relu(self.hidden(x))      # 激励函数ReLU处理隐含层的输出
        x = self.out(x)             # 最终输出值
        return x

CUDA = torch.cuda.is_available()

if CUDA:
	#初始化输入神经元数目为1，隐含层数目为40，输出神经元数目为1的神经网络模型
	net = Net(input_feature=1, num_hidden=40, outputs=1).cuda()
	inputs = x.cuda()
	target = y.cuda()
else:
	net = Net(input_feature=1, num_hidden=40, outputs=1)
	inputs = x
	target = y


# optimizer 是训练的工具
optimizer = optim.SGD(net.parameters(), lr=0.01)  # 传入 net 的所有参数, 学习率
criterion = nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)
 
def draw(output,loss):
	
	if CUDA:
		output = output.cpu()
	plt.cla()
	plt.scatter(x.numpy(), y.numpy())
	plt.plot(x.numpy(), output.data.numpy(),'r-', lw=5)
	plt.text(-2,-20,'Loss=%s' % (loss.item()),fontdict={'size':20,'color':'red'})
	plt.pause(0.005)

def train(model,criterion,optimizer,epochs):
	for epoch in range(epochs):
		#forward
		output = model(inputs)
		loss = criterion(output,target)

		#backward
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()#所有的optimizer都实现了step()方法，这个方法会更新所有的参数。它能按两种方式来使用

		if epoch % 80 == 0:
			draw(output,loss)
		#plt.savefig('plot2.png', format='png')

	return model,loss

net,loss = train(net,criterion,optimizer,10000)

print("final loss:",loss.item())