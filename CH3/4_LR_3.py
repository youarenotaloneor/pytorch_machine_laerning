#coding=utf-8

import torch
import matplotlib.pyplot as plt
from torch import nn,optim
from time import perf_counter

#用linspace产生（-3，3）区间内的100000个点，并使用unsqueeze函数增加一个维度
x = torch.unsqueeze(torch.linspace(-3,3,100000),dim=1)

#假设真实函数是y=x，我们在上面增加一些误差，更加符合实际情况
y = x +1.2*torch.rand(x.size())


class LR(nn.Module):
	def __init__(self):
		super(LR,self).__init__()
		self.linear = nn.Linear(1,1)

	def forward(self,x):
		out = self.linear(x)
		return out

#如果支持CUDA，则采用CUDA加速
CUDA = torch.cuda.is_available()

if CUDA:
	LR_model = LR().cuda()
	inputs = x.cuda()
	target = y.cuda()
else:
	LR_model = LR()
	inputs = x
	target = y

criterion = nn.MSELoss()
optimizer = optim.SGD(LR_model.parameters(),lr=1e-4)



#可视化
#plt.ion()
#plt.show()
def draw(output,loss):
	
	if CUDA:
		output = output.cpu()
	plt.cla()
	plt.scatter(x.numpy(), y.numpy())
	plt.plot(x.numpy(), output.data.numpy(),'r-', lw=5)
	plt.text(0.5,0,'Loss=%s' % (loss.item()),fontdict={'size':20,'color':'red'})
	plt.pause(0.005)

def train(model,criterion,optimizer,epochs):
	for epoch in range(epochs):
		#forward
		output = model(inputs)
		loss = criterion(output,target)

		#backward
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		
		if epoch % 80 == 0:
			draw(output,loss)
		
	return model,loss


start = perf_counter()
LR_model,loss = train(LR_model,criterion,optimizer,10000)
finish = perf_counter()
time = finish-start
print("计算时间:%s" % time)
print("final loss:",loss.item())
print("weights:",list(LR_model.parameters()))







