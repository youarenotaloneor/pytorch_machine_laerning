#coding=utf-8

import torch
import matplotlib.pyplot as plt
from time import perf_counter

#准备数据

#生成矩阵X
def Produce_X(x):
	x0 = torch.ones(x.numpy().size) #用ones产生初始值为1，大小与x相同的向量
	X = torch.stack((x,x0),dim=1)   #stack函数将两个向量拼合
	print(X)
	return X


x = torch.linspace(-3,3,100000)#用linspace产生（-3，3）区间内的100000个点
X = Produce_X(x)
y = x +1.2*torch.rand(x.size())#假设真实函数是y=x，我们在上面增加一些误差，更加符合实际情况
w = torch.rand(2) #定义权重w的变量
print(w)

'''
#散点图查看样本数据的分布情况
plt.scatter(x.numpy(),y.numpy(),s=0.001)
plt.show()
'''

#如果支持CUDA，则采用CUDA加速
CUDA =  torch.cuda.is_available()

if CUDA:
	inputs = X.cuda() 
	target = y.cuda()
	w = w.cuda()
	w.requires_grad=True
else:
	inputs = X 
	target = y
	w = w
	w.requires_grad=True

#可视化
#plt.ion() 
#plt.show()

def draw(output,loss):
	#print loss
	if CUDA:
		output= output.cpu()
	plt.cla()
	plt.scatter(x.numpy(), y.numpy())
	plt.plot(x.numpy(), output.data.numpy(),'r-', lw=5)
	plt.text(0.5,0,'Loss=%s' % (loss.item()),fontdict={'size':20,'color':'red'})
	plt.pause(0.005)#绘图延迟，并且保留之前所化的图像

def train(epochs=1,learning_rate=0.01):
	for epoch in range(epochs):
	
		#前向传播
		output = inputs.mv(w) #公式：y=Xw
		loss = (output - target).pow(2).sum()/100000 #公式：J = (∑(y-y')^2)/100000

		#反向传播
		loss.backward() 
		w.data -= learning_rate * w.grad  #更新权重w，公式：w_(t+1)= w_(t) - 𝜼*▽J
		w.grad.zero_() #清空grad的值
		
		
		if epoch % 80 == 0:
			draw(output,loss)
		

	return w,loss

start = perf_counter()
w,loss = train(10000,learning_rate=1e-4)  #学习率设置为1x10^(-4)
finish = perf_counter()
time = finish-start

print("计算时间:%s" % time)
print("final loss:",loss.item())
print("weights:",w.data)
