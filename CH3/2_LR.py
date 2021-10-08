#coding=utf-8

import torch
import matplotlib.pyplot as plt

#准备数据

#生成矩阵X
def Produce_X(x):
	x0 = torch.ones(x.numpy().size) #用ones产生初始值为1，大小与x相同的向量
	X = torch.stack((x,x0),dim=1)   #stack函数将两个向量拼合
	return X


x = torch.Tensor([1.4,5,11,16,21])
y = torch.Tensor([14.4,29.6,62,85.5,113.4])
X = Produce_X(x)

#定义权重w的变量
w = torch.rand(2,requires_grad=True)

inputs = X 
target = y

#可视化
#plt.ion() 
#plt.show()

#绘图
def draw(output,loss):
	plt.cla()
	plt.scatter(x.numpy(), y.numpy())
	
	plt.plot(x.numpy(), output.data.numpy(),'r-', lw=5)
	plt.text(0.5, 0,'Loss=%s' % (loss.item()),fontdict={'size':20,'color':'red'})
	#plt.text(3, 9,'Loss=%s' % (loss.item()),fontdict={'size':20,'color':'red'})
	#plt.axis([10, 160, 0, 0.03])

	plt.pause(0.005)

#训练
def train(epochs=1,learning_rate=0.01):
	for epoch in range(epochs):

		#前向传播
		output = inputs.mv(w) #公式：y=Xw
		loss = (output - target).pow(2).sum()#公式：J = ∑(y-y')^2

		#反向传播
		loss.backward() 
		w.data -= learning_rate * w.grad  #更新权重w，公式：w_(t+1)= w_(t) - 𝜼*▽J
		
		w.grad.zero_() #清空grad的值

		if epoch % 80 == 0:
			draw(output,loss)

	#plt.savefig('plot1.png', format='png')

	return w,loss

		
w,loss = train(10000,learning_rate = 1e-4)  #学习率设置为1x10^(-4)


'''
# 静态显示最后图像
plt.ioff()  
plt.show()

'''

print("final loss:",loss.item())
print("weights:",w.data)
