#coding=utf-8

import torch
import matplotlib.pyplot as plt

#数据
x=torch.Tensor([1.4,5,11,16,21])
y=torch.Tensor([14.4,29.6,62,85.5,113.4])

#将x，y转化为numpy数据类型，绘制散点图
plt.scatter(x.numpy(),y.numpy())
plt.show()