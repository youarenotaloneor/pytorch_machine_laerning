#coding=utf-8
import visdom

#####显示文字#####
vis = visdom.Visdom()
vis.text('Hello, world!')


#####显示图像#####
from PIL import Image
import torchvision.transforms.functional as TF

demopic = Image.open("demopic.jpg")
image_tensor = TF.to_tensor(demopic)
print(image_tensor.shape)
vis.image(image_tensor)


#####显示多张图像#####
import torch

image_tensors = torch.Tensor([image_tensor.numpy(),image_tensor.numpy()])
print(image_tensors.shape)
vis.images(image_tensors)


#####二维散点图#####
import numpy as np


Y = np.random.rand(100)
old_scatter = vis.scatter(     # 新开一个scatter窗口
X=torch.rand(100,2),
Y=(Y[Y > 0] + 1.5).astype(int),
opts=dict(
    legend=['Apples', 'Pears'], #图例名称
    xtickmin=-1, #x轴的最小值
    xtickmax=2.5,	#x轴的最大值
    xtickstep=0.5,#x轴的步长
    ytickmin=-1, #y轴的最小值
    ytickmax=2.5,#y轴的最大值
    ytickstep=0.5,#y轴的步长
    markersymbol='dot'
),  
)

import time
time.sleep(2)

vis.update_window_opts(
win=old_scatter,                 # 更新窗口
opts=dict(
    legend=['Apples', 'Pears'],
    xtickmin=0,
    xtickmax=1,
    xtickstep=0.5,
    ytickmin=0,
    ytickmax=1,
    ytickstep=0.5,
    markersymbol='dot'
),
)


#####三维散点图#####

Y = np.random.rand(100)
three_d_scatter = vis.scatter(     # 新开一个scatter窗口
X=torch.rand(100,3),
Y=(Y[Y > 0] + 1.5).astype(int),
opts=dict(
    legend=['Apples', 'Pears'], #图例名称
    xtickmin=-1, #x轴的最小值
    xtickmax=2.5,	#x轴的最大值
    xtickstep=0.5,#x轴的步长
    ytickmin=-1, #y轴的最小值
    ytickmax=2.5,#y轴的最大值
    ytickstep=0.5,#y轴的步长
    markersymbol='dot'
),  
)


#####折线图#####
vis.line(X=torch.FloatTensor([1,2,3,4]), Y=torch.FloatTensor([3,7,2,6]))

#####直方图#####
vis.bar(X=torch.rand(20))

vis.bar(
    X=torch.randn(5,3).abs(),
    opts=dict(
        stacked=True,
        legend=['Facebook', 'Google', 'Twitter'],
        rownames=['2012', '2013', '2014', '2015', '2016']
    )
)

#####热力图#####
X=np.outer(np.arange(1, 6), np.arange(1, 11))
print(X)
X=torch.from_numpy(X)
vis.heatmap(
    X=X,
    opts=dict(
        columnnames=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
        rownames=['y1', 'y2', 'y3', 'y4', 'y5'],
        colormap='Electric',
    )
)


