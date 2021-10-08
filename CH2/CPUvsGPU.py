#coding=utf-8
import torch
from time import perf_counter

x = torch.rand(1000,10000)
y = torch.rand(10000,10000)

#CPU
start = perf_counter()
x.mm(y)
finish = perf_counter()
time = finish-start
print("CPU计算时间:%s" % time)

#GPU
if torch.cuda.is_available():
	x = x.cuda()
	y = y.cuda()
	start = perf_counter()
	x.mm(y)
	finish = perf_counter()
	time_cuda = finish-start
	print("GPU加速计算的时间:%s" % time_cuda)
	print("CPU计算时间是GPU加速计算时间的%s倍" % str(time/time_cuda))
	
else:
	print("未支持CUDA")
	

