import torch.nn as nn
import torch
import time

class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.conv1 =nn.Conv2d(1,10,2,1,2)
	def forward(self,data):
		print(data.size( ))
		print(data.device)
		result = self.conv1(data)

net1 = Net().cuda()
net1 = nn.DataParallel(net1,device_ids=[0,1])

start_time = time.time()
for i in range(2):
	a = torch.randn(40,1,400,400)
	net1(a)
end_time = time.time()
print(end_time-start_time)
