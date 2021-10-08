import torch.multiprocessing as mp
import torch

def action(element,t):
	t[element] += (element+1) * 1000

if __name__ == "__main__":
	t = torch.zeros(2)
	t.share_memory_()
	print('before mp: t=')
	print(t)

	p0 = mp.Process(target=action,args=(0,t))
	p1 = mp.Process(target=action,args=(1,t))
	p0.start()
	p1.start()
	p0.join()
	p1.join()
	print('after mp: t=')
	print(t)