import torch.multiprocessing as mp

def action(times):
	init = 0 
	for i in range(times):
		init += i
	return init


if __name__ =='__main__':
	times = [1000,1000000]
	pool = mp.Pool(processes=2)
	res = pool.map(action,times)
	print(res)