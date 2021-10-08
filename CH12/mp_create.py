import torch.multiprocessing as mp

def action(name,times):
	init = 0 
	for i in range(times):
		init += i
	print("this process is " + name)


if __name__ =='__main__':
	process1 = mp.Process(target=action,args=('process1',10000000))
	process2 = mp.Process(target=action,args=('process2',1000))

	process1.start()
	process2.start()

	#process2.join()
	
	print("main process")