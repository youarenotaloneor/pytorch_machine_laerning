import torch.multiprocessing as mp

def action(q,name,times):
	init = 0 
	for i in range(times):
		init += i
	print("this process is " + name)
	q.put(init)

if __name__ =='__main__':
	q = mp.Queue()
	process1 = mp.Process(target=action,args=(q,'process1',10000000))
	process2 = mp.Process(target=action,args=(q,'process2',1000))

	process1.start()
	process2.start()

	process1.join()
	process2.join()
	
	result1 = q.get()
	result2 = q.get()

	print(result1)
	print(result2)
	print("main process")