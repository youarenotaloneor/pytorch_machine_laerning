import torch.multiprocessing as mp
import time

def action(name,times):
	init = 0 
	for i in range(times):
		init += i
	print("this process is " + name)

def mpfun():
	process1 = mp.Process(target=action,args=('process1',100000000))
	process2 = mp.Process(target=action,args=('process2',100000000))

	process1.start()
	process2.start()

	process1.join()
	process2.join()

def spfun():
	action('main process',100000000)
	action('main process',100000000)

if __name__ =='__main__':
	start_time = time.time()
	mpfun()
	end_time = time.time()
	print(end_time-start_time)
	
	start_time2 = time.time()
	spfun()
	end_time2 = time.time()
	print(end_time2-start_time2)