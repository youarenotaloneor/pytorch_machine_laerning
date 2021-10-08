import torch.multiprocessing as mp
import time

def action(v,num,lock):
	lock.acquire()
	for i in range(5):
		time.sleep(0.1)
		v.value += num
		print(v.value)
	lock.release()


if __name__ == "__main__":
	lock = mp.Lock()
	v = mp.Value('i',0)
	p1 = mp.Process(target=action,args=(v,1,lock))
	p2 = mp.Process(target=action,args=(v,2,lock))
	p1.start()
	p2.start()
	p1.join()
	p2.join()