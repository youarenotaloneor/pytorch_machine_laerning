#coding=utf-8
import torch
from torch.autograd import Function
import numpy as np

class SimpleOp(Function):

    @staticmethod
    def forward(ctx,input):
        
        #ctx为上下文context，save_for_backward函数可以将对象保存起来，用于后续的backward函数
        ctx.save_for_backward(input)

        #中间的计算完全可以使用numpy计算
        numpy_input = input.detach().numpy()
        result1 = numpy_input *4
        result2 = np.linalg.norm(result1, keepdims=True)

        #将计算的结果转换成Tensor，并返回
        return input.new(result2)

    @staticmethod
    def backward(ctx,grad_output):
        #backward函数的输出的参数个数需要与forward函数的输入的参数个数一致。
        #grad_output默认值为tensor([1.])，对应的forward的输出为标量。
        #ctx.saved_tensors会返回forward函数内存储的对象
        input, = ctx.saved_tensors
        grad = 2*(1/input.norm().item())*(2*input)

        #grad为反向传播后为input计算的梯度，返回后会赋值到input.grad属性
        return grad

simpleop = SimpleOp.apply

input = torch.Tensor([1,1])
input.requires_grad=True
print("input:",input)
result = simpleop(input)
print("result:",result)
result.backward()
print("input grad:",input.grad)
