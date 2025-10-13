import torch
import numpy as np


x = torch.tensor(2.0, requires_grad=True)

y = x**2+x*3+4

y.backward()
print(x.grad)

randomTensor = torch.rand(3,3)
print(randomTensor)