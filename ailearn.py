import torch

features = torch.tensor([
    [1.0,2.0],
    [3.0,4.0],
    [5.0,6.0],
    [7.0,8.0],
    [9.0,10.0],
])

y_true = torch.tensor([
    [8.0],
    [9.0],
    [10.0],
    [7.0],
    [15.0]
])

w = torch.randn(2,1)
b = torch.randn(1)

Y_prediction = features @ w + b

print(Y_prediction)