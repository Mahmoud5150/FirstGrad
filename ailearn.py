import torch
import torch.nn.functional as F

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

w = torch.randn(2,1 , requires_grad=True)
b = torch.randn(1, requires_grad=True)

print(f"Old weight: {w}")
print(f"Old bias: {b}")

lr = 0.001

for epoch in range(5000):

    Y_prediction = F.relu(features @ w + b)
    loss = ((Y_prediction-y_true)**2).mean()

    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    w.grad.zero_()
    b.grad.zero_()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:4d} | Loss : {loss.item():.6f}")

print(f"New weight: {w}")
print(f"New bias: {b}")
print(f"New y pred: {Y_prediction}")
print(f"New loss: {loss.item():.6f}")