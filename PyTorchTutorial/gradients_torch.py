import torch

print("Running")

X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)

w = torch.tensor(0.0, requires_grad=True,  dtype=torch.float32)

def forward(x):
    return w*x

def loss(y, y_predited):
    return ((y_predited - y)**2).mean()

print(f'Prediction before training f(5) = {forward(5):.3f}')


learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
    #Training
    #prediction forward pass
    y_pred = forward(X)

    # Cost function MSE
    l = loss(Y, y_pred)

    # Gradient = backward
    l.backward()  #dl/dw

    # update weights
    with torch.no_grad():
        w -= learning_rate * w.grad
    w.grad.zero_()

    if (epoch+1) % 10 == 0:
        print(f'epoch {epoch+1}  w = {w.item():.3f} loss = {l.item():.8f}')


print(f'Prediction after training f(5) = {forward(5):.3f}')




