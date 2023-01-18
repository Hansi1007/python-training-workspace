import torch
import torch.nn as nn

print('Script is running')

X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)
X_test = torch.tensor([5], dtype=torch.float32)

number_sample, number_feature = X.shape
print(number_sample, number_feature)
print(X)

input_size = number_feature
output_size = number_feature

model = nn.Linear(input_size ,output_size , bias=True)

# w = torch.tensor([1,2,3,4], dtype=torch.float32, requires_grad=True)
# def forward(x):
#    return w * X

learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)


print(f'Prediction befor training f(5) = {model(X_test).item():.3f} ')

for epoch in range(n_iters):
    y_pred = model(X)

    l = loss(y_pred, Y)

    l.backward() # dl/dw

    # update weigths
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w,b] = model.parameters()
        print(f'epoch {epoch}  w = {w[0][0].item():.3f}  loss = {l:.8f}')

print(f'Prediction after training f(5) = {model(X_test).item():.3f} ')







