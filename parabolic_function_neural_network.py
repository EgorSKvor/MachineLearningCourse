import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (25.0,7.0)

#Входящие данные сломанная косинусоида
x_train=torch.rand(1000)
x_train=x_train * 10.0 - 5.0
y_sub_train=torch.cos(x_train)**2
noisy=torch.rand(y_sub_train.shape) / 6
y_train=y_sub_train + noisy
x_train.unsqueeze_(1)
y_train.unsqueeze_(1)

# На данной модели происходит обучение (правильная косинусоида)
x_val = torch.linspace(-5,5,1000)
y_val = torch.cos(x_val.data)**2
x_val.unsqueeze_(1)
y_val.unsqueeze_(1)

class OurNet(torch.nn.Module):
    def __init__(self, n_hid_n):
        super(OurNet, self).__init__()
        self.fc1 = torch.nn.Linear(1, n_hid_n)
        self.act1 = torch.nn.Sigmoid()
        self.fc4 = torch.nn.Linear(n_hid_n, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc4(x)
        return x

our_net = OurNet(100)

def predict(net, x, y):
    y_pred = net.forward(x)
    plt.plot(x.numpy(), y.numpy(), 'o', c='g', label='То что должно быть')
    plt.plot(x.numpy(), y_pred.data.numpy(), 'o', c='r', label='Предсазание сети')
    plt.legend(loc='upper left')
    plt.show()

predict(our_net, x_val, y_val)
optimizer = torch.optim.Adam(our_net.parameters(), lr=0.001)

def loss(pred, true):
  sq = (pred-true)**2
  return sq.mean()

for i in range(30000):
  optimizer.zero_grad()
  y_pred = our_net.forward(x_train)
  loss_val = loss(y_pred,y_train)
  loss_val.backward()
  optimizer.step()

  if not i % 2000:
    print(loss_val)

predict(our_net, x_val, y_val)
