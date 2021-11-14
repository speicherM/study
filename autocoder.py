import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision
from torch import nn
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.encode = nn.Sequential(
            # 28x28
            nn.Conv2d(1, 128, 5),  # 24
            nn.BatchNorm2d(128),  #
            nn.ReLU(),
            nn.MaxPool2d(2,return_indices=True),  # 12

            nn.Conv2d(128, 128, 7),  # 6
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, return_indices=True),  # 3

            nn.Conv2d(128, 1, 3),  # 1
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(1, 128, 3),  # 3
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxUnpool2d(2),  # 6

            nn.ConvTranspose2d(128, 128, 7),  # 12
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxUnpool2d(2),  # 24

            nn.ConvTranspose2d(128, 1, 5),  # 28
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )

    def show_activate_map(self, x):
        plt.figure(0)
        y = x.view(-1, 1).cpu().detach().numpy()
        plt.hist(y)
        for i in range(len(self.encode)):
            if i % 3 == 2:
                y = self.encode[i](x)
                temp = y.view(-1, 1).cpu().detach().numpy()
                # plt.figure(i)
                # plt.hist(temp)
            # if i == len(self.encode)-1:
            #     print(y.item())
            x = self.encode[i](x)
        print(x.shape)

        for i in range(len(self.decode)):
            if i % 3 == 2:
                print(i, self.decode[i])
                y = self.decode[i](x)
                temp = y.view(-1, 1).cpu().detach().numpy()
                plt.figure(i)
                plt.hist(temp)
            # if i == len(self.encode)-1:
            #     print(y.item())
            x = self.decode[i](x)
        plt.show()

    def encode_call(self, x):
        indices = []
        for i in range(len(self.encode)):
            if i % 4 == 3:
                x, indices_ = self.encode[i](x)
                indices.append(indices_)
            else:
                x = self.encode[i](x)
        return x, indices

    def decode_call(self, x, indices):
        indices = iter(indices[::-1])
        for i in range(len(self.decode)):
            if i % 4 == 3:
                x = self.decode[i](x, next(indices))
            else:
                x = self.decode[i](x)
        return x

    def forward(self, x):
        x, indices = self.encode_call(x)
        x = self.decode_call(x, indices)
        # x = self.encode(x)
        # x = self.decode(x)
        return x


training_data = datasets.FashionMNIST(
    root="data",
    download=True,
    train=True,
    transform=ToTensor(),
)
training_data = datasets.MNIST(
    root="data",
    download=True,
    train=True,
    transform=ToTensor(),
)#.data[:640]

test_data = datasets.MNIST(
    root="data",
    download=True,
    train=False,
    transform=ToTensor(),
)

print(training_data)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data = DataLoader(training_data, batch_size=64, shuffle=True)
test_data = DataLoader(test_data,batch_size=64)
# writer = SummaryWriter("runs/boardTest")


try:
    net = torch.load('auto-encoder11.pth')
    index_array = np.array([])
    predict_array = np.array([])
    for x, y in test_data:
        x = x.to(device)
        x, _ = net.encode_call(x)
        predict_array = np.append(predict_array, x.cpu().detach().numpy())
        index_array = np.append(index_array, y.numpy())
    print(predict_array.shape, index_array.shape)
    figure = plt.figure()
    plt.scatter(index_array, predict_array)
    writer = SummaryWriter("./autocoder1")
    writer.add_figure("number and encode", figure)
    writer.close()
except:
    net = Net().to(device)
    #net =  torch.load('auto-encoder.pth').to(device)
    loss_fc = nn.L1Loss(size_average=True)  # confirm
    optim = torch.optim.Adam(net.parameters(), lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.9)
    epochs = 2
    loss_array = []
    batch_array = []
    os.system('rd /s/q autocoder')
    writer = SummaryWriter("./autocoder")
    for epoch in range(epochs):
        for batch, (x,y) in enumerate(data):
            # if batch == 10:#len(data) - 1:
            #     break
            x = x.to(device)
            ans = net(x)
            # print(x.shape)
            loss = loss_fc(ans, x)
            optim.zero_grad()
            loss.backward()
            optim.step()
            lr_scheduler.step()

            parameter = net.parameters()
            if batch % 100 == 0:
                print("loss" ,loss.item())
                torch.save(net, 'auto-encoder.pth')
                writer.add_scalar('train_loss', loss.item(), batch + epoch * (len(data) - 1))
                input_img = torchvision.utils.make_grid(x)
                output_img = torchvision.utils.make_grid(ans)
                img = torchvision.utils.make_grid([input_img,output_img])
                writer.add_image("image",img)
    index_array = np.array([])
    predict_array = np.array([])
    fask_img = torch.randn(1,1,28,28).to(device)
    print(fask_img.shape)
    writer.add_graph(net,fask_img)
    for x, y in test_data:
        x = x.to(device)
        x,_= net.encode_call(x)
        predict_array = np.append(predict_array, x.cpu().detach().numpy())
        index_array = np.append(index_array, y.numpy())
    print(predict_array.shape, index_array.shape)
    figure = plt.figure()
    plt.scatter(index_array, predict_array)
    writer.add_figure("number and encode", figure)
    writer.close()

