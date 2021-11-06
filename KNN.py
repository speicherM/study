import torch
from torch import nn
import torchvision
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
training_data = datasets.MNIST(
    root="data",
    download=True,
    train=True,
    transform=ToTensor(),
)
print(len(training_data))
training_targets = training_data.targets
training_data = training_data.data.type(torch.float32).view(-1, 28 * 28).to(device)

test_data = datasets.MNIST(
    root="data",
    download=True,
    train=False,
    transform=ToTensor(),
)
test_targets = test_data.targets
test_data = test_data.data.type(torch.float32)


class KNearestNeighbour(nn.Module):
    def __init__(self, data=None, targets=None, p=2, k=1):
        super(KNearestNeighbour,self).__init__()
        if p == 1:
            self.disfc = self.__distance_L1
        else:
            self.disfc = self.__distance_L2
        self.data = data
        self.targets = targets
        self.k = k

    def setData(self, data=None, targets=None):
        self.data = data
        self.targets = targets

    def setK(self, K=1):
        self.k = K

    def __distance_L1(self, p1, p2):
        p = p1 - p2
        return torch.norm(p, p=1, dim=1)

    def __distance_L2(self, p1, p2):
        p = p1 - p2
        return torch.norm(p, p=2, dim=1)

    def predict(self, X):
        data = self.data
        targets = self.targets
        disfc = self.disfc
        pred_result = torch.zeros(X.shape[0])
        for i, x in enumerate(X):
            distances = disfc(data, x)
            _, index = torch.sort(distances)
            pred_result[i], _ = torch.mode(targets[index[:self.k]])
        return pred_result

    def __call__(self, x):
        return self.predict(x)


def getData(data, target, k_folds=5):
    index = np.arange(len(data))
    np.random.shuffle(index)
    data = data[index]
    target = target[index]
    step = (int) (len(data) / k_folds)
    #print(step,len(target))
    data_validation = data[:step]
    target_validation = target[:step]

    data_training = data[step + 1:]
    target_training = target[step + 1:]
    return data_training, target_training, data_validation, target_validation


def accuracy(pred, y):
    temp = torch.sum(pred == y).item()
    #print(pred,y)
    #os.system("pause")
    return temp * 1.0 / len(y)


KNN = KNearestNeighbour().to(device)
k_folds = 10
K = 10
accuracy_arr = np.ndarray(shape=(K, k_folds))
for k in range(1, K+1):
    for i in range(k_folds):
        data_training, target_training, \
        data_validation, target_validation \
            = getData(training_data, training_targets,k_folds)
        KNN.setK(k)
        KNN.setData(data_training, target_training)
        pred = KNN(data_validation)
        accuracy_arr[k - 1][i] = accuracy(pred, target_validation)
        print("-"*10)
#print(accuracy_arr.shape)
figure = plt.figure()
x = np.expand_dims(np.arange(1,K+1), 0).repeat(k_folds, axis=0).T
print("accuracy",accuracy_arr)
plt.scatter(x,accuracy_arr)
y = np.mean(accuracy_arr,axis=1)
x = np.arange(1,k+1)
plt.plot(x,y,'r')
plt.show()
writer = SummaryWriter("runs/boardTest")
writer.add_figure("SuerparameterSelection",figure)
writer.close()