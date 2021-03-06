import numpy as np 
from numpy import vstack
import pandas as pd
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from matplotlib import pyplot
from torch.nn.init import kaiming_uniform_,xavier_uniform_
from torch.nn import ReLU,Softmax
from torch.optim import SGD
from torch.nn import Conv2d, MaxPool2d
from torch.nn import Linear,CrossEntropyLoss, Module
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


class CNN(Module):
    def __init__(self, n_channels): 
        super(CNN,self).__init__()
        self.hidden1 = Conv2d(n_channels, 32, (3,3))
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        self.pool1 = MaxPool2d((2,2), stride=(2,2))
        self.hidden2 = Conv2d(32,32,(3,3))
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        self.pool2 = MaxPool2d((2,2), stride=(2,2))
        self.hidden3 = Linear(5*5*32, 100)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = ReLU()
        self.hidden4 = Linear(100, 10)
        xavier_uniform_(self.hidden4.weight)
        self.act4 = Softmax(dim=1)

    def forward(self, X):
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.pool1(X)
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.pool2(X)
        ## Flatten
        X = X.view(-1, 4*4*50)
        X = self.hidden3(X)
        X = self.act3(X)
        ## Output
        X = self.hidden4(X)
        X = self.act4(X)
        return X

def prepare_data(path):
    trans = Compose([ToTensor()])
    ## Download and define the datasets
    train = MNIST(path, train=True, download=True, transform = trans)
    test = MNIST(path, train =False, download = True, transform = trans)
    ## Define how to enumerate the datasets
    train_dl = DataLoader(train,batch_size = 32, shuffle=True)
    test_dl = DataLoader(test, batch_size= 32, shuffle=False)
    ## How to enumerate the datasets
    return train_dl, test_dl

def train_model(train_dl, model):
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr = 0.01, momentum=0.9)
    ## enumerate epochs
    for epoch in range(10):
        ## Enumerate minibatches
        for i, (inputs, targets) in enumerate(train_dl):
            optimizer.zero_grad()
            yhat = model(inputs)
            loss = criterion(yhat, targets)
            loss.backward()
            optimizer.step()
            writer.add_scalar('Batch Train Loss', loss, i)
        writer.add_scalar('Epoch Train Loss', loss, epoch)



def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        yhat = model(inputs)
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        yhat = np.argmax(yhat, axis=1)
        actual = actual.reshape((len(actual),1))
        yhat = yhat.reshape((len(yhat), 1))
    predictions,actuals = vstack(predictions), vstack(actuals)
    acc = accuracy_score(actuals, predictions)
    print("val_acc:",acc)
    writer.add_scalar('Val Acc', acc, epoch)
    return acc

## Location to save or load the data
path = '~/.torch/datasets/mnist'
## Transforms to perform to the images


train_dl, test_dl = prepare_data(path)
print(len(train_dl.dataset), len(test_dl.dataset))
# define the network
model = CNN(1)
# # train the model
train_model(train_dl, model)
# evaluate the model
acc = evaluate_model(test_dl, model)
print('Accuracy: %.3f' % acc)