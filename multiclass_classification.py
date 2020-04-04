'''
We use the iris dataset to classify different 
iris species using a multi-layered perceptron
'''

from numpy import vstack
import torch
from torch.nn import Module
from torch.nn import Linear
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import CrossEntropyLoss
from torch import Tensor
from torch.optim import SGD
from torch.utils.data import Dataset
from torch.utils.data import random_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import test_train_split

class CSVDataset(Dataset):
    # Load the dataset
    def __init__(self, path):
        df = pd.read_csv(path, header=None)
        ## Store the inputs and outputs
        self.X = df.values[:,:-1]
        self.y = df.values[:,-1]
        ## Ensure the inputs are floats
        self.X = self.X.astype('float32')
        ##Label Encode the y values
        self.y = LabelEncoder().fit_transform(self.y)

    #number of rows in dataset
    def __len__(self):
        return len(self.X)

    #Get an item 
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]
    
    ## get indices for train/test split
    def get_splits(self, n_test=0.33):
        ## Determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        ## calculate the split
        return random_split(self,[train_size, test_size])


## Model Definition

class MLP(Module):
    ## model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        ## 1st hidden layer 
        self.hidden1 = Linear(n_inputs, 10)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        ## second hidden layer
        self.hidden2 = Linear(10,8)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        self.hidden3 = Linear(8, 3)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Softmax(dim=1)
     
    def forward(self, X):
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.act3(X)
        return X

def prepare_dataset(path):
    ## load the dataset
    dataset = CSVDataset(path)
    #calculate split
    train, test = dataset.get_splits()
    ## Data Loaders
    train_dl = DataLoader(train, batch_size = 32, shuffle=True)
    test_dl = DataLoader(test, batch_size = 1024, shuffle=False)
    return train_dl, test_dl

def train_model(train_dl, model):
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr = 0.01, momentum=0.9)
    # enumerate epochs
    for epoch in range(500):
        #enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            ## clear the gradients
            optimizer.zero_grad()
            ## compute the model output
            yhat = model(inputs)
            ## calculate the loss
            loss = criterion(yhat, targets)
            ## credit assignments
            loss.backward()
            ## Update model weights
            optimizer.step()

def evaluate_model(test_dl, model):
    predictions, actual = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # Evaluate model
        yhat = model(inputs)
        # Retrieve numpy array
        yhat = yhat.detech().numpy()
        actual = targets.numpy()
        yhat = argmax(yhat, axis=1)
        # reshape for stacking
        actual = actual.reshape((len(actual), 1))
        yhat = yhat.reshape((len(yhat), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    ## calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc

def predict(row, model):
    ## convert row to data
    row = Tensor([row])
    ## make a prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()

    return yhat
