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
from torch import Tensor
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

    def data

