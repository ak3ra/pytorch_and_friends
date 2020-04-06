import numpy as np 
from numpy import vstack
from torch.utils.data import Dataset
from torch.utils.data import random_splt
from pandas import read_csv
from torch.nn import Module
from torch.nn.init import xavier_uniform_
from torch.nn import Linear, Sigmoid



class CSVDataset(Dataset):
    ## load the dataset
    def __init__(self, path):
        ##laod the csv  file as a dataframe
        df = read_csv(path)
        self.X = df.values[:,:-1].astype('float32')
        self.y = df.values[:,-1].astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):
        return [self.X[idx], self.y[idx]]

    def get_splits(self, n_test=0.33):
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        ## calculate the split
        return random_split(self, [train_size, test_size])
    
class MLP(Module):
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        ## input to the first layer
        self.hidden1 = Linear(n_inputs, 10)
        xavier_uniform_(self.hidden1.weight)
        self.act1 = Sigmoid()
        # second hidden layer
        self.hidden2 = Linear(10,8)
        xavier_uniform_(self.hidden2.weight)
        self.act2 = Sigmoid()
        self.hidden3 = Linear(8,1)
        xavier_uniform_(self.hidden3.weight)

    ## forward prop
    def forward(self, X):
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.hidden3(X)
        return X
    

def prepare_data(path):
    dataset = CSVDataset(path)
    train, test = dataset.get_splits()
    train_dl = Data

