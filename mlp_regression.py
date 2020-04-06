import numpy as np 
from numpy import vstack
from torch.utils.data import Dataset
from pandas import read_csv




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

