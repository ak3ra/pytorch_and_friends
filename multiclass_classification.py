'''
We use the iris dataset to classify different iris species using a multi-layered perceptron
'''

from numpy import vstack
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder


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

     

