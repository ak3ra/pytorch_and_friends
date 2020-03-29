from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from torch.nn import Linear 
from torch.nn import ReLU
from torch.nn import BCELoss
from torch.nn import Module
from torch.optim import SGD
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import Sigmoid

## Dataset definition

class CSVDataset(Dataset):
    def __init__(self, path):
        ##Load the csv dataset as Dataframe
        df = pd.read_csv(path, header=None)
        ### Store the inputs and outputs
        self.X = df.values[:,:-1]
        self.y = df.values[:,-1]
        ## make them floats
        self.X = self.X.astype('float32')
        ## encode the targets
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

    ## number of rows in the dataset
    def __len__(self):
        return len(self.X)
    ## get a row from the dataset

    def __getitem__(self, index):
        return [self.X[index], self.y[index]]

    ### get index for test and train rows
    def get_splits(self, n_test = 0.33):
        #determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        ## calculate the split
        return random_split(self,[train_size, test_size])

## model definition

class MLP(Module):
    ## define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 10)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(10,8)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        self.hidden3 = Linear(8,1)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Sigmoid()

        ## Forward propagate input

        def forward(self, X):
            X = self.hidden1(X)
            X = self.act1(X)
            X = self.hidden2(X)
            X = self.act2(X)
            X = self.hidden3(X)
            X = self.act3(X)

            return X

def prepare_data(path):
    data = CSVDataset(path)<<
    train, test = data.get_splits()
    train_dl = DataLoader(train, batch_size = 32, shuffle = True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)

    return train_dl, test_dl

## Train the model

def train_model(train_dl, model):
    ## Optimizer 
    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr = 0.01, )






