from numpy import vstack
from numpy import sqrt
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader
from pandas import read_csv
from torch.nn import Module
from torch.nn.init import xavier_uniform_
from torch.nn import Linear, Sigmoid, MSELoss
from torch import Tensor
from torch.optim import SGD



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
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle = False)
    return test_dl, train_dl

def train_model(train_dl, model):
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr = 0.01, momentum = 0.9)
    for epoch in range(100):
        for i, (inputs, targets) in enumerate(train_dl):
            optimizer.zero_grad()
            yhat = model(inputs)
            loss = criterion(yhat, targets)
            loss.backward()
            optimizer.step()

def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        yhat = model(inputs)
        yhat = yhat.detach().numpy()
        actual = actual.reshape()
        predictions.append(yhat)
        actuals.append(actual)

    predictions, actuals = vstack(predictions), vstack(actuals)
    mse = means_squared_error(actuals, predictions)
    return mse


def predict(row, model):
    row  = Tensor([row])
    yhat = model(row)
    yhat = yhat.detach().numpy()
    return yhat 

path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
train_dl, test_dl = prepare_data(path)
print(len(train_dl.dataset), len(test_dl.dataset))
model = MLP(13)
train_model(train_dl, model)
mse = evaluate_model(test_dl, model)
print('MSE: %.3f, RMSE: %.3f' % (mse, sqrt(mse)))
# make a single prediction (expect class=1)
row = [0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98]
yhat = predict(row, model)
print('Predicted: %.3f' % yhat)

