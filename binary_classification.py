from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch import Tensor
from torch.utils.data import random_split
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
        df = read_csv(path, header=None)
        ### Store the inputs and outputs
        self.X = df.values[:,:-1]
        self.y = df.values[:,-1]
        ## make them floats
        self.X = self.X.astype('float32')
        ## encode the targets
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')
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
    data = CSVDataset(path)
    train, test = data.get_splits()
    train_dl = DataLoader(train, batch_size = 32, shuffle = True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)

    return train_dl, test_dl

## Train the model

def train_model(train_dl, model):
    ## Optimizer 
    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr = 0.01, momentum=0.9)
    ## enumerate epochs
    for epoch in range(100):
        for i, (inputs, targets) in enumerate(train_dl):
            ##Clear the gradients
            optimizer.zero_grad()
            ## compute the model output
            yhat = model(inputs)
            ## Calculate the loss
            loss = criterion(yhat, targets)
            ## credit assignment
            print(loss)
            loss.backward()
            ## update model weights
            optimizer.step()
## evaluate the model

def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        ## evaluate the model on the test set
        yhat = model(inputs)
        ## retrieve np arrays
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual),1))
        ## Round into class values
        yhat = yhat.round()
        #store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actual)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)

    return acc

## make a class prediction for one row of data
def predict(row, model):
    #convert row to data
    row = Tensor([row])
    #make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat

# Prepare the data
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv'
train_dl, test_dl = prepare_data(path)
print(len(train_dl.dataset), len(test_dl.dataset))
## define the network
model = MLP(34)
## Train the model
train_model(train_dl, model)
## Evaluate the model
acc = evaluate_model(test_dl, model)
print('Accuracy: %.3f',acc)
# make a single prediction (expect class=1)
row = [1,0,0.99539,-0.05889,0.85243,0.02306,0.83398,-0.37708,1,0.03760,0.85243,-0.17755,0.59755,-0.44945,0.60536,-0.38223,0.84356,-0.38542,0.58212,-0.32192,0.56971,-0.29674,0.36946,-0.47357,0.56811,-0.51171,0.41078,-0.46168,0.21266,-0.34090,0.42267,-0.54487,0.18641,-0.45300]
yhat = predict(row, model)
print('Predicted: %.3f (class=%d)' % (yhat, yhat.round()))




