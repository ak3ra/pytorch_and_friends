from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder

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
        test_size = round(n_test * )




