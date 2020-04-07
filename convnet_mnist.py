from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from matplotlib import pyplot
from torch.nn import module 
from torch.nn.init import kaiming_uniform_
from torch.nn import ReLU
from torch.nn import Conv2d


class CNN(Module):
    def __init__(self, n_channels)
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
    self.act4 = SoftMax(dim=1)

## Location to save or load the data
path = '~/.torch/datasets/mnist'
## Transforms to perform to the images
trans = Compose([ToTensor()])
## Download and define the datasets
train = MNIST(path, train=True, download=True, transform = trans)
test = MNIST(path, train =False, download = True, transform = trans)
## Define how to enumerate the datasets
train_dl = DataLoader(train,batch_size = 32, shuffle=True)
test_dl = DataLoader(test, batch_size= 32, shuffle=True)
## How to enumerate the datasets
i, (inputs, targets) = next(enumerate(train_dl))
## plot some images
for i in range(25):
    pyplot.subplot(5,5, i+1)
    pyplot.imshow(inputs[i][0], cmap='gray')

pyplot.show()


