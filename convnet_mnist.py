from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose
from matplotlib import pyplot


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


