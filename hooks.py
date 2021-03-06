import torch
import torch.nn as nn
a = torch.ones(5)
a.requires_grad = True

b = 2*a

b.retain_grad() #

## nn.parameter

class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Linear(10, 5)
    def forward(self, x):
        return self.linear(x)

myNet = net()
#print(list(myNet.parameters())) 

###### net 2 ###############

class net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Linear(10, 5)
        self.tens = nn.Parameter(torch.ones(3,4))

    def forward(self, x):
        return self.linear(x)
    
new_net = net2()
print(list(new_net.parameters()))

###################

class net3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Linear(10, 5)
        self.net = net2()

    def forward(self, x):
        return self.linear(x)

########### simplification by using a list ######

layer_list = [nn.Conv2d(5,5,3), nn.BatchNorm2d(5), nn.Linear(5,2)]

class mynet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = layer_list
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

net = mynet()
# print(list(net.parameters()))

################ wrapping the list within the nn.ModuleList class 

layer_list = [nn.Conv2d(5,5,3), nn.BatchNorm2d(5), nn.Linear(5,2)]

class mynetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(layer_list)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

net = mynetwork()
print(list(net.parameters()))


### For weight initialization ####

import matplotlib.pyplot as plt
# %matplotlib inline

class myneto(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(10,10,3)
        self.bn = nn.BatchNorm2d(10)

    def weights_init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean =0, std = 1)
                nn.init.constant_(module.bias, 0)
net = myneto()
net.weights_init()

for module in net.modules():
    if isinstance in net.modules():
        weights = module.weight
        weights = weights.reshape(-1).detach().cpu.numpy()
        print(module.bias)
        plt.hist(weights)
        plt.show()
        plt.save("weights.png")
