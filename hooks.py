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
