import torch
a = torch.ones(5)
a.requires_grad = True

b = 2*a

b.retain_grad() #

##