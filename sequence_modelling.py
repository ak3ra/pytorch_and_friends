import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 

torch.manual_seed(1)

lstm = nn.LSTM(3,3)
inputs = [torch.randn(1,3) for _ in range(5)]

hidden = (torch.randn(1,1,3),
        torch.randn(1,1,3))

for i in inputs:
    out, hidden = lstm(i.view(1,1,-1), hidden)

inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1,1,3), torch.randn(1,1,3))
out, hidden = lstm(inputs, hidden)
print(out)
print(hidden)

## prepare dataset

def prepare_sequence(seq, to_ix):
    idx = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

training_data = [
    ("Akera is a mathematician".split(),["DET","NN", "V", "DET", "NN"]),
    ("Everybody read the Harry Potter books".split(),["NN","V", "DET", "NN"])

]

word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
tag_to_idx = {"DET":0, "NN":1, "V":2}

EMBEDDING_DIM = 6
HIDDEN_DIM = 6

# Creating the model

class LSTMTagger(nn.Module):
    def __init__(self,embedding_dim, hiddend_dim,vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hiddend_dim = hiddend_dim
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        ## to complete
