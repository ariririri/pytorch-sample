import torch
import torch.nn as nn
import  torch.utils.data.dataset as dataset
from itertools import product
from torch import optim
import numpy as np
import torch.nn.functional as F

class PlusDataset(dataset.Dataset):
    def __init__(self):
        super(PlusDataset).__init__()
        self.dataset  = list(product([0, 1,2,3,4], [0, 1, 2, 3, 4]))
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        temp = torch.eye(5)
        inp = self.dataset[idx]
        out = inp[0] + inp[1]
        inp = torch.cat((temp[inp[0]], temp[inp[1]]),0).view(2, -1)
        inp = inp.float()
        
        return inp, torch.tensor(out)


class NaiveRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NaiveRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.hidden = torch.zeros(hidden_size)
        
    def forward(self, inputs):
        outputs = [self.hidden]
        seq_len = inputs.shape[1]
        batch_size = inputs.shape[0]
        for i in range(seq_len):
            inp = inputs[:, i]
            #self.hidden = self.hiddenになるとinplaceな演算扱いになり正しく動作しない.
            #こうあるべきかは不明
            hidden = F.relu(self.h2h(outputs[-1]) + self.i2h(inp))
            outputs.append(hidden)
        
        _y = torch.cat(outputs[1:]).view(seq_len, batch_size, -1)
        return torch.transpose(_y, 0, 1)

class NMath(nn.Module):
    def __init__(self):
        super(NMath, self).__init__()
        self.rnn = NaiveRNN(5, 5)
        self.l = nn.Linear(5, 10)
        
    def forward(self, x):
        pred = self.rnn(x)
        h = pred[:, -1]
        return self.l(h)

if __name__ == "__main__":

    n_math  = NMath()
    n_optimizer = optim.SGD(n_math.parameters(), lr=0.1)
    
    data = PlusDataset()
    dataloader = torch.utils.data.DataLoader(data, batch_size=5)
    criterion = nn.CrossEntropyLoss()
    
    for i in range(1000):
        for x, y in dataloader:
            n_optimizer.zero_grad()
            n_math.train()
            h = n_math(x)
            loss = criterion(h, y)
            torch.autograd.set_detect_anomaly(True)
            loss.backward(retain_graph=True)
            n_optimizer.step()
        if i % 100 == 0:
            print(loss) 
        
