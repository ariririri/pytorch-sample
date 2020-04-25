"""irisに対するテスト
"""

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(4, 10)
        self.l2 = nn.Linear(10, 8)
        self.l3 = nn.Linear(8, 3)
    
    def forward(self ,x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

def train_epoch(model, data_loader):
    model.eval()
    for batch_idx, (data, target) in enumerate(data_loader): # 入力と正解
         optimizer.zero_grad() # Weightの初期化
         output = model(data) # 仮説で値代入
         output.dtype
         loss = criterion(output, target) # 損失
         loss.backward() # 微分の計算
         optimizer.step() # パラメータの更新
         if batch_idx % 10 == 0:
             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                 batch_idx, batch_idx * len(data), len(data_loader.dataset),
                 100. * batch_idx / len(data_loader), loss.item()))

def valid_epoch(model, data_loader):
    model.train()
    with torch.no_grad()
        for batch_idx, (data, target) in enumerate(data_loader): # 入力と正解
             optimizer.zero_grad() # Weightの初期化
             output = model(data) # 仮説で値代入
             output.dtype
             loss = criterion(output, target) # 損失
             # 本来は全体でロスを数えて荷重平均を取る,accuracyを計算する

             if batch_idx % 10 == 0:
                 print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                     batch_idx, batch_idx * len(data), len(data_loader.dataset),
                     100. * batch_idx / len(data_loader), loss.item()))

def main():
    try:
        iris = datasets.load_iris()
        X_train, X_valid, y_train, y_valid = train_test_split(iris.data, iris.target, test_size=0.2)

        X_train = torch.tensor(X_train).float()
        y_train = torch.tensor(y_train)

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        valid_dataset = torch.utils.data.TensorDataset(X_valid, y_valid)

        model = Net()

        batch_size  = 3 # ミニバッチのデータの数
        max_epoch = 100 #
        train_loader = torch.utils.data.DataLoader(dataset, 
                           batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss() # 損失の定義
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01) #(確率的)勾配降下法
    
        for epoch in range(max_epoch):
            train_epoch(model, train_loader)
            valid_epoch(model, test_loader)
    except:
        import traceback
        print(traceback.format_exc())
        import IPython
        IPython.embed()

    
if __name__ == "__main__":
    main()