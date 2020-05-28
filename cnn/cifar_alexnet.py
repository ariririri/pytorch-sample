"""
mnistをCNNで解く
"""
from pathlib import Path
from collections import namedtuple

import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter


def load_dataloader(hparams):
    transform = transforms.Compose(
        [transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 実際はnum_workersは1でいい.
    trainset = datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    
    testset = datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=hparams.batch_size,
                                             shuffle=False, num_workers=2)

    
    return train_loader, test_loader

class Hparams():
    batch_size = 16
    input_size = 784
    output_size = 10
    learning_rate = 0.01
    epoch_num = 5

def load_hparams():
    hparams = Hparams()

    return hparams


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def main():
    here = Path(__file__).resolve()
    log_dir = here.parents[1] / "logs"
    writer = SummaryWriter(log_dir=log_dir)

    hparams = load_hparams()
    train_loader, test_loader = load_dataloader(hparams)
    # 必要ない
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet(10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=hparams.learning_rate)

    for epoch in range(hparams.epoch_num):
        model.train()
        for idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        # trainとtestでロスの比較が違う
        writer.add_scalar("Loss/train", loss, epoch)
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
              epoch, epoch , hparams.epoch_num,
              100. * epoch / hparams.epoch_num, loss.item()))

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss += criterion(output, target).item() 
                pred = output.argmax(dim=1, keepdim=True) # 予測の最大値を取得
                correct += pred.eq(target.view_as(pred)).sum().item()
            loss /= len(test_loader.dataset)
            writer.add_scalar("Loss/valid", loss, epoch)
            writer.add_scalar("Accuracy/valid", correct / len(test_loader.dataset), epoch)
    
        writer.add_histogram("weight", model.cv1.weight, epoch)
        writer.add_image('test_image', data[0], epoch)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    
    

if __name__ == "__main__":
    main()
