import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(3,32,3,1,1)
        self.dense1 = nn.Linear(32 * 3 * 3,128)
        self.dense2 = nn.Linear(128,10) 

    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = x.view(x.size(0),-1)
        x = F.relu(self.dense1(x))
        x = self.dense(x)
        return x

class Net2(nn.Module):
    def __init__(self):
        super(Net2,self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3,32,3,3,1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.dense = nn.Sequential(
            nn.Linear(32*3*3,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )

        self.test = 100

    def forward(self,x):
        out = self.conv(x)
        out = out.view(out.size(0),-1)
        out = self.dense(out)
        return out

class Net3(nn.Module):
    def __init__(self):
        super(Net3,self).__init__()

        self.conv = nn.Sequential()
        self.conv.add_module("conv1",nn.Conv2d(3,32,3,1,1))
        self.conv.add_module("relu1",nn.ReLU())
        self.conv.add_module("pool1",nn.MaxPool2d(2,2))

        self.dense = nn.Sequential()
        self.dense.add_module("dense1",nn.Linear(32 * 3 *3 ,128))
        self.dense.add_module("relu2",nn.ReLU())
        self.dense.add_module("dense2",nn.Linear(128,10))

    def forward(self,x):
        out = self.conv(x)
        out = out.view(out.size(0),-1)
        out =self.dense(out)
        return out

class Net4(nn.Module):
    def __init__(self):
        super(Net4,self).__init__()

        self.conv = nn.Sequential(
            OrderedDict([
                ("conv1",nn.Conv2d(3,32,3,1,1)),
                ("relu1",nn.ReLU()),
                ("pool1",nn.MaxPool2d(2))
            ])
        )

        self.dense = nn.Sequential(
            OrderedDict([
                ("dense1",nn.Linear(32*3*3,128)),
                ("relu2",nn.ReLU()),
                ("dense2",nn.Linear(128,10))
            ])
        )

    def forward(self,x):
        out = self.conv(x)
        out = out.view(out.size(0),-1)
        out =self.dense(out)
        return out

m = Net4()
print(m)