import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__() 
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.conv4 = nn.Conv2d(128, 256, 1)
        
        self.pool=nn.MaxPool2d(2,2)
        
        self.fc=nn.Linear(43264,1024)
        self.bn = nn.BatchNorm1d(1024)
        
        self.fc2=nn.Linear(1024,512)
        self.bn2 = nn.BatchNorm1d(512)
        
        self.dp1 = nn.Dropout(0.1)
        self.dp2 = nn.Dropout(0.2)
        self.dp3 = nn.Dropout(0.3)
        self.dp4 = nn.Dropout(0.4)
        self.dp5 = nn.Dropout(0.5)
        self.dp6 = nn.Dropout(0.6)

        self.out=nn.Linear(512,136)
        
    def forward(self, x):
        x = self.dp1(self.pool(F.relu(self.conv1(x))))
        x = self.dp2(self.pool(F.relu(self.conv2(x))))
        x = self.dp3(self.pool(F.relu(self.conv3(x))))
        x = self.dp4(self.pool(F.relu(self.conv4(x))))
#         print("Aaage: ",x.shape)
        x = x.reshape(x.shape[0],-1)
        # print("Poore: ",x.shape)
        # return 
        x = self.dp5(F.relu(self.fc(x)))
        x = self.dp6(F.relu(self.fc2(x)))
        x = self.out(x)
        return x
