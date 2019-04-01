
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    '''
    only takes 32x32 images (fully connected layers dictate this)
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1,
                 padding=0, dilation=1, groups=1, bias=True)    #6x28x28 conv filters
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 10, 5)                        #16x10x10 (after the application of maxpool= 16*5*5
        self.fc1 = nn.Linear(10 * 5 * 5, 50)
        self.fc2 = nn.Linear(50, 30)
        self.fc3 = nn.Linear(30, 10)

    def forward(self, x):       #x.shape = 3*32*32
        x = self.pool(F.relu(self.conv1(x)))    # output = 6*14*14
        x = self.pool(F.relu(self.conv2(x)))    # output = 16*5*5
        x = x.view(-1, 10 * 5 * 5)      #the second 5x5 is the size of the output of the above
                                        # this is what determines input size
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net_adaptive_pool(nn.Module):
    '''
    takes any sized images because of the adaptive max_pool layer
    wnt 2 fully connected conv layers and then start to pool
    '''
    def __init__(self):
        super(Net_adaptive_pool, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 10, 3)
        self.adaptive_mp2d=nn.AdaptiveMaxPool2d((5,5))
        self.fc1 = nn.Linear(10 * 5 * 5, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward1(self,x):
        x = F.relu(self.conv1(x))  # output = 10*32*32
        x = F.relu(self.conv2(x))  # output = 10*30*30
        x = self.pool(F.relu(self.conv2(x)))  # output = 10*14,14
        return x

    def forward(self, x):       #x.shape = 3*32*32
        x=self.forward1(x)
        x = self.adaptive_mp2d(F.relu(self.conv2(x)))    # output = 10*12*12
        x = x.view(-1, 10 * 5 * 5)      #the second 5x5 is the size of the output of the above
                                        # this is what determines input size
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Net_adaptive_pool_Small(Net_adaptive_pool):
    def __init__(self):
        super(Net_adaptive_pool_Small, self).__init__()
        self.adaptive_mp2d = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Linear(10 , 10)

    def forward(self, x):  # x.shape = 3*32*32
        x = self.forward1(x)

        x = self.adaptive_mp2d(F.relu(self.conv2(x)))  # output = 10*1*1
        x = x.view(-1, 10 )  # the second is the size of the output of the above
        x = F.relu(self.fc1(x))
        x = self.fc1(x)
        return x