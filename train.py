"""
Training an image classifier
this project replaces the layer before the fully connected layers
in a convolutional network with an adaptive max pool layer
to allow using any sized images for input

Without it the first fully connected layer dictats the exact size
of an input image

We will do the following steps in order:

1. Load and normalizing the CIFAR10 training and test datasets using
   ``torchvision``
2. Define a Convolution Neural Network
3. Define a loss function
4. Train the network on the training data
5. Test the network on the test data

"""
import torch
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import time
import copy

def get_transforms_and_classes():
    '''
    cifar10 is normalized this way
    :return: transforms, and classes
    '''
    return (transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck'))

def load_data():
    '''
    lets get all the data we need
    :return:
    '''
    transform,_ = get_transforms_and_classes()
    trainset    = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    testset     = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader  = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
    return trainset, trainloader,testset,testloader

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
    '''
    def __init__(self):
        super(Net_adaptive_pool, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, stride=1,
                 padding=0, dilation=1, groups=1, bias=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 10, 3)
        self.adaptive_mp2d=nn.AdaptiveMaxPool2d((5,5))
        self.fc1 = nn.Linear(10 * 5 * 5, 50)
        self.fc2 = nn.Linear(50, 30)
        self.fc3 = nn.Linear(30, 10)

    def forward(self, x):       #x.shape = 3*32*32
        x = self.pool(F.relu(self.conv1(x)))    # output = 10*14*14
        x = self.pool(F.relu(self.conv2(x)))  # output = 6*5*5
        x = self.adaptive_mp2d(F.relu(self.conv2(x)))    # output = 10*5*5
        x = x.view(-1, 10 * 5 * 5)      #the second 5x5 is the size of the output of the above
                                        # this is what determines input size
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net_adaptive_pool_Small(nn.Module):
    def __init__(self):
        super(Net_adaptive_pool_Small, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1,
                               padding=0, dilation=1, groups=1, bias=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 10, 5)
        self.adaptive_mp2d = nn.AdaptiveMaxPool2d((1,1))
        self.fc1 = nn.Linear(10 , 10)
        self.fc2 = nn.Linear(10, 10)


    def forward(self, x):  # x.shape = 3*32*32
        x = self.pool(F.relu(self.conv1(x)))  # output = 6*14*14
        # x = self.pool(F.relu(self.conv2(x)))  # output = 10*5*5
        x = self.adaptive_mp2d(F.relu(self.conv2(x)))  # output = 10*5*5
        x = x.view(-1, 10 )  # the second 5x5 is the size of the output of the above
        # this is what determines input size
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def forward(net, criterion, optimizer, dataloader,device, writer = None, train=True):
    '''
    forward pass, both train and eval
    :param net:
    :param criterion:
    :param optimizer:
    :param dataloader:
    :param train:
    :return:
    '''

    running_loss = 0.0
    running_corrects = 0

    if (train==True):
        net.train()     #going to train
    else:
        net.eval()      #going to eval

    for i, data in enumerate(dataloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(train == True):

            # forward + backward + optimize
            outputs = net(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            if(train==True):
                loss.backward()
                optimizer.step()

        # save stats
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)

        # if i % 100 == 99:  # print every 100 mini-batches
        #     write/print out additional info if needed here

    len_dataset = float(len(dataloader.dataset))
    return  (running_loss/len_dataset , running_corrects.double()/len_dataset)

def train_test(net, criterion, optimizer, device):
    # tensorboard tracker
    writer = SummaryWriter()
    NUMB_EPOCHS = 35
    PATH = "./model_weights.pth"
    best_acc = 0.0

    #lets get our data
    trainset, trainloader, testset, testloader = load_data()

    for epoch in range(NUMB_EPOCHS):  # loop over the dataset multiple times
        trn_lss, trn_acc = forward(net, criterion, optimizer, trainloader,device,writer=writer)

        tst_lss, tst_acc = forward(net, criterion, optimizer, testloader,device, writer = writer, train=False)

        writer.add_scalars('losses', {"trn_lss": trn_lss,
                                      "tst_lss": tst_lss}, epoch)

        writer.add_scalars('accuracy', {"trn_acc": trn_acc,
                                        "tst_acc": tst_acc}, epoch)

        print('Training loss: %.3f  accuracy: %.3f' % (trn_lss, trn_acc) +
              ' Testing loss: %.3f  accuracy: %.3f' % (tst_lss, tst_acc))

        if (tst_acc > best_acc):
            best_acc = tst_acc
            best_model_wts = copy.deepcopy(net.state_dict())
            torch.save(net.state_dict(), PATH)

    print('Finished Training')
    writer.close()

def check_each_class_accuracy(net, device):
    '''
    check the accuracy of each class, should be way above 10%
    :return:
    '''
    # lets get our data
    _,_,_, testloader = load_data()

    _,classes = get_transforms_and_classes()
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

if __name__ == "__main__":
    PATH = "./model_weights.pth"

    # choose which network to use
    # net = Net()
    # net = Net_adaptive_pool_Small()
    net = Net_adaptive_pool()

    # load trained weights if there
    try:
        net.load_state_dict(torch.load(PATH))
    except FileNotFoundError:
        print("File " + PATH + " not present")

    # using GPU?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # 3. Define a Loss function and optimizer
    # Let's use a Classification Cross-Entropy loss and SGD with momentum.
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train_test(net, criterion, optimizer,device)
    check_each_class_accuracy(net, device)
