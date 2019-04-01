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

import time
import copy
import settings as s
import models

import sys
from os.path import abspath
sys.path.insert(0, abspath('..'))
# from .. hyperparam_helper.cyclic_LR_scheduler import  LearningRateFinder
from hyperparam_helper import cyclic_LR_scheduler
# print (sys.path)

def get_transforms():
    '''
    cifar10 is normalized this way
    :return: transforms, and classes
    '''
    return transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def load_data(batch_size =s.BATCH_SIZE):
    '''
    lets get all the data we need
    :return:
    '''
    transform = get_transforms()
    trainset    = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testset     = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader  = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    return trainset, trainloader,testset,testloader

import settings


def forward(net, criterion, optimizer, dataloader,device, writer = None, train=True):
    '''
    forward pass, 1 epoch, both train and eval
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
        labels, loss, preds = forward_onebatch(net, criterion, optimizer,device, train,data)

        # save stats
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)

        # if iS % 100 == 99:  # print every 100 mini-batches
        #     write/print out additional info if needed here

    len_dataset = float(len(dataloader.dataset))
    return  (running_loss/len_dataset , running_corrects.double()/len_dataset)


def forward_onebatch(net, criterion, optimizer, device, train, data):

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

        if (train == True):
            loss.backward()
            optimizer.step()
    return labels, loss, preds


def train_test(net, criterion, optimizer, device, numEpochs = s.NUMB_EPOCHS, path =s.PATH ):
    # tensorboard tracker
    writer = SummaryWriter()

    #assumme the worst
    best_acc = 0.0

    #lets get our data
    trainset, trainloader, testset, testloader = load_data()

    for epoch in range(numEpochs):  # loop over the dataset multiple times
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
            torch.save(net.state_dict(), s.PATH)

    print('Finished Training')
    writer.close()

def find_LR(net, criterion, optimizer, device, batch_size):

    numb_epochs = s.FEW_TEST_EPOCHS

    #lets get our data
    _, trainloader, _, _ = load_data()

     #how many batches
    total_batches = (len(trainloader.dataset)//batch_size)*numb_epochs
    switch_after_this_many_batches = total_batches//s.NUMB_LR_DATA_POINTS

    lr_finder = cyclic_LR_scheduler.LearningRateFinder(optimizer=optimizer,min_lr=settings.MIN_LR, max_lr=settings.MAX_LR, num_batches=s.NUMB_LR_DATA_POINTS)

    total_batches =0
    total_loss = 0.0
    for epoch in range(numb_epochs):
        for _, data in enumerate(trainloader, 0):
            _, loss, _ = forward_onebatch(net, criterion, optimizer, device, train=True, data = data)

            total_loss+=loss
            total_batches+=1
            if(total_batches%switch_after_this_many_batches == 0):
                # only want to change this every few steps
                total_loss=total_loss/switch_after_this_many_batches
                print('Training loss: %.3f ' % (loss) + 'Learning rate is %.4f' % (lr_finder.get_currentLR()))
                total_loss = 0.0
                try:
                    lr_finder.batch_step()
                except StopIteration:
                    break


def check_each_class_accuracy(net, device):
    '''
    check the accuracy of each class, should be way above 10%
    :return:
    '''
    # lets get our data
    _,_,_, testloader = load_data()

    classes = s.classes
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(data)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))



if __name__ == "__main__" :

    # choose which network to use
    # net = models.Net()
    # net = models.Net_adaptive_pool_Small()
    net = models.Net_adaptive_pool()

    # using GPU?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    net.to(device)

    # 3. Define a Loss function and optimizer
    # Let's use a Classification Cross-Entropy loss and SGD with momentum.
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=s.DEFAULT_LR, momentum=s.DEFAULT_MOMENTUM)

    #first lets find the best learning rate
    find_LR(net, criterion, optimizer, device, batch_size=s.BATCH_SIZE)

    #reset optimizer
    optimizer = optim.SGD(net.parameters(), lr=s.DEFAULT_LR, momentum=s.DEFAULT_MOMENTUM)

    # load trained weights if there
    try:
        net.load_state_dict(torch.load(s.PATH))
    except FileNotFoundError as e:
        print("File " + s.PATH + " not present")
    except RuntimeError as e:
        print("Runtime problem, likely file size mismatch")

    train_test(net, criterion, optimizer,device)
    check_each_class_accuracy(net, device)
