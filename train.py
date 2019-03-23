# -*- coding: utf-8 -*-
"""
Training a classifier
=====================

Generally, when you have to deal with image, text, audio or video data,
you can use standard python packages that load data into a numpy array.
Then you can convert this array into a ``torch.*Tensor``.

-  For images, packages such as Pillow, OpenCV are useful
-  For audio, packages such as scipy and librosa
-  For text, either raw Python or Cython based loading, or NLTK and
   SpaCy are useful

Specifically for vision, we have created a package called
``torchvision``, that has data loaders for common datasets such as
Imagenet, CIFAR10, MNIST, etc. and data transformers for images, viz.,
``torchvision.datasets`` and ``torch.utils.data.DataLoader``.

This provides a huge convenience and avoids writing boilerplate code.

For this tutorial, we will use the CIFAR10 dataset.
It has the classes: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’,
‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’. The images in CIFAR-10 are of
size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.

.. figure:: /_static/img/cifar10.png
   :alt: cifar10

   cifar10


Training an image classifier
----------------------------

We will do the following steps in order:

1. Load and normalizing the CIFAR10 training and test datasets using
   ``torchvision``
2. Define a Convolution Neural Network
3. Define a loss function
4. Train the network on the training data
5. Test the network on the test data

1. Loading and normalizing CIFAR10
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using ``torchvision``, it’s extremely easy to load CIFAR10.
"""
import torch
import torchvision
import torchvision.transforms as transforms
import utils

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


########################################################################
# 2. Define a Convolution Neural Network
#
# take 3-channel images (instead of 1-channel images as it was defined).
# 2 networks presented here
#cifar 10 is 32x32

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

PATH = "./model_weights.pth"
# choose which network to use
# net = Net()
net = Net_adaptive_pool()
try:
    net.load_state_dict(torch.load(PATH))
except FileNotFoundError:
    print("File "+ PATH + " not present")
# net = Net_adaptive_pool_Small()


#using GPU?
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# 3. Define a Loss function and optimizer
# Let's use a Classification Cross-Entropy loss and SGD with momentum.
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

########################################################################
# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize.

def forward(net, criterion, optimizer, dataloader, train=True):

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

        # if i % 2000 == 1999:  # print every 2000 mini-batches
        #     print(".", end="")
    len_dataset = float(len(dataloader.dataset))
    return  (running_loss/len_dataset , running_corrects.double()/len_dataset)

import time
import copy

best_acc=0.0
PATH = "./model_weights.pth"
for epoch in range(10):  # loop over the dataset multiple times
    trn_lss,trn_acc = forward(net,criterion, optimizer, trainloader)

    tst_lss, tst_acc = forward(net, criterion, optimizer, testloader, train=False)

    print('Training loss: %.3f  accuracy: %.3f'%(trn_lss,trn_acc) +
          ' Testing loss: %.3f  accuracy: %.3f' % (tst_lss, tst_acc))

    if(trn_acc > best_acc):
        best_acc = trn_acc
        best_model_wts = copy.deepcopy(net.state_dict())
        torch.save(net.state_dict(), PATH)


print('Finished Training')


########################################################################
# That looks waaay better than chance, which is 10% accuracy (randomly picking
# a class out of 10 classes).
# Seems like the network learnt something.
#
# Hmmm, what are the classes that performed well, and the classes that did
# not perform well:

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


#
# **Exercise:** Try increasing the width of your network (argument 2 of
# the first ``nn.Conv2d``, and argument 1 of the second ``nn.Conv2d`` –
# they need to be the same number), see what kind of speedup you get.
#
# **Goals achieved**:
#
# - Understanding PyTorch's Tensor library and neural networks at a high level.
# - Train a small neural network to classify images
#
# Training on multiple GPUs
# -------------------------
# If you want to see even more MASSIVE speedup using all of your GPUs,
# please check out :doc:`data_parallel_tutorial`.
#
# Where do I go next?
# -------------------
#
# -  :doc:`Train neural nets to play video games </intermediate/reinforcement_q_learning>`
# -  `Train a state-of-the-art ResNet network on imagenet`_
# -  `Train a face generator using Generative Adversarial Networks`_
# -  `Train a word-level language model using Recurrent LSTM networks`_
# -  `More examples`_
# -  `More tutorials`_
# -  `Discuss PyTorch on the Forums`_
# -  `Chat with other users on Slack`_
#
# .. _Train a state-of-the-art ResNet network on imagenet: https://github.com/pytorch/examples/tree/master/imagenet
# .. _Train a face generator using Generative Adversarial Networks: https://github.com/pytorch/examples/tree/master/dcgan
# .. _Train a word-level language model using Recurrent LSTM networks: https://github.com/pytorch/examples/tree/master/word_language_model
# .. _More examples: https://github.com/pytorch/examples
# .. _More tutorials: https://github.com/pytorch/tutorials
# .. _Discuss PyTorch on the Forums: https://discuss.pytorch.org/
# .. _Chat with other users on Slack: http://pytorch.slack.com/messages/beginner/
