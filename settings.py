PATH = "./model_weights.pth"
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
NUMB_EPOCHS = 35
PATH_NAP = "./model_weights_NAP.pth"
where_diff_sized_images_are = "./diff_sized/"

#for learning rate finder
MAX_LR = 10.0
MIN_LR = .0001
BATCH_SIZE = 256
FEW_TEST_EPOCHS=3   #sum the errors for this many cycles to see if they diverge
NUMB_LR_DATA_POINTS = 10
DEFAULT_LR = 0.001
DEFAULT_MOMENTUM = 0.9