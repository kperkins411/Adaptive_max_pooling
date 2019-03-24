'''
testing a fully trained network on different sized images
'''
import torch
import torchvision
import torchvision.transforms as transforms
import utils
from tensorboardX import SummaryWriter
from train import Net_adaptive_pool,get_transforms_and_classes
from glob import glob
import time

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].
transform, classes =  get_transforms_and_classes()

# get a list of all the files of iterest
where_diff_sized_images_are = "./diff_sized/"

from PIL import Image
def image_loader(image_name, size = None):
    '''
    loads a single image,
    will stay on CPU for eval
    :param image_name:
    :return:
    '''
    image = Image.open(image_name)
    if (size is not None):
        image.thumbnail(size)
    image = transform(image)
    image.unsqueeze_(0) #batch size of 1
    return  image


def evaluate_dir(filelist,net,size=None):
    '''
    evaluate each file in filelist
    compare prediction to ground truth
    print out results
    :param filelist:
    :param net:
    :param size:
    :return:
    '''
    #keep track of number correct
    num_correct = 0;
    results=[]          #dont want print statements screwing with time, save and print after timimg calculated
    start = time.time()
    for fle in filelist:
        # get an image
        input = image_loader(fle,size)
        outputs = net(input)
        _, pred = torch.max(outputs, 1)

        # count correct preds
        predicted = classes[pred]
        if predicted in fle:
            num_correct += 1

        #save to print after timimg calculated
        results.append(f"for image {fle} the network predicted {classes[pred]}")
        # print(f"for image {fle} the network predicted {classes[pred]}")
    total_time = time.time()-start
    for item in results:
        print(item)
    print (f"\nInference took {total_time} seconds")
    print(f"Network accuracy = {num_correct/len(filelist)}\n")

if __name__ == "__main__":
    PATH = "./model_weights.pth"

    # choose which network to use
    # net = Net()
    net = Net_adaptive_pool()

    # load trained weights if there
    try:
        net.load_state_dict(torch.load(PATH))
    except FileNotFoundError:
        print("File " + PATH + " not present")

    # eval on CPU not GPU for a single image
    # its not worth the latency of transferring to GPU
    device = "cpu"
    net.to(device)
    net.eval()  # going to eval

    filelist = glob(where_diff_sized_images_are + "*.jpg")

    print("Evaluate original size")
    evaluate_dir(filelist,net)

    #now lets resize to orig cifar10 expectations to see if results change
    print("Shrink images to 32,32 (orig cifar10 size) and evaluate")
    print("accuracy should fall since we lost information when shrinking images")
    evaluate_dir(filelist, net, size = (32,32))