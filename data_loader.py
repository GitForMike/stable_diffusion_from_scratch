from torchvision import datasets, transforms 
from torch.utils.data import ConcatDataset, Subset
import numpy as np


def load_transformed_dataset(img_size, data_dir):
    data_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1] 
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ]
    data_transform = transforms.Compose(data_transforms)
    '''
    train = torchvision.datasets.StanfordCars(root=".", download=True, 
                                         transform=data_transform)

    test = torchvision.datasets.StanfordCars(root=".", download=True, 
                                         transform=data_transform, split='test')
    '''
    train = datasets.CIFAR10(root = data_dir, train = True, download = True, transform = data_transform)
    test = datasets.CIFAR10(root = data_dir, train = False, download = True, transform = data_transform)

    print(type(train))
    #evens = list(range(0, len(train), 2))
    #odds = list(range(0, len(test), 2))
    train = Subset(train, list(range(0, 100)))
    test = Subset(test, list(range(0, 100)))
    #train = Subset(train, evens)
    #test = Subset(test, odds)

    print(type(train))

    return ConcatDataset([train, test])

