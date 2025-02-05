import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _load_cifar(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes

def loader_to_device(loader, device):
    return [(x.to(device), y.to(device)) for x, y in loader]

def get_data_loaders(batch_size, classes=False):
    trainloader, testloader, classes = _load_cifar(batch_size)
    if classes:
        return trainloader, testloader, classes
    else:    
        return trainloader, testloader

def get_device(pos=None):
    if torch.cuda.is_available():
        if pos is None:
            raise ValueError('Please specify the GPU position')
        else:
            device = torch.device(f'cuda:{pos}')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    return device
    

def get_resnet18(device):
    from resnet import ResNet18
    model = ResNet18()
    model = model.to(device)
    return model

class Logger():
    def __init__(self):
        self.loss = []
        self.accuracy = []
        self.index = []

    def append(self, loss, accuracy, index):
        self.loss.append(loss)
        self.accuracy.append(accuracy)
        self.index.append(index)
    
    def to_dict(self):
        return {
            'loss': self.loss,
            'accuracy': self.accuracy,
            'index': self.index
        }
