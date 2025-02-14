import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)

trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(images, labels):
    images = images / 2 + 0.5
    npimages = images.numpy()

    # 创建子图
    fig, axes = plt.subplots(1, len(labels), figsize=(10, 2))
    for i, ax in enumerate(axes):
        ax.imshow(np.transpose(npimages[i], (1, 2, 0)))  
        ax.set_title(classes[labels[i]]) 
        ax.axis('off') 
    plt.show()  

if __name__ == '__main__':
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    imshow(images, labels)
    print(' '.join(f'{classes[labels[j]]}' for j in range(4)))