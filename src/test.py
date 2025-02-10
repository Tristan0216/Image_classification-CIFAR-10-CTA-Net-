import torch
from torch import optim
from tqdm import tqdm
from mynet import *
from dataload import *

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyNet().to(device)
    model.load_state_dict(torch.load('src\\MyNet\\MyNet.pth', map_location=device))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        with tqdm(testloader, desc=f'Test', unit='batch') as t:
            for images, labels in t:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total * 100
        print(f'Test Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    main()