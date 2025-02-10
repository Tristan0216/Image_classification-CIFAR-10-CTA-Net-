import torch
from net import *
from dataload import *
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CTANet().to(device)

    # if we have the pre-trained model then fine-tune it
    if os.path.exists('src\\CTA-Net\\CTA-Net.pth'):
        model.load_state_dict(torch.load('src\\CTA-Net\\CTA-Net.pth'))
        print("Pre-training model loaded successfully!")
    else:
        print("No pre-training model found, start training from scratch.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    num_epoches = 20

    min_loss = float('inf')

    loss_values = []
    accuracy_values = []

    save_dir = 'src\\CTA-Net'
    os.makedirs(save_dir, exist_ok=True)
    log_file_path = os.path.join(save_dir, 'training_log.txt')

    with open(log_file_path, 'w') as log_file:
        log_file.write("Epoch\tLoss\tAccuracy\n")
    
    for epoch in range(num_epoches):
        with tqdm(trainloader, desc=f'Epoch {epoch+1}/{num_epoches}', unit='batch') as t:
            model.train()
            epoch_loss = 0
            correct = 0
            total = 0

            for images, labels in t:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(images) 

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            avg_loss = epoch_loss / len(trainloader)
            loss_values.append(avg_loss)

            epoch_accuracy = 100 * correct / total
            accuracy_values.append(epoch_accuracy)

            with open(log_file_path, 'a') as log_file:
                log_file.write(f"\t{epoch+1}\t{avg_loss:.4f}\t{epoch_accuracy:.2f}\n")

            if min_loss > avg_loss:
                min_loss = avg_loss
                print(f'Epoch {epoch+1}/{num_epoches}, AverageLoss: {avg_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
                torch.save(model.state_dict(), 'src\\CTA-Net\\CTA-Net.pth')
                print('Model saved')
            else:
                print(f'Epoch {epoch+1}/{num_epoches}, AverageLoss: {avg_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

    save_dir = 'src\\graph(CTA-Net)'
    os.makedirs(save_dir, exist_ok=True)
    save_path_loss = os.path.join(save_dir, 'loss_curve.png')
    save_path_accuracy = os.path.join(save_dir, 'accuracy_curve.png')

    # save the loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(loss_values)), loss_values, label='Average Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss Value')
    plt.grid(True)
    plt.yticks(np.arange(0.003, 1.8, 0.05))

    plt.savefig(save_path_loss)
    plt.show()
    print(f"The loss curves have been saved to {save_path_loss}")
    
    # save the accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epoches+1), accuracy_values, label="Accuracy", color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid()
    plt.ylim(0, 100) 
    plt.savefig(save_path_accuracy)
    plt.show()
    print(f"The accuracy curves have been saved to {save_path_accuracy}")

    print(f"The training log has been saved to {log_file_path}")
if __name__ == '__main__':
    main()
    