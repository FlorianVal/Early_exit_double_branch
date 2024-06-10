import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

from src.BranchyResnet import BranchyResNet18

def load_data(batch_size, num_workers):
    # Define transforms for the training data and testing data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load training data
    trainset = datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)

    # Load test data
    testset = datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    return trainloader, testloader

def initialize_model(learning_rate, momentum, device):
    # Initialize the network
    net = BranchyResNet18().to(device)

    # Define a Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    return net, criterion, optimizer

def train(net, criterion, optimizer, trainloader, device, num_epochs, writer):
    loss = {}

    epoch_tqdm = tqdm(range(num_epochs))  # loop over the dataset multiple times
    # Training loop
    for epoch in epoch_tqdm:
        loss["running_loss"] = 0.0
        batch_tqdm = tqdm(enumerate(trainloader, 0), total=len(trainloader))
        for i, data in batch_tqdm:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            for head, output in enumerate(outputs):
                loss["head_" + str(head)] = criterion(output, labels)
            loss["total_loss"] = sum(loss["head_" + str(head)] for head in range(len(outputs)))

            loss["total_loss"].backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            accuracy = correct / total

            # Log accuracy
            writer.add_scalar('Accuracy/total_accuracy', accuracy, epoch * len(trainloader) + i)

            # print statistics
            loss["running_loss"] += loss["total_loss"].item()
            batch_tqdm.set_postfix(loss="{:.4f}".format(loss["running_loss"] / (i + 1)), accuracy="{:.4f}".format(accuracy))
        
            # Log the losses to TensorBoard
            writer.add_scalar('Loss/total_loss', loss["total_loss"], epoch * len(trainloader) + i)
            for head in range(len(outputs)):
                writer.add_scalar(f'Loss/head_{head}', loss[f"head_{head}"], epoch * len(trainloader) + i)

def main(learning_rate, momentum, batch_size, num_workers, num_epochs):
    
    trainloader, testloader = load_data(batch_size, num_workers) 
    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    net, criterion, optimizer = initialize_model(learning_rate, momentum, device)

    # Create a SummaryWriter for use in TensorBoard
    writer = SummaryWriter()

    train(net, criterion, optimizer, trainloader, device, num_epochs, writer)

    # Close the SummaryWriter
    writer.close()
    print('Finished Training')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a network')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
    args = parser.parse_args()
    main(args.learning_rate, args.momentum, args.batch_size, args.num_workers, args.num_epochs)
