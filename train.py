import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.profiler
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

from src.BranchyResnet import BranchyResNet18

def load_data(batch_size, num_workers):
    # Define transforms for the training data and testing data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load training data
    trainset = datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers, pin_memory=False)

    # Load test data
    testset = datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers, pin_memory=False)

    return trainloader, testloader

def initialize_model(learning_rate, momentum, device):
    # Initialize the network
    net = BranchyResNet18().to(device)

    # Define a Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    return net, criterion, optimizer

def train(net, criterion, optimizer, trainloader, device, num_epochs, writer):
    metric = {}

    epoch_tqdm = tqdm(range(num_epochs))
    for epoch in epoch_tqdm:
        metric["running_loss"] = 0.0
        batch_tqdm = tqdm(enumerate(trainloader, 0), total=len(trainloader))
        for i, data in batch_tqdm:
            # Start the profiler
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs, outputs_reg = net(inputs)

            for head, output in enumerate(outputs):
                metric["loss_head_" + str(head)] = criterion(output, labels)
            metric["total_loss"] = sum(metric["loss_head_" + str(head)] for head in range(len(outputs)))

            metric["total_loss"].backward()
            optimizer.step()

            _, predicted = torch.max(outputs, -1)
            predicted = predicted.squeeze()
            total = labels.size(0)
            for head, prediction in enumerate(predicted):
                correct = (prediction == labels).sum().item()
                accuracy = correct / total
                metric["accuracy_head_" + str(head)] = accuracy

            metric["running_loss"] += metric["total_loss"].item()
            batch_tqdm.set_postfix(loss="{:.4f}".format(metric["running_loss"] / (i + 1)), accuracy="{:.4f}".format(accuracy))



        # Log the losses and accuracy to TensorBoard
        writer.add_scalar('Loss/total_loss', metric["total_loss"], epoch * len(trainloader) + i)
        for head in range(len(outputs)):
            writer.add_scalar(f'Loss/head_{head}', metric[f"loss_head_{head}"], epoch * len(trainloader) + i)
            writer.add_scalar(f'Accuracy/head_{head}', metric[f"accuracy_head_{head}"], epoch * len(trainloader) + i)

def main(learning_rate, momentum, batch_size, num_workers, num_epochs):
    
    trainloader, testloader = load_data(batch_size, num_workers) 
    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    net, criterion, optimizer = initialize_model(learning_rate, momentum, device)

    # Create a SummaryWriter for use in TensorBoard
    writer = SummaryWriter()

    # Log the arguments used for this run
    for arg, value in vars(args).items():
        writer.add_text(f'Arguments/{arg}', str(value), 0)

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
