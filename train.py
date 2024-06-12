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
    class CustomCriterion(nn.Module):
        def __init__(self):
            super(CustomCriterion, self).__init__()

        def forward(self, outputs, regression, labels):
            ce = torch.functional.F.cross_entropy(outputs, labels, reduction="none")
            regularization = torch.sigmoid(regression.squeeze())
            regul_ce = ce * regularization
            centered_mean_reg = (0.5 - regularization.mean())**2 # force mean of regularization to be 0.5
            max_std_reg = 1 / (regularization.std() + 1e-6) # force std of regularization to be high ( reg should be 1 or 0)
            split_value_reg = torch.exp(-10*(0.5 - regularization)**2).mean() # Value near 0.5 are penalized strongly and are forced to go to 0 or 1
            avoid_zero_reg = (1/(regularization+1e-2)).mean() # Avoid 0 values in the regularization
            #loss = regul_ce.mean() * regul_loss * (1 / (regularization.std() + 1e-6))
            regul_loss = split_value_reg + max_std_reg# * avoid_zero_reg
            loss = regul_ce.mean() + regul_loss
            #print(f"Loss elements : regul_ce.mean() = {regul_ce.mean()}, regul_loss = {regul_loss}")
            #loss = torch.exp(-10*(0.5 - regularization)**2).mean() * (1 / (regularization.std() + 1e-6))
            return ce, regularization, loss
        
    criterion = CustomCriterion()
    final_head_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    return net, criterion, final_head_criterion, optimizer

def train(net, criterion, final_head_criterion, optimizer, trainloader, device, num_epochs, writer):
    metric = {}

    epoch_tqdm = tqdm(range(num_epochs))
    for epoch in epoch_tqdm:
        metric["running_loss"] = 0.0
        batch_tqdm = tqdm(enumerate(trainloader, 0), total=len(trainloader))
        log_interval = len(trainloader) // 10
        for i, data in batch_tqdm:
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs, outputs_reg = net(inputs)
            final_output = outputs[-1]
            head_outputs = outputs[:-1]

            for head, (output, reg) in enumerate(zip(head_outputs, outputs_reg)):
                ce, reg, loss = criterion(output, reg, labels)
                metric["ce_head_" + str(head)] = ce.mean()
                metric["reg_mean_head_" + str(head)] = reg.mean()
                metric["reg_std_head_" + str(head)] = reg.std()
                metric["reg_var_head_" + str(head)] = reg.var()
                metric["loss_head_" + str(head)] = loss
                if i % log_interval == 0:
                    print(reg)
                    print(reg.std())
                    print(1/(reg.std() + 1e-6))
            head += 1
            loss = final_head_criterion(final_output, labels)
            metric["loss_head_" + str(head)] = loss
            metric["total_loss"] = sum(metric["loss_head_" + str(head)] for head in range(len(outputs)))
            metric["total_loss"].backward()
            optimizer.step()

            _, predicted = torch.max(outputs, -1)
            predicted = predicted.squeeze()
            

            total = labels.size(0)
            for head, prediction in enumerate(predicted):
                # prediction is just [batch_size]
                if head != len(predicted) - 1:
                    filtered_prediction = prediction[outputs_reg[head].squeeze() > 0.5]
                    filtered_labels = labels[outputs_reg[head].squeeze() > 0.5]
                    num_not_filtered_data = len(filtered_labels)
                    if num_not_filtered_data != 0:
                        filtered_correct = (filtered_prediction == filtered_labels).sum().item()
                        relative_accuracy = filtered_correct / num_not_filtered_data
                    else:
                        relative_accuracy = 0
                    metric["relative_accuracy_head_" + str(head)] = relative_accuracy
                correct = (prediction == labels).sum().item()
                accuracy = correct / total
                metric["accuracy_head_" + str(head)] = accuracy

            metric["running_loss"] += metric["total_loss"].item()
            batch_tqdm.set_postfix(loss="{:.4f}".format(metric["running_loss"] / (i + 1)), accuracy="{:.4f}".format(accuracy))

            if i % log_interval == 0:
                # Log the losses and accuracy to TensorBoard
                writer.add_scalar('Loss/total_loss', metric["total_loss"], epoch * len(trainloader) + i)
                for key, value in metric.items():
                    if key != "total_loss" and key != "running_loss":
                        head = key.split("_")[-1]
                        base_key = "_".join(key.split("_")[:-2])
                        writer.add_scalar(f'{base_key}/head_{head}', value, epoch * len(trainloader) + i)

def main(learning_rate, momentum, batch_size, num_workers, num_epochs):
    
    trainloader, testloader = load_data(batch_size, num_workers) 
    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    net, criterion, final_head_criterion, optimizer = initialize_model(learning_rate, momentum, device)

    # Create a SummaryWriter for use in TensorBoard
    writer = SummaryWriter()

    # Log the arguments used for this run
    for arg, value in vars(args).items():
        writer.add_text(f'Arguments/{arg}', str(value), 0)

    train(net, criterion, final_head_criterion, optimizer, trainloader, device, num_epochs, writer)

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
