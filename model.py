# Author 1: Mohammed Irfan Ali
# Author 2: Rayyan Muhammad Rafikh
# Date: 4/8/24 

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torchsummary import summary

# Define batch sizes for training and testing
batch_size_train = 64
batch_size_test = 1000

def data_loaders():
    # Define transformation pipelines for training and testing data
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST datasets for training and testing
    train_dataset = torchvision.datasets.MNIST('/files/', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)

    test_dataset = torchvision.datasets.MNIST('/files/', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test)

    # Retrieve an example batch from the test loader
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    # Print the size of a single test batch
    print("One test batch size: ", example_data.shape)

    return train_loader, test_loader, examples, example_data, example_targets

def plot_test_dataset(example_data, example_targets):
    # Plot the first 6 images from the test dataset
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Image: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1,10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50,10)

    def forward(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train_network(model_path, num_epochs, train_loader, test_loader):
    # Initialize the network
    network = Network()
    summary(network, (1, 28, 28))

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=0.01)

    # Check if the trained model file exists
    if os.path.exists(model_path):
        # Load the trained model weights
        checkpoint = torch.load(model_path)
        network.load_state_dict(checkpoint['model_state_dict'])
        train_accuracies = checkpoint['train_accuracies']
        test_accuracies = checkpoint['test_accuracies']
        print("Trained model and accuracies loaded from file.")
    else:
        # Training loop
        train_accuracies = []
        test_accuracies = []

        for epoch in range(num_epochs):
            # Train the model
            network.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = network(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            # Evaluate on the training set
            network.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in train_loader:
                    output = network(data)
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            train_accuracy = 100 * correct / total
            train_accuracies.append(train_accuracy)

            # Evaluate on the test set
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in test_loader:
                    output = network(data)
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            test_accuracy = 100 * correct / total
            test_accuracies.append(test_accuracy)

            print(f"Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

        # Save the trained model
        torch.save({
        'model_state_dict': network.state_dict(),
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies
        }, model_path)

    return train_accuracies, test_accuracies

def plot_validation(num_epochs, train_accuracies, test_accuracies):
    # Plot training and testing accuracies
    plt.plot(range(1, num_epochs+1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, num_epochs+1), test_accuracies, label='Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig('accuracy_plot.png')
    plt.show()

def main():
    model_path = 'trainedModel.pth'
    num_epochs = 5
    train_loader, test_loader, examples, example_data, example_targets = data_loaders()

    # Plot the first 6 images from the test dataset without shuffling
    plot_test_dataset(example_data, example_targets)

    # Train the model or load a pre-trained one
    train_accuracies, test_accuracies = train_network(model_path, num_epochs, train_loader, test_loader)

    # Plot training and validation accuracies
    plot_validation(num_epochs, train_accuracies, test_accuracies)

if __name__ == "__main__":
    main()
