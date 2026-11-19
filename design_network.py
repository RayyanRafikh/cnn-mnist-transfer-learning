# Author 1: Mohammed Irfan Ali
# Author 2: Rayyan Muhammad Rafikh
# Date: 4/8/24 

######################################################################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# Define the network architecture
class Network(nn.Module):
    
    def __init__(self, num_conv_layers, filter_size, num_filters, hidden_nodes, dropout_rate, activation):
        super(Network, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        for _ in range(num_conv_layers):
            self.conv_layers.append(nn.Conv2d(in_channels, num_filters, filter_size, padding=(filter_size - 1) // 2))
            in_channels = num_filters
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_input_size = self.calculate_fc_input_size(num_conv_layers, filter_size, num_filters)
        self.fc = nn.Linear(self.fc_input_size, hidden_nodes)
        self.output = nn.Linear(hidden_nodes, 10)
        self.activation = activation

    def forward(self, x):
        for conv in self.conv_layers:
            x = self.activation(conv(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(-1, self.fc_input_size)
        x = self.activation(self.fc(x))
        x = self.output(x)
        return x

    def calculate_fc_input_size(self, num_conv_layers, filter_size, num_filters):
        input_size = 28
        for _ in range(num_conv_layers):
            input_size = (input_size - filter_size + 2 * ((filter_size - 1) // 2)) // 1 + 1
        input_size //= 2
        return input_size * input_size * num_filters

# Define the experiment parameters
num_conv_layers_options = [2, 3, 4]
filter_size_options = [3, 5]
num_filters_options = [16, 32]
hidden_nodes_options = [128, 256]
dropout_rate_options = [0.2, 0.3, 0.4]
activation_options = [nn.ReLU(), nn.Tanh()]
num_epochs = 10
batch_size = 64

# Load the MNIST Fashion dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the training and evaluation functions
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return running_loss / len(test_loader), accuracy

# Execute the experiment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
results = []

# Linear search strategy
best_num_conv_layers = None
best_filter_size = None
best_num_filters = None
best_hidden_nodes = None
best_dropout_rate = None
best_activation = None
best_accuracy = 0.0

# Optimize the number of convolutional layers
for num_conv_layers in num_conv_layers_options:
    filter_size = filter_size_options[0]
    num_filters = num_filters_options[0]
    hidden_nodes = hidden_nodes_options[0]
    dropout_rate = dropout_rate_options[0]
    activation = activation_options[0]
    model = Network(num_conv_layers, filter_size, num_filters, hidden_nodes, dropout_rate, activation).to(device)
    optimizer = optim.Adam(model.parameters())
    train_losses = []
    test_losses = []
    test_accuracies = []
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
    final_accuracy = test_accuracies[-1]
    if final_accuracy > best_accuracy:
        best_accuracy = final_accuracy
        best_num_conv_layers = num_conv_layers
        torch.save(model.state_dict(), f"best_model_num_conv_layers_{num_conv_layers}.pth")
    results.append({
        'num_conv_layers': num_conv_layers,
        'filter_size': filter_size,
        'num_filters': num_filters,
        'hidden_nodes': hidden_nodes,
        'dropout_rate': dropout_rate,
        'activation': activation.__class__.__name__,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies
    })

# Optimize the filter size
for filter_size in filter_size_options:
    num_conv_layers = best_num_conv_layers
    num_filters = num_filters_options[0]
    hidden_nodes = hidden_nodes_options[0]
    dropout_rate = dropout_rate_options[0]
    activation = activation_options[0]
    model = Network(num_conv_layers, filter_size, num_filters, hidden_nodes, dropout_rate, activation).to(device)
    optimizer = optim.Adam(model.parameters())
    train_losses = []
    test_losses = []
    test_accuracies = []
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
    final_accuracy = test_accuracies[-1]
    if final_accuracy > best_accuracy:
        best_accuracy = final_accuracy
        best_filter_size = filter_size
        torch.save(model.state_dict(), f"best_model_filter_size_{filter_size}.pth")
    results.append({
        'num_conv_layers': num_conv_layers,
        'filter_size': filter_size,
        'num_filters': num_filters,
        'hidden_nodes': hidden_nodes,
        'dropout_rate': dropout_rate,
        'activation': activation.__class__.__name__,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies
    })

# Optimize the number of filters
for num_filters in num_filters_options:
    num_conv_layers = best_num_conv_layers
    filter_size = best_filter_size
    hidden_nodes = hidden_nodes_options[0]
    dropout_rate = dropout_rate_options[0]
    activation = activation_options[0]
    model = Network(num_conv_layers, filter_size, num_filters, hidden_nodes, dropout_rate, activation).to(device)
    optimizer = optim.Adam(model.parameters())
    train_losses = []
    test_losses = []
    test_accuracies = []
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
    final_accuracy = test_accuracies[-1]
    if final_accuracy > best_accuracy:
        best_accuracy = final_accuracy
        best_num_filters = num_filters
        torch.save(model.state_dict(), f"best_model_num_filters_{num_filters}.pth")
    results.append({
        'num_conv_layers': num_conv_layers,
        'filter_size': filter_size,
        'num_filters': num_filters,
        'hidden_nodes': hidden_nodes,
        'dropout_rate': dropout_rate,
        'activation': activation.__class__.__name__,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies
    })

# Optimize the number of hidden nodes
for hidden_nodes in hidden_nodes_options:
    num_conv_layers = best_num_conv_layers
    filter_size = best_filter_size
    num_filters = best_num_filters
    dropout_rate = dropout_rate_options[0]
    activation = activation_options[0]
    model = Network(num_conv_layers, filter_size, num_filters, hidden_nodes, dropout_rate, activation).to(device)
    optimizer = optim.Adam(model.parameters())
    train_losses = []
    test_losses = []
    test_accuracies = []
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
    final_accuracy = test_accuracies[-1]
    if final_accuracy > best_accuracy:
        best_accuracy = final_accuracy
        best_hidden_nodes = hidden_nodes
        torch.save(model.state_dict(), f"best_model_hidden_nodes_{hidden_nodes}.pth")
    results.append({
        'num_conv_layers': num_conv_layers,
        'filter_size': filter_size,
        'num_filters': num_filters,
        'hidden_nodes': hidden_nodes,
        'dropout_rate': dropout_rate,
        'activation': activation.__class__.__name__,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies
    })

# Optimize the dropout rate
for dropout_rate in dropout_rate_options:
    num_conv_layers = best_num_conv_layers
    filter_size = best_filter_size
    num_filters = best_num_filters
    hidden_nodes = best_hidden_nodes
    activation = activation_options[0]
    model = Network(num_conv_layers, filter_size, num_filters, hidden_nodes, dropout_rate, activation).to(device)
    optimizer = optim.Adam(model.parameters())
    train_losses = []
    test_losses = []
    test_accuracies = []
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
    final_accuracy = test_accuracies[-1]
    if final_accuracy > best_accuracy:
        best_accuracy = final_accuracy
        best_dropout_rate = dropout_rate
        torch.save(model.state_dict(), f"best_model_dropout_rate_{dropout_rate}.pth")
    results.append({
        'num_conv_layers': num_conv_layers,
        'filter_size': filter_size,
        'num_filters': num_filters,
        'hidden_nodes': hidden_nodes,
        'dropout_rate': dropout_rate,
        'activation': activation.__class__.__name__,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies
    })

# Analyze and visualize the results
print("Best Configuration:")
print(f"Number of Convolutional Layers: {best_num_conv_layers}")
print(f"Filter Size: {best_filter_size}")
print(f"Number of Filters: {best_num_filters}")
print(f"Hidden Nodes: {best_hidden_nodes}")
print(f"Dropout Rate: {best_dropout_rate}")
print(f"Activation Function: {best_activation.__class__.__name__}")
print(f"Best Accuracy: {best_accuracy:.4f}")

# Plot the training and test losses for each configuration
plt.figure(figsize=(12, 8))
for i, result in enumerate(results):
    plt.subplot(2, 3, i+1)
    plt.plot(result['train_losses'], label='Training Loss')
    plt.plot(result['test_losses'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f"Conv Layers: {result['num_conv_layers']}, Filter Size: {result['filter_size']}, Num Filters: {result['num_filters']}, Hidden Nodes: {result['hidden_nodes']}, Dropout: {result['dropout_rate']}, Activation: {result['activation']}")

plt.tight_layout()
plt.show()

# Plot the test accuracies for each configuration
plt.figure(figsize=(12, 8))
for i, result in enumerate(results):
    plt.subplot(2, 3, i+1)
    plt.plot(result['test_accuracies'])
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title(f"Conv Layers: {result['num_conv_layers']}, Filter Size: {result['filter_size']}, Num Filters: {result['num_filters']}, Hidden Nodes: {result['hidden_nodes']}, Dropout: {result['dropout_rate']}, Activation: {result['activation']}")

plt.tight_layout()
plt.show()
