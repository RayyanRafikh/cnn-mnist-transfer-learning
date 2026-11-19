# Author 1: Mohammed Irfan Ali
# Author 2: Rayyan Muhammad Rafikh
# Date: 4/8/24 

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Import the MNIST network architecture from Task 1
from model import Network

# Greek data set transform
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36 / 128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)

# Set the path to the directory containing the Greek letter folders
training_set_path = 'greek_train'

# Load the Greek dataset
dataset = torchvision.datasets.ImageFolder(
    training_set_path,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        GreekTransform(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
)

# Split the dataset into training and validation sets
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# Create data loaders for training and validation sets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=5, shuffle=False)

# Load the pre-trained MNIST model
network = Network()
checkpoint = torch.load('trainedModel.pth')
network.load_state_dict(checkpoint['model_state_dict'])

# Freeze the network weights
for param in network.parameters():
    param.requires_grad = False

# Replace the last layer with a new Linear layer with three nodes
num_classes = len(dataset.classes)
network.fc2 = nn.Linear(50, num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.fc2.parameters(), lr=0.001)

# Training loop
num_epochs = 10
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Training
    network.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = network(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)

    # Validation
    network.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = network(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    epoch_loss = running_loss / len(val_loader)
    val_losses.append(epoch_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

# Plot the training and validation losses
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss vs. Epoch')
plt.legend()
plt.show()

# Print the modified network architecture
print(network)

# Testing with your own examples
def predict_greek_letter(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        GreekTransform(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = transform(image).unsqueeze(0)

    # Make the prediction
    network.eval()
    with torch.no_grad():
        output = network(image)
        _, predicted = torch.max(output.data, 1)
        predicted_label = dataset.classes[predicted.item()]
    return predicted_label

# Specify the paths to your own example images
example_images = ['alpha1.jpg', 'alpha2.jpg','alpha3.jpg','alpha4.jpg','alpha5.jpg','alpha6.jpg','alpha7.jpg','beta1.jpg', 'beta2.jpg','beta3.jpg','beta4.jpg','beta5.jpg','beta6.jpg','beta7.jpg','gamma1.jpg','gamma2.jpg','gamma3.jpg','gamma4.jpg','gamma5.jpg','gamma6.jpg','gamm7.jpg']

# Iterate over the example images and make predictions
for image_path in example_images:
    predicted_label = predict_greek_letter(image_path)
    print(f"Image: {image_path}, Predicted Label: {predicted_label}")