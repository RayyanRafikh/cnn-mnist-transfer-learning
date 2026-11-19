# Author 1: Mohammed Irfan Ali
# Author 2: Rayyan Muhammad Rafikh
# Date: 4/8/24 

import torch
import torchvision
import matplotlib.pyplot as plt
import cv2
import numpy as np
from model import Network  # Import the custom model

def visualize_filters(weights):
    # Visualize convolutional filters
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < weights.shape[0]:
            ax.imshow(weights[i, 0])  # Display filter
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"Filter {i+1}")
        else:
            ax.axis('off')  # Hide axes
    plt.tight_layout()
    plt.show()

def apply_filters(image, weights):
    # Apply convolutional filters to the image
    filtered_images = []
    with torch.no_grad():
        for i in range(weights.shape[0]):
            filter_weights = weights[i, 0].numpy()
            filtered_image = cv2.filter2D(image, -1, filter_weights)
            filtered_images.append(filtered_image)
    return filtered_images

def visualize_filtered_images(filtered_images):
    # Visualize filtered images
    print("Number of filtered images:", len(filtered_images))  # Print the number of filtered images
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for i, ax in enumerate(axes.flat):
        if i < len(filtered_images):  # Check if index is within the range of filtered_images
            ax.imshow(filtered_images[i], cmap='gray')  # Display filtered image
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"Filtered Image {i+1}")
        else:
            ax.axis('off')  # Hide axes
    plt.tight_layout()
    plt.show()

def main():
    # Load the trained model
    model = Network()
    model.load_state_dict(torch.load('trainedModel.pth')['model_state_dict'])
    print(model)

    # Get the weights of the first layer (conv1)
    weights = model.conv1.weight.detach()
    print("Filter weights shape:", weights.shape)

    # Visualize the filters
    visualize_filters(weights)

    # Load the first training example image
    train_dataset = torchvision.datasets.MNIST(root='/files/', train=True, download=True,
                                               transform=torchvision.transforms.ToTensor())
    image, _ = train_dataset[0]
    image = image.squeeze().numpy()

    # Apply the filters to the image
    filtered_images = apply_filters(image, weights)
    print("Number of filtered images:", len(filtered_images))

    # Visualize the filtered images
    visualize_filtered_images(filtered_images)

if __name__ == "__main__":
    main()
