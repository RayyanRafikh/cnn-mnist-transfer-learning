# Author 1: Mohammed Irfan Ali
# Author 2: Rayyan Muhammad Rafikh
# Date: 4/8/24 


import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from Recognition import MyNetwork

# Define the transformation pipeline for preprocessing input images
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize images to 28x28
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize image data
])

# Specify the directory containing handwritten digit images
images_directory = "handwritten digits"

# Load the pre-trained model
model = MyNetwork()
model.load_state_dict(torch.load("trained_model.pth"))
model.eval()

# Initialize a figure to display handwritten digits and their classifications
fig, axes = plt.subplots(1, 10, figsize=(20, 5))

# Iterate over digit images
for i in range(10):
    # Construct image path
    image_path = os.path.join(images_directory, f"{i}.jpg")

    # Check if image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found.")
        continue

    # Load and preprocess image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    preprocessed_image = transform(image).unsqueeze(0)

    # Perform inference using the model
    with torch.no_grad():
        output = model(preprocessed_image)
        _, predicted = torch.max(output, 1)

    # Display handwritten digit and its classified result
    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(f'Predicted: {predicted.item()}')
    axes[i].axis('off')

plt.show()
