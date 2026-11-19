# Author 1: Mohammed Irfan Ali
# Author 2: Rayyan Muhammad Rafikh
# Date: 4/8/24 


import torch
import torchvision
import matplotlib.pyplot as plt
from model import Network  # Import the custom model

def load_test_data():
    # Load MNIST test dataset with normalization
    test_dataset = torchvision.datasets.MNIST(
        root='/files/',
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
    )
    # Create DataLoader for the test dataset
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)
    return test_loader

def evaluate_model(network, test_loader):
    # Evaluate the model on the test data
    network.eval()  # Set the model to evaluation mode
    predictions = []  # List to store predicted labels
    labels = []  # List to store true labels
    
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)  # Forward pass
            _, predicted = torch.max(output.data, 1)  # Get predicted labels
            
            # Print example predictions for the first batch
            for i in range(len(data)):
                print(f"Example {i+1}:")
                print(f"Output Values: {[f'{value:.2f}' for value in output[i].exp().tolist()]}")
                print(f"Predicted Label: {predicted[i].item()}")
                print(f"Correct Label: {target[i].item()}")
                print()
                
                predictions.append(predicted[i].item())
                labels.append(target[i].item())
            
            break  # Only print predictions for the first batch
    
    return data, predictions, labels

def plot_examples(data, predictions):
    # Plot the first 9 examples with predictions
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    fig.subplots_adjust(hspace=0.5)
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i][0], cmap='gray')  # Display image
        ax.set_title(f"Prediction: {predictions[i]}")  # Display predicted label
        ax.axis('off')  # Hide axes
    
    plt.tight_layout()
    plt.show()

def main():
    # Load the trained model
    network = Network()
    network.load_state_dict(torch.load('trainedModel.pth')['model_state_dict'])
    
    # Load the test data
    test_loader = load_test_data()
    
    # Evaluate the model on the first 10 examples
    data, predictions, labels = evaluate_model(network, test_loader)
    
    # Plot the first 9 examples with predictions
    plot_examples(data, predictions)

if __name__ == "__main__":
    main()
